"""
News-based trading filter.

Blocks trading during high-impact economic events
to avoid volatility spikes and unpredictable price movements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable

from ..communication.alerting import AlertManager, AlertSeverity, get_alert_manager
from ..utils.logger import get_logger
from .currency_impact import CurrencyImpactMapper
from .news_provider import (
    AggregatedNewsProvider,
    EconomicEvent,
    EventImpact,
    ForexFactoryProvider,
    InvestingComProvider,
)

logger = get_logger("news.filter")


# =============================================================================
# Configuration
# =============================================================================

class BlackoutAction(Enum):
    """Action to take during news blackout."""
    
    BLOCK_ENTRY = "block_entry"
    """Block new entries but keep existing positions."""
    
    CLOSE_POSITIONS = "close_positions"
    """Close existing positions and block new entries."""
    
    TIGHTEN_STOPS = "tighten_stops"
    """Tighten stop losses but allow trading."""


@dataclass
class NewsFilterConfig:
    """Configuration for news filter."""
    
    enabled: bool = True
    """Whether news filter is active."""
    
    blackout_before_minutes: int = 30
    """Minutes before event to start blackout."""
    
    blackout_after_minutes: int = 15
    """Minutes after event to end blackout."""
    
    min_impact: EventImpact = EventImpact.HIGH
    """Minimum impact level to trigger blackout."""
    
    default_action: BlackoutAction = BlackoutAction.BLOCK_ENTRY
    """Default action during blackout."""
    
    refresh_interval_minutes: int = 60
    """How often to refresh calendar data."""
    
    cache_dir: Path = field(default_factory=lambda: Path("data/cache/news"))
    """Directory for caching news data."""
    
    # Per-symbol overrides
    symbol_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    """
    Symbol-specific configuration overrides.
    
    Example:
        {
            "USDJPY": {
                "min_impact": "medium",
                "action": "close_positions",
                "blackout_before_minutes": 45,
            }
        }
    """
    
    def get_symbol_config(self, symbol: str) -> dict[str, Any]:
        """Get effective config for a symbol."""
        base_config = {
            "blackout_before_minutes": self.blackout_before_minutes,
            "blackout_after_minutes": self.blackout_after_minutes,
            "min_impact": self.min_impact,
            "action": self.default_action,
        }
        
        if symbol in self.symbol_overrides:
            overrides = self.symbol_overrides[symbol]
            
            if "blackout_before_minutes" in overrides:
                base_config["blackout_before_minutes"] = overrides["blackout_before_minutes"]
            if "blackout_after_minutes" in overrides:
                base_config["blackout_after_minutes"] = overrides["blackout_after_minutes"]
            if "min_impact" in overrides:
                impact_str = overrides["min_impact"]
                if isinstance(impact_str, str):
                    base_config["min_impact"] = EventImpact(impact_str)
                else:
                    base_config["min_impact"] = impact_str
            if "action" in overrides:
                action_str = overrides["action"]
                if isinstance(action_str, str):
                    base_config["action"] = BlackoutAction(action_str)
                else:
                    base_config["action"] = action_str
        
        return base_config


# =============================================================================
# Blackout Result
# =============================================================================

@dataclass
class BlackoutStatus:
    """Status of news blackout for a symbol."""
    
    symbol: str
    """Trading symbol."""
    
    is_blocked: bool
    """Whether trading is blocked."""
    
    action: BlackoutAction | None
    """Action to take if blocked."""
    
    blocking_events: list[EconomicEvent] = field(default_factory=list)
    """Events causing the blackout."""
    
    next_event: EconomicEvent | None = None
    """Next upcoming event affecting this symbol."""
    
    blackout_ends: datetime | None = None
    """When the current blackout ends."""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "is_blocked": self.is_blocked,
            "action": self.action.value if self.action else None,
            "blocking_events": [
                {
                    "title": e.title,
                    "currency": e.currency,
                    "impact": e.impact.value,
                    "datetime_utc": e.datetime_utc.isoformat(),
                }
                for e in self.blocking_events
            ],
            "next_event": {
                "title": self.next_event.title,
                "currency": self.next_event.currency,
                "impact": self.next_event.impact.value,
                "datetime_utc": self.next_event.datetime_utc.isoformat(),
            } if self.next_event else None,
            "blackout_ends": self.blackout_ends.isoformat() if self.blackout_ends else None,
        }


# =============================================================================
# News Filter
# =============================================================================

class NewsFilter:
    """
    Filters trading based on economic calendar events.
    
    Features:
    - Automatic calendar refresh from multiple sources
    - Per-symbol blackout tracking
    - Configurable blackout windows and actions
    - Alert notifications when blackouts start/end
    """
    
    def __init__(
        self,
        config: NewsFilterConfig | None = None,
        symbols: list[str] | None = None,
        alert_manager: AlertManager | None = None,
        on_blackout_start: Callable[[str, list[EconomicEvent]], None] | None = None,
        on_blackout_end: Callable[[str], None] | None = None,
    ):
        """
        Initialize news filter.
        
        Args:
            config: Filter configuration
            symbols: Trading symbols to track
            alert_manager: Alert manager for notifications
            on_blackout_start: Callback when blackout starts
            on_blackout_end: Callback when blackout ends
        """
        self.config = config or NewsFilterConfig()
        self.alert_manager = alert_manager or get_alert_manager()
        
        # Callbacks
        self.on_blackout_start = on_blackout_start
        self.on_blackout_end = on_blackout_end
        
        # Initialize providers
        self.provider = AggregatedNewsProvider(
            providers=[
                ForexFactoryProvider(cache_dir=self.config.cache_dir),
                InvestingComProvider(cache_dir=self.config.cache_dir),
            ]
        )
        
        # Currency mapper
        self.currency_mapper = CurrencyImpactMapper(symbols)
        
        # Event cache
        self._events: list[EconomicEvent] = []
        self._events_lock = Lock()
        self._last_refresh: datetime | None = None
        
        # Blackout state tracking
        self._active_blackouts: dict[str, BlackoutStatus] = {}
        self._blackout_lock = Lock()
        
        # Background refresh
        self._refresh_thread: Thread | None = None
        self._stop_refresh = False
        
        # Initial load
        if self.config.enabled:
            self._refresh_events()
        
        logger.info(
            f"NewsFilter initialized: "
            f"enabled={self.config.enabled}, "
            f"min_impact={self.config.min_impact.value}, "
            f"blackout_before={self.config.blackout_before_minutes}m, "
            f"blackout_after={self.config.blackout_after_minutes}m"
        )
    
    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to track."""
        self.currency_mapper.add_symbol(symbol)
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from tracking."""
        self.currency_mapper.remove_symbol(symbol)
        with self._blackout_lock:
            if symbol in self._active_blackouts:
                del self._active_blackouts[symbol]
    
    def check_blackout(self, symbol: str) -> BlackoutStatus:
        """
        Check if trading is blocked for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            BlackoutStatus with current state
        """
        if not self.config.enabled:
            return BlackoutStatus(
                symbol=symbol,
                is_blocked=False,
                action=None,
            )
        
        # Ensure symbol is tracked
        self.currency_mapper.add_symbol(symbol)
        
        # Get symbol-specific config
        sym_config = self.config.get_symbol_config(symbol)
        min_impact = sym_config["min_impact"]
        blackout_before = timedelta(minutes=sym_config["blackout_before_minutes"])
        blackout_after = timedelta(minutes=sym_config["blackout_after_minutes"])
        action = sym_config["action"]
        
        # Impact priority
        impact_priority = {
            EventImpact.HIGH: 3,
            EventImpact.MEDIUM: 2,
            EventImpact.LOW: 1,
            EventImpact.HOLIDAY: 0,
        }
        min_priority = impact_priority[min_impact]
        
        now = datetime.now(timezone.utc)
        blocking_events = []
        next_event = None
        blackout_ends = None
        
        with self._events_lock:
            for event in self._events:
                # Check impact level
                if impact_priority[event.impact] < min_priority:
                    continue
                
                # Check if event affects this symbol
                if not self.currency_mapper.is_pair_affected(
                    symbol, event.country, event.title
                ):
                    continue
                
                # Check if in blackout window
                event_start = event.datetime_utc - blackout_before
                event_end = event.datetime_utc + blackout_after
                
                if event_start <= now <= event_end:
                    blocking_events.append(event)
                    if blackout_ends is None or event_end > blackout_ends:
                        blackout_ends = event_end
                
                # Track next upcoming event
                elif now < event_start:
                    if next_event is None or event.datetime_utc < next_event.datetime_utc:
                        next_event = event
        
        is_blocked = len(blocking_events) > 0
        
        # Create status
        status = BlackoutStatus(
            symbol=symbol,
            is_blocked=is_blocked,
            action=action if is_blocked else None,
            blocking_events=blocking_events,
            next_event=next_event,
            blackout_ends=blackout_ends,
        )
        
        # Track state changes and trigger callbacks
        self._track_blackout_state(symbol, status)
        
        return status
    
    def _track_blackout_state(self, symbol: str, status: BlackoutStatus) -> None:
        """Track blackout state changes and trigger callbacks/alerts."""
        with self._blackout_lock:
            was_blocked = symbol in self._active_blackouts
            
            if status.is_blocked and not was_blocked:
                # Blackout started
                self._active_blackouts[symbol] = status
                
                # Send alert
                event_titles = ", ".join(e.title for e in status.blocking_events[:3])
                self.alert_manager.warning(
                    title=f"News Blackout Started: {symbol}",
                    message=(
                        f"Trading blocked due to high-impact news: {event_titles}. "
                        f"Blackout ends at {status.blackout_ends.strftime('%H:%M UTC') if status.blackout_ends else 'unknown'}."
                    ),
                    source="news_filter",
                    metadata={
                        "symbol": symbol,
                        "events": [e.title for e in status.blocking_events],
                        "action": status.action.value if status.action else None,
                    },
                )
                
                # Callback
                if self.on_blackout_start:
                    try:
                        self.on_blackout_start(symbol, status.blocking_events)
                    except Exception as e:
                        logger.error(f"Error in blackout start callback: {e}")
                
                logger.info(
                    f"Blackout started for {symbol}: "
                    f"{len(status.blocking_events)} events, "
                    f"action={status.action.value if status.action else 'none'}"
                )
            
            elif not status.is_blocked and was_blocked:
                # Blackout ended
                del self._active_blackouts[symbol]
                
                # Send alert
                self.alert_manager.info(
                    title=f"News Blackout Ended: {symbol}",
                    message=f"Trading resumed for {symbol}.",
                    source="news_filter",
                    metadata={"symbol": symbol},
                )
                
                # Callback
                if self.on_blackout_end:
                    try:
                        self.on_blackout_end(symbol)
                    except Exception as e:
                        logger.error(f"Error in blackout end callback: {e}")
                
                logger.info(f"Blackout ended for {symbol}")
            
            elif status.is_blocked:
                # Update existing blackout
                self._active_blackouts[symbol] = status
    
    def get_upcoming_events(
        self,
        symbol: str | None = None,
        hours_ahead: int = 24,
        min_impact: EventImpact | None = None,
    ) -> list[EconomicEvent]:
        """
        Get upcoming events.
        
        Args:
            symbol: Filter by symbol (optional)
            hours_ahead: Hours to look ahead
            min_impact: Minimum impact level
            
        Returns:
            List of upcoming events
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        
        if min_impact is None:
            min_impact = self.config.min_impact
        
        impact_priority = {
            EventImpact.HIGH: 3,
            EventImpact.MEDIUM: 2,
            EventImpact.LOW: 1,
            EventImpact.HOLIDAY: 0,
        }
        min_priority = impact_priority[min_impact]
        
        events = []
        
        with self._events_lock:
            for event in self._events:
                # Time filter
                if not (now <= event.datetime_utc <= cutoff):
                    continue
                
                # Impact filter
                if impact_priority[event.impact] < min_priority:
                    continue
                
                # Symbol filter
                if symbol:
                    self.currency_mapper.add_symbol(symbol)
                    if not self.currency_mapper.is_pair_affected(
                        symbol, event.country, event.title
                    ):
                        continue
                
                events.append(event)
        
        return sorted(events, key=lambda e: e.datetime_utc)
    
    def get_all_blackouts(self) -> dict[str, BlackoutStatus]:
        """Get current blackout status for all tracked symbols."""
        result = {}
        
        for symbol in self.currency_mapper.tracked_pairs.keys():
            result[symbol] = self.check_blackout(symbol)
        
        return result
    
    def refresh_events(self) -> int:
        """
        Manually refresh event data.
        
        Returns:
            Number of events fetched
        """
        return self._refresh_events()
    
    def _refresh_events(self) -> int:
        """Refresh event cache from providers."""
        try:
            # Fetch events for next 7 days
            start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            end = start + timedelta(days=7)
            
            events = self.provider.fetch_events(
                start_date=start,
                end_date=end,
                min_impact=EventImpact.LOW,  # Fetch all, filter later
            )
            
            with self._events_lock:
                self._events = events
                self._last_refresh = datetime.now(timezone.utc)
            
            logger.info(f"Refreshed news calendar: {len(events)} events")
            return len(events)
            
        except Exception as e:
            logger.error(f"Failed to refresh news calendar: {e}")
            return 0
    
    def start_auto_refresh(self) -> None:
        """Start background refresh thread."""
        if self._refresh_thread is not None:
            return
        
        self._stop_refresh = False
        self._refresh_thread = Thread(
            target=self._refresh_loop,
            daemon=True,
            name="NewsFilterRefresh",
        )
        self._refresh_thread.start()
        logger.info("Started news filter auto-refresh")
    
    def stop_auto_refresh(self) -> None:
        """Stop background refresh thread."""
        self._stop_refresh = True
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=5)
            self._refresh_thread = None
        logger.info("Stopped news filter auto-refresh")
    
    def _refresh_loop(self) -> None:
        """Background refresh loop."""
        interval = self.config.refresh_interval_minutes * 60
        
        while not self._stop_refresh:
            try:
                # Check if refresh needed
                if self._last_refresh is None:
                    should_refresh = True
                else:
                    elapsed = (datetime.now(timezone.utc) - self._last_refresh).total_seconds()
                    should_refresh = elapsed >= interval
                
                if should_refresh:
                    self._refresh_events()
                
                # Sleep in small increments to allow quick shutdown
                for _ in range(60):  # Check every second
                    if self._stop_refresh:
                        break
                    import time
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in refresh loop: {e}")
    
    def get_status(self) -> dict[str, Any]:
        """Get filter status summary."""
        with self._events_lock:
            event_count = len(self._events)
            last_refresh = self._last_refresh
        
        with self._blackout_lock:
            active_blackouts = dict(self._active_blackouts)
        
        # Count events by impact
        impact_counts = {impact: 0 for impact in EventImpact}
        with self._events_lock:
            for event in self._events:
                impact_counts[event.impact] += 1
        
        return {
            "enabled": self.config.enabled,
            "event_count": event_count,
            "last_refresh": last_refresh.isoformat() if last_refresh else None,
            "active_blackouts": len(active_blackouts),
            "blackout_symbols": list(active_blackouts.keys()),
            "tracked_symbols": list(self.currency_mapper.tracked_pairs.keys()),
            "events_by_impact": {k.value: v for k, v in impact_counts.items()},
            "config": {
                "min_impact": self.config.min_impact.value,
                "blackout_before_minutes": self.config.blackout_before_minutes,
                "blackout_after_minutes": self.config.blackout_after_minutes,
                "default_action": self.config.default_action.value,
            },
        }
