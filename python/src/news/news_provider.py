"""
Economic calendar data providers.

Fetches news events from Forex Factory and Investing.com
for use in trading filters.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ..utils.logger import get_logger

logger = get_logger("news.provider")


# =============================================================================
# Data Models
# =============================================================================

class EventImpact(Enum):
    """Impact level of economic event."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    HOLIDAY = "holiday"
    
    @classmethod
    def from_string(cls, value: str) -> EventImpact:
        """Parse impact from string."""
        value_lower = value.lower().strip()
        
        if value_lower in ("high", "red", "3", "bull3", "gry"):
            return cls.HIGH
        elif value_lower in ("medium", "orange", "2", "bull2", "yel"):
            return cls.MEDIUM
        elif value_lower in ("low", "yellow", "1", "bull1", "ora"):
            return cls.LOW
        elif value_lower in ("holiday", "gray", "0"):
            return cls.HOLIDAY
        
        return cls.LOW


@dataclass
class EconomicEvent:
    """Represents an economic calendar event."""
    
    id: str
    """Unique identifier for the event."""
    
    title: str
    """Event title/name."""
    
    country: str
    """Country code (ISO 3166-1 alpha-2)."""
    
    currency: str
    """Currency code affected."""
    
    datetime_utc: datetime
    """Event time in UTC."""
    
    impact: EventImpact
    """Impact level."""
    
    actual: str | None = None
    """Actual value (if released)."""
    
    forecast: str | None = None
    """Forecast value."""
    
    previous: str | None = None
    """Previous value."""
    
    source: str = ""
    """Data source identifier."""
    
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""
    
    def __hash__(self) -> int:
        """Hash based on title, country, and time."""
        return hash((self.title, self.country, self.datetime_utc.isoformat()))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "country": self.country,
            "currency": self.currency,
            "datetime_utc": self.datetime_utc.isoformat(),
            "impact": self.impact.value,
            "actual": self.actual,
            "forecast": self.forecast,
            "previous": self.previous,
            "source": self.source,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EconomicEvent:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            country=data["country"],
            currency=data["currency"],
            datetime_utc=datetime.fromisoformat(data["datetime_utc"]),
            impact=EventImpact(data["impact"]),
            actual=data.get("actual"),
            forecast=data.get("forecast"),
            previous=data.get("previous"),
            source=data.get("source", ""),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Base Provider
# =============================================================================

class NewsProvider(ABC):
    """Abstract base class for economic calendar providers."""
    
    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl_hours: int = 1,
        request_timeout: int = 30,
    ):
        """
        Initialize provider.
        
        Args:
            cache_dir: Directory for caching data
            cache_ttl_hours: Cache time-to-live in hours
            request_timeout: HTTP request timeout in seconds
        """
        self.cache_dir = cache_dir or Path("data/cache/news")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.request_timeout = request_timeout
        
        # Rate limiting
        self._last_request_time: float = 0
        self._min_request_interval: float = 2.0  # seconds
        
        # Session with headers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })
    
    @abstractmethod
    def fetch_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[EconomicEvent]:
        """
        Fetch economic events.
        
        Args:
            start_date: Start of date range (defaults to today)
            end_date: End of date range (defaults to end of week)
            
        Returns:
            List of economic events
        """
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Provider source name."""
        pass
    
    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a key."""
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{self.source_name}_{key_hash}.json"
    
    def _load_from_cache(self, cache_key: str) -> list[EconomicEvent] | None:
        """Load events from cache if valid."""
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            cached_time = datetime.fromisoformat(data["cached_at"])
            if datetime.now(timezone.utc) - cached_time > self.cache_ttl:
                return None
            
            return [EconomicEvent.from_dict(e) for e in data["events"]]
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, events: list[EconomicEvent]) -> None:
        """Save events to cache."""
        cache_path = self._get_cache_path(cache_key)
        
        try:
            data = {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "source": self.source_name,
                "events": [e.to_dict() for e in events],
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")


# =============================================================================
# Investing.com Provider
# =============================================================================

class InvestingComProvider(NewsProvider):
    """
    Investing.com economic calendar provider.
    
    Uses Investing.com's AJAX API for calendar data.
    """
    
    BASE_URL = "https://www.investing.com"
    CALENDAR_API = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
    
    # Country ID mapping for Investing.com
    COUNTRY_IDS = {
        "US": 5,
        "EU": 72,
        "GB": 4,
        "JP": 35,
        "AU": 25,
        "NZ": 43,
        "CA": 6,
        "CH": 12,
        "CN": 37,
        "DE": 17,
        "FR": 22,
    }
    
    COUNTRY_TO_CURRENCY = {
        5: ("US", "USD"),
        72: ("EU", "EUR"),
        4: ("GB", "GBP"),
        35: ("JP", "JPY"),
        25: ("AU", "AUD"),
        43: ("NZ", "NZD"),
        6: ("CA", "CAD"),
        12: ("CH", "CHF"),
        37: ("CN", "CNY"),
        17: ("DE", "EUR"),
        22: ("FR", "EUR"),
    }
    
    @property
    def source_name(self) -> str:
        return "investing"
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        
        # Additional headers for Investing.com
        self.session.headers.update({
            "X-Requested-With": "XMLHttpRequest",
            "Referer": "https://www.investing.com/economic-calendar/",
        })
    
    def fetch_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[EconomicEvent]:
        """
        Fetch events from Investing.com via AJAX API.
        
        Args:
            start_date: Start date (defaults to today)
            end_date: End date (defaults to 7 days from start)
            
        Returns:
            List of economic events
        """
        if start_date is None:
            start_date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        
        if end_date is None:
            end_date = start_date + timedelta(days=7)
        
        # Check cache
        cache_key = f"{start_date.date()}_{end_date.date()}"
        cached = self._load_from_cache(cache_key)
        if cached:
            logger.debug(f"Loaded {len(cached)} events from cache")
            return cached
        
        events = []
        
        try:
            self._rate_limit()
            
            # First get the main page to establish cookies
            logger.info("Fetching Investing.com calendar via AJAX API")
            self.session.get(
                f"{self.BASE_URL}/economic-calendar/",
                timeout=self.request_timeout,
            )
            
            # Prepare API request data
            country_ids = [str(v) for v in self.COUNTRY_IDS.values()]
            post_data = {
                "dateFrom": start_date.strftime("%Y-%m-%d"),
                "dateTo": end_date.strftime("%Y-%m-%d"),
                "country[]": country_ids,
                "importance[]": ["2", "3"],  # Medium and High importance
                "limit_from": "0",
            }
            
            # Fetch via AJAX API
            api_response = self.session.post(
                self.CALENDAR_API,
                data=post_data,
                timeout=self.request_timeout,
            )
            api_response.raise_for_status()
            
            # Parse the JSON response containing HTML
            response_data = api_response.json()
            html_data = response_data.get("data", "")
            
            # Parse the HTML table rows
            events = self._parse_ajax_response(html_data)
            
            # Filter by date range
            events = [
                e for e in events
                if start_date <= e.datetime_utc <= end_date
            ]
            
            logger.info(f"Fetched {len(events)} events from Investing.com")
            
            # Cache results
            self._save_to_cache(cache_key, events)
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Investing.com calendar: {e}")
        except Exception as e:
            logger.error(f"Error parsing Investing.com calendar: {e}")
        
        return events
    
    def _parse_calendar_page(self, html: str) -> list[EconomicEvent]:
        """Parse Investing.com calendar HTML."""
        soup = BeautifulSoup(html, "html.parser")
        events = []
        
        # Find calendar table
        calendar_table = soup.find("table", id="economicCalendarData")
        if not calendar_table:
            logger.warning("Calendar table not found")
            return events
        
        current_date = None
        
        for row in calendar_table.find_all("tr"):
            try:
                # Check for date header row
                date_header = row.find("td", class_="theDay")
                if date_header:
                    date_text = date_header.get_text(strip=True)
                    current_date = self._parse_inv_date(date_text)
                    continue
                
                # Skip non-event rows
                if not row.get("event_attr_id"):
                    continue
                
                # Get time
                time_cell = row.find("td", class_="time")
                event_time = None
                if time_cell:
                    time_text = time_cell.get_text(strip=True)
                    if time_text and time_text not in ("", "All Day", "Tentative"):
                        event_time = self._parse_inv_time(time_text)
                
                # Get country flag
                flag_cell = row.find("td", class_="flagCur")
                country = "US"
                currency = "USD"
                if flag_cell:
                    flag_span = flag_cell.find("span", class_="cemark")
                    if flag_span:
                        # Extract country from flag class
                        flag_class = " ".join(flag_span.get("class", []))
                        for code in self.COUNTRY_IDS.keys():
                            if code.lower() in flag_class.lower():
                                country = code
                                currency = self.COUNTRY_TO_CURRENCY.get(
                                    self.COUNTRY_IDS[code], (code, code)
                                )[1]
                                break
                
                # Get impact
                impact = EventImpact.LOW
                impact_cell = row.find("td", class_="sentiment")
                if impact_cell:
                    bulls = impact_cell.find_all("i", class_="grayFullBullishIcon")
                    bull_count = len(bulls) if bulls else 0
                    if bull_count >= 3:
                        impact = EventImpact.HIGH
                    elif bull_count == 2:
                        impact = EventImpact.MEDIUM
                
                # Get event title
                event_cell = row.find("td", class_="event")
                title = ""
                if event_cell:
                    title_link = event_cell.find("a")
                    title = title_link.get_text(strip=True) if title_link else ""
                
                if not title:
                    continue
                
                # Get actual/forecast/previous
                actual_cell = row.find("td", class_="act")
                actual = actual_cell.get_text(strip=True) if actual_cell else None
                
                forecast_cell = row.find("td", class_="fore")
                forecast = forecast_cell.get_text(strip=True) if forecast_cell else None
                
                previous_cell = row.find("td", class_="prev")
                previous = previous_cell.get_text(strip=True) if previous_cell else None
                
                # Build datetime
                if current_date:
                    if event_time:
                        event_dt = current_date.replace(
                            hour=event_time[0],
                            minute=event_time[1],
                        )
                    else:
                        event_dt = current_date
                else:
                    event_dt = datetime.now(timezone.utc)
                
                # Create event
                event_id = hashlib.md5(
                    f"{title}_{country}_{event_dt.isoformat()}".encode()
                ).hexdigest()[:16]
                
                event = EconomicEvent(
                    id=event_id,
                    title=title,
                    country=country,
                    currency=currency,
                    datetime_utc=event_dt,
                    impact=impact,
                    actual=actual if actual else None,
                    forecast=forecast if forecast else None,
                    previous=previous if previous else None,
                    source=self.source_name,
                )
                events.append(event)
                
            except Exception as e:
                logger.debug(f"Error parsing calendar row: {e}")
                continue
        
        return events
    
    def _parse_ajax_response(self, html: str) -> list[EconomicEvent]:
        """
        Parse Investing.com AJAX API response HTML.
        
        The API returns HTML table rows directly.
        """
        soup = BeautifulSoup(html, "html.parser")
        events = []
        
        current_date = None
        
        # Find all table rows in the response
        for row in soup.find_all("tr"):
            try:
                # Check for date header row
                date_cell = row.find("td", class_="theDay")
                if date_cell:
                    date_text = date_cell.get_text(strip=True)
                    current_date = self._parse_inv_date(date_text)
                    continue
                
                # Check if this is an event row
                event_attr_id = row.get("event_attr_id")
                if not event_attr_id and not row.get("id", "").startswith("eventRowId"):
                    continue
                
                # Get time
                time_cell = row.find("td", class_="time")
                event_time = None
                if time_cell:
                    time_text = time_cell.get_text(strip=True)
                    if time_text and time_text not in ("", "All Day", "Tentative", "Tent."):
                        event_time = self._parse_inv_time(time_text)
                
                # Get currency and country from flag cell
                flag_cell = row.find("td", class_="flagCur")
                country = "US"
                currency = "USD"
                if flag_cell:
                    # Get currency text
                    currency_text = flag_cell.get_text(strip=True)
                    if currency_text:
                        currency = currency_text.strip()
                    
                    # Get country from flag span
                    flag_span = flag_cell.find("span", class_="ceFlags")
                    if flag_span:
                        title = flag_span.get("title", "")
                        # Map country name to code
                        country_map = {
                            "United States": "US",
                            "United Kingdom": "GB",
                            "European Union": "EU",
                            "Eurozone": "EU",
                            "Japan": "JP",
                            "Australia": "AU",
                            "New Zealand": "NZ",
                            "Canada": "CA",
                            "Switzerland": "CH",
                            "China": "CN",
                            "Germany": "DE",
                            "France": "FR",
                        }
                        country = country_map.get(title, "US")
                
                # Get impact (sentiment/importance)
                impact = EventImpact.LOW
                impact_cell = row.find("td", class_="sentiment")
                if impact_cell:
                    # Count bull icons (filled ones indicate importance)
                    bulls = impact_cell.find_all("i", class_="grayFullBullishIcon")
                    bull_count = len(bulls) if bulls else 0
                    if bull_count >= 3:
                        impact = EventImpact.HIGH
                    elif bull_count == 2:
                        impact = EventImpact.MEDIUM
                
                # Get event title
                event_cell = row.find("td", class_="event")
                title = ""
                if event_cell:
                    title_link = event_cell.find("a")
                    if title_link:
                        title = title_link.get_text(strip=True)
                    else:
                        title = event_cell.get_text(strip=True)
                
                if not title:
                    continue
                
                # Get actual/forecast/previous values
                actual_cell = row.find("td", class_="act")
                actual = actual_cell.get_text(strip=True) if actual_cell else None
                
                forecast_cell = row.find("td", class_="fore")
                forecast = forecast_cell.get_text(strip=True) if forecast_cell else None
                
                previous_cell = row.find("td", class_="prev")
                previous = previous_cell.get_text(strip=True) if previous_cell else None
                
                # Build datetime
                if current_date is not None:
                    if event_time:
                        event_dt = current_date.replace(
                            hour=event_time[0],
                            minute=event_time[1],
                        )
                    else:
                        event_dt = current_date
                else:
                    # Try to get datetime from data attribute
                    dt_attr = row.get("data-event-datetime", "")
                    if dt_attr:
                        try:
                            event_dt = datetime.strptime(dt_attr, "%Y/%m/%d %H:%M:%S")
                            event_dt = event_dt.replace(tzinfo=timezone.utc)
                        except ValueError:
                            event_dt = datetime.now(timezone.utc)
                    else:
                        event_dt = datetime.now(timezone.utc)
                
                # Create event ID
                event_id = hashlib.md5(
                    f"{title}_{currency}_{event_dt.isoformat()}".encode()
                ).hexdigest()[:16]
                
                event = EconomicEvent(
                    id=event_id,
                    title=title,
                    country=country,
                    currency=currency,
                    datetime_utc=event_dt,
                    impact=impact,
                    actual=actual if actual else None,
                    forecast=forecast if forecast else None,
                    previous=previous if previous else None,
                    source=self.source_name,
                )
                events.append(event)
                
            except Exception as e:
                logger.debug(f"Error parsing AJAX row: {e}")
                continue
        
        return events
    
    def _parse_inv_date(self, date_text: str) -> datetime:
        """Parse Investing.com date format."""
        # Formats: "Wednesday, January 15, 2025"
        try:
            # Try common formats
            for fmt in [
                "%A, %B %d, %Y",
                "%B %d, %Y",
                "%d %B %Y",
            ]:
                try:
                    dt = datetime.strptime(date_text, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            # Fallback: extract date parts
            parts = re.findall(r"(\w+)\s+(\d+),?\s+(\d{4})", date_text)
            if parts:
                month_str, day_str, year_str = parts[0]
                month_map = {
                    "january": 1, "february": 2, "march": 3, "april": 4,
                    "may": 5, "june": 6, "july": 7, "august": 8,
                    "september": 9, "october": 10, "november": 11, "december": 12,
                }
                month = month_map.get(month_str.lower(), 1)
                return datetime(int(year_str), month, int(day_str), tzinfo=timezone.utc)
            
            raise ValueError(f"Unknown date format: {date_text}")
            
        except Exception as e:
            logger.warning(f"Failed to parse date '{date_text}': {e}")
            return datetime.now(timezone.utc)
    
    def _parse_inv_time(self, time_text: str) -> tuple[int, int]:
        """Parse Investing.com time format."""
        # Formats: "08:30", "14:00"
        try:
            time_text = time_text.strip()
            
            if ":" in time_text:
                parts = time_text.split(":")
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
                return (hour, minute)
            
            raise ValueError(f"Unknown time format: {time_text}")
            
        except Exception as e:
            logger.warning(f"Failed to parse time '{time_text}': {e}")
            return (0, 0)


# =============================================================================
# Aggregated Provider
# =============================================================================

class AggregatedNewsProvider:
    """
    Aggregates events from multiple providers.
    
    Merges and deduplicates events from Forex Factory and Investing.com.
    """
    
    def __init__(
        self,
        providers: list[NewsProvider] | None = None,
        cache_dir: Path | None = None,
    ):
        """
        Initialize aggregated provider.
        
        Args:
            providers: List of providers (defaults to Investing.com)
            cache_dir: Cache directory
        """
        if providers is None:
            cache_dir = cache_dir or Path("data/cache/news")
            providers = [
                InvestingComProvider(cache_dir=cache_dir),
            ]
        
        self.providers = providers
    
    def fetch_events(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        min_impact: EventImpact = EventImpact.LOW,
    ) -> list[EconomicEvent]:
        """
        Fetch and merge events from all providers.
        
        Args:
            start_date: Start date
            end_date: End date
            min_impact: Minimum impact level to include
            
        Returns:
            Merged and deduplicated list of events
        """
        all_events: dict[str, EconomicEvent] = {}
        
        # Impact priority for deduplication (prefer higher impact)
        impact_priority = {
            EventImpact.HIGH: 3,
            EventImpact.MEDIUM: 2,
            EventImpact.LOW: 1,
            EventImpact.HOLIDAY: 0,
        }
        
        min_priority = impact_priority[min_impact]
        
        for provider in self.providers:
            try:
                events = provider.fetch_events(start_date, end_date)
                
                for event in events:
                    # Skip low impact if filtered
                    if impact_priority[event.impact] < min_priority:
                        continue
                    
                    # Create dedup key
                    key = self._make_dedup_key(event)
                    
                    # Keep higher impact version
                    if key in all_events:
                        existing = all_events[key]
                        if impact_priority[event.impact] > impact_priority[existing.impact]:
                            all_events[key] = event
                    else:
                        all_events[key] = event
                        
            except Exception as e:
                logger.error(f"Failed to fetch from {provider.source_name}: {e}")
        
        # Sort by datetime
        events = sorted(all_events.values(), key=lambda e: e.datetime_utc)
        
        logger.info(f"Aggregated {len(events)} events from {len(self.providers)} providers")
        
        return events
    
    def _make_dedup_key(self, event: EconomicEvent) -> str:
        """Create deduplication key for an event."""
        # Normalize title for comparison
        title_normalized = re.sub(r"[^a-z0-9]", "", event.title.lower())
        
        # Round time to nearest 15 minutes for fuzzy matching
        dt_rounded = event.datetime_utc.replace(
            minute=(event.datetime_utc.minute // 15) * 15,
            second=0,
            microsecond=0,
        )
        
        return f"{event.currency}_{title_normalized[:30]}_{dt_rounded.date()}"
