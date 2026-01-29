"""
Alert and notification system for trading errors.

Provides a unified interface for sending alerts through various channels:
- Console/Log output (always enabled)
- File logging (always enabled)
- Future: Email, Slack, Discord, etc.

Includes rate limiting to prevent alert storms.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

from ..utils.logger import get_logger

logger = get_logger("communication.alerting")


# =============================================================================
# Alert Severity
# =============================================================================

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Alert Data
# =============================================================================

@dataclass
class Alert:
    """Alert data structure."""
    
    severity: AlertSeverity
    """Severity level."""
    
    title: str
    """Short alert title."""
    
    message: str
    """Detailed alert message."""
    
    source: str
    """Source component that generated the alert."""
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When the alert was generated."""
    
    error_type: str | None = None
    """Optional error type/code."""
    
    exception: Exception | None = None
    """Optional exception that caused the alert."""
    
    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional context data."""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "exception": str(self.exception) if self.exception else None,
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        """Format alert as string."""
        return (
            f"[{self.severity.value.upper()}] {self.title}\n"
            f"Source: {self.source}\n"
            f"Time: {self.timestamp.isoformat()}\n"
            f"Message: {self.message}"
        )


# =============================================================================
# Alert Handler Interface
# =============================================================================

class AlertHandler(ABC):
    """Abstract base class for alert handlers."""
    
    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """
        Send an alert.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully
        """
        pass
    
    @abstractmethod
    def supports_severity(self, severity: AlertSeverity) -> bool:
        """
        Check if handler supports a severity level.
        
        Args:
            severity: Severity to check
            
        Returns:
            True if supported
        """
        pass


# =============================================================================
# Console/Log Handler
# =============================================================================

class LogAlertHandler(AlertHandler):
    """Handler that logs alerts to the logging system."""
    
    def __init__(
        self,
        min_severity: AlertSeverity = AlertSeverity.INFO,
    ):
        """
        Initialize log alert handler.
        
        Args:
            min_severity: Minimum severity to handle
        """
        self.min_severity = min_severity
        self._severity_order = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL,
        ]
    
    def send(self, alert: Alert) -> bool:
        """Log the alert."""
        if not self.supports_severity(alert.severity):
            return False
        
        log_message = (
            f"ALERT [{alert.severity.value.upper()}] - {alert.title}: "
            f"{alert.message} (source: {alert.source})"
        )
        
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(log_message)
        elif alert.severity == AlertSeverity.ERROR:
            logger.error(log_message)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        return True
    
    def supports_severity(self, severity: AlertSeverity) -> bool:
        """Check if severity meets minimum threshold."""
        min_idx = self._severity_order.index(self.min_severity)
        severity_idx = self._severity_order.index(severity)
        return severity_idx >= min_idx


# =============================================================================
# File Alert Handler
# =============================================================================

class FileAlertHandler(AlertHandler):
    """Handler that writes alerts to a JSON file."""
    
    def __init__(
        self,
        file_path: str | Path,
        min_severity: AlertSeverity = AlertSeverity.WARNING,
        max_alerts: int = 1000,
    ):
        """
        Initialize file alert handler.
        
        Args:
            file_path: Path to alert file
            min_severity: Minimum severity to handle
            max_alerts: Maximum alerts to keep in file
        """
        self.file_path = Path(file_path)
        self.min_severity = min_severity
        self.max_alerts = max_alerts
        self._lock = Lock()
        
        self._severity_order = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL,
        ]
        
        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def send(self, alert: Alert) -> bool:
        """Write alert to file."""
        if not self.supports_severity(alert.severity):
            return False
        
        with self._lock:
            try:
                # Load existing alerts
                alerts = []
                if self.file_path.exists():
                    try:
                        with open(self.file_path, "r", encoding="utf-8") as f:
                            alerts = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        alerts = []
                
                # Add new alert
                alerts.append(alert.to_dict())
                
                # Trim to max size
                if len(alerts) > self.max_alerts:
                    alerts = alerts[-self.max_alerts:]
                
                # Write back
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump(alerts, f, indent=2, ensure_ascii=False)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to write alert to file: {e}")
                return False
    
    def supports_severity(self, severity: AlertSeverity) -> bool:
        """Check if severity meets minimum threshold."""
        min_idx = self._severity_order.index(self.min_severity)
        severity_idx = self._severity_order.index(severity)
        return severity_idx >= min_idx


# =============================================================================
# Discord Alert Handler
# =============================================================================

class DiscordAlertHandler(AlertHandler):
    """
    Handler that sends alerts to Discord via webhook.
    
    Formats alerts as rich embeds with color-coded severity.
    """
    
    # Discord embed colors by severity
    SEVERITY_COLORS = {
        AlertSeverity.INFO: 0x3498DB,      # Blue
        AlertSeverity.WARNING: 0xF39C12,   # Orange
        AlertSeverity.ERROR: 0xE74C3C,     # Red
        AlertSeverity.CRITICAL: 0x8E44AD,  # Purple
    }
    
    def __init__(
        self,
        webhook_url: str,
        min_severity: AlertSeverity = AlertSeverity.WARNING,
        username: str = "Trading Bot",
        avatar_url: str | None = None,
    ):
        """
        Initialize Discord alert handler.
        
        Args:
            webhook_url: Discord webhook URL
            min_severity: Minimum severity to send
            username: Bot username to display
            avatar_url: Optional avatar URL
        """
        self.webhook_url = webhook_url
        self.min_severity = min_severity
        self.username = username
        self.avatar_url = avatar_url
        
        self._severity_order = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL,
        ]
    
    def send(self, alert: Alert) -> bool:
        """Send alert to Discord."""
        if not self.supports_severity(alert.severity):
            return False
        
        try:
            import requests
            
            # Build embed
            embed = {
                "title": f"{alert.severity.value.upper()}: {alert.title}",
                "description": alert.message,
                "color": self.SEVERITY_COLORS.get(alert.severity, 0x95A5A6),
                "timestamp": alert.timestamp.isoformat(),
                "fields": [
                    {
                        "name": "Source",
                        "value": alert.source,
                        "inline": True,
                    },
                ],
                "footer": {
                    "text": "Trading Bot Alert",
                },
            }
            
            # Add error type if present
            if alert.error_type:
                embed["fields"].append({
                    "name": "Error Type",
                    "value": alert.error_type,
                    "inline": True,
                })
            
            # Add metadata fields
            for key, value in list(alert.metadata.items())[:5]:
                embed["fields"].append({
                    "name": key.replace("_", " ").title(),
                    "value": str(value)[:100],
                    "inline": True,
                })
            
            # Build payload
            payload = {
                "username": self.username,
                "embeds": [embed],
            }
            
            if self.avatar_url:
                payload["avatar_url"] = self.avatar_url
            
            # Send request
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            
            if response.status_code in (200, 204):
                return True
            else:
                logger.warning(
                    f"Discord webhook returned {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return False
                
        except ImportError:
            logger.warning("requests library not available for Discord alerts")
            return False
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False
    
    def supports_severity(self, severity: AlertSeverity) -> bool:
        """Check if severity meets minimum threshold."""
        min_idx = self._severity_order.index(self.min_severity)
        severity_idx = self._severity_order.index(severity)
        return severity_idx >= min_idx


# =============================================================================
# Rate Limiter
# =============================================================================

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    max_alerts_per_minute: int = 10
    """Maximum alerts per minute per source."""
    
    cooldown_seconds: float = 60.0
    """Cooldown period after hitting limit."""
    
    dedup_window_seconds: float = 300.0
    """Window for deduplicating identical alerts."""


class AlertRateLimiter:
    """
    Rate limiter for alerts.
    
    Prevents alert storms by:
    - Limiting alerts per source per minute
    - Deduplicating identical alerts within a window
    """
    
    def __init__(self, config: RateLimitConfig | None = None):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        
        # Track alerts per source: {source: [(timestamp, count)]}
        self._source_counts: dict[str, list[float]] = {}
        
        # Track recent alerts for deduplication: {hash: timestamp}
        self._recent_alerts: dict[str, float] = {}
        
        # Cooldown tracking: {source: cooldown_until}
        self._cooldowns: dict[str, float] = {}
        
        self._lock = Lock()
    
    def should_allow(self, alert: Alert) -> bool:
        """
        Check if alert should be allowed.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert should be sent
        """
        with self._lock:
            now = time.time()
            
            # Check cooldown
            if alert.source in self._cooldowns:
                if now < self._cooldowns[alert.source]:
                    return False
                else:
                    del self._cooldowns[alert.source]
            
            # Check deduplication
            alert_hash = self._hash_alert(alert)
            if alert_hash in self._recent_alerts:
                if now - self._recent_alerts[alert_hash] < self.config.dedup_window_seconds:
                    return False
            
            # Check rate limit
            if alert.source not in self._source_counts:
                self._source_counts[alert.source] = []
            
            # Clean old entries
            cutoff = now - 60.0
            self._source_counts[alert.source] = [
                ts for ts in self._source_counts[alert.source]
                if ts > cutoff
            ]
            
            # Check limit
            if len(self._source_counts[alert.source]) >= self.config.max_alerts_per_minute:
                # Enter cooldown
                self._cooldowns[alert.source] = now + self.config.cooldown_seconds
                logger.warning(
                    f"Alert rate limit exceeded for source '{alert.source}', "
                    f"entering {self.config.cooldown_seconds}s cooldown"
                )
                return False
            
            # Record alert
            self._source_counts[alert.source].append(now)
            self._recent_alerts[alert_hash] = now
            
            # Cleanup old dedup entries
            self._cleanup_dedup_cache(now)
            
            return True
    
    def _hash_alert(self, alert: Alert) -> str:
        """Generate hash for alert deduplication."""
        return f"{alert.source}:{alert.severity.value}:{alert.title}:{alert.error_type}"
    
    def _cleanup_dedup_cache(self, now: float) -> None:
        """Remove old entries from dedup cache."""
        cutoff = now - self.config.dedup_window_seconds
        self._recent_alerts = {
            k: v for k, v in self._recent_alerts.items()
            if v > cutoff
        }


# =============================================================================
# Alert Manager
# =============================================================================

class AlertManager:
    """
    Central alert management system.
    
    Features:
    - Multiple alert handlers (log, file, etc.)
    - Rate limiting to prevent alert storms
    - Severity-based routing
    """
    
    def __init__(
        self,
        rate_limit_config: RateLimitConfig | None = None,
        alert_file_path: str | Path | None = None,
    ):
        """
        Initialize alert manager.
        
        Args:
            rate_limit_config: Rate limiting configuration
            alert_file_path: Path for file alerts (optional)
        """
        self.handlers: list[AlertHandler] = []
        self.rate_limiter = AlertRateLimiter(rate_limit_config)
        
        # Always add log handler
        self.handlers.append(LogAlertHandler(min_severity=AlertSeverity.INFO))
        
        # Add file handler if path provided
        if alert_file_path is not None:
            self.handlers.append(
                FileAlertHandler(
                    file_path=alert_file_path,
                    min_severity=AlertSeverity.WARNING,
                )
            )
        
        self._stats = {
            "total_alerts": 0,
            "rate_limited": 0,
            "sent": 0,
            "failed": 0,
        }
        self._stats_lock = Lock()
        
        logger.info(
            f"AlertManager initialized with {len(self.handlers)} handlers"
        )
    
    def add_handler(self, handler: AlertHandler) -> None:
        """Add an alert handler."""
        self.handlers.append(handler)
        logger.debug(f"Added alert handler: {type(handler).__name__}")
    
    def send_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        error_type: str | None = None,
        exception: Exception | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Send an alert through all handlers.
        
        Args:
            severity: Alert severity
            title: Short title
            message: Detailed message
            source: Source component
            error_type: Optional error type
            exception: Optional exception
            metadata: Optional additional data
            
        Returns:
            True if at least one handler sent successfully
        """
        alert = Alert(
            severity=severity,
            title=title,
            message=message,
            source=source,
            error_type=error_type,
            exception=exception,
            metadata=metadata or {},
        )
        
        with self._stats_lock:
            self._stats["total_alerts"] += 1
        
        # Check rate limiting
        if not self.rate_limiter.should_allow(alert):
            with self._stats_lock:
                self._stats["rate_limited"] += 1
            return False
        
        # Send through all handlers
        sent_count = 0
        for handler in self.handlers:
            if handler.supports_severity(severity):
                try:
                    if handler.send(alert):
                        sent_count += 1
                except Exception as e:
                    logger.error(f"Handler {type(handler).__name__} failed: {e}")
        
        with self._stats_lock:
            if sent_count > 0:
                self._stats["sent"] += 1
            else:
                self._stats["failed"] += 1
        
        return sent_count > 0
    
    # Convenience methods
    def info(
        self,
        title: str,
        message: str,
        source: str,
        **kwargs: Any,
    ) -> bool:
        """Send info alert."""
        return self.send_alert(AlertSeverity.INFO, title, message, source, **kwargs)
    
    def warning(
        self,
        title: str,
        message: str,
        source: str,
        **kwargs: Any,
    ) -> bool:
        """Send warning alert."""
        return self.send_alert(AlertSeverity.WARNING, title, message, source, **kwargs)
    
    def error(
        self,
        title: str,
        message: str,
        source: str,
        **kwargs: Any,
    ) -> bool:
        """Send error alert."""
        return self.send_alert(AlertSeverity.ERROR, title, message, source, **kwargs)
    
    def critical(
        self,
        title: str,
        message: str,
        source: str,
        **kwargs: Any,
    ) -> bool:
        """Send critical alert."""
        return self.send_alert(AlertSeverity.CRITICAL, title, message, source, **kwargs)
    
    def get_stats(self) -> dict[str, int]:
        """Get alert statistics."""
        with self._stats_lock:
            return dict(self._stats)


# =============================================================================
# Global Alert Manager Instance
# =============================================================================

_global_alert_manager: AlertManager | None = None


def get_alert_manager() -> AlertManager:
    """
    Get or create global alert manager.
    
    Automatically configures Discord handler if configured in trading.yaml.
    """
    global _global_alert_manager
    
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager(
            alert_file_path="logs/alerts/alerts.json"
        )
        
        # Try to add Discord handler from config
        try:
            from ..utils.config import get_config
            config = get_config()
            discord_config = config.trading.communication.discord
            
            if discord_config.enabled and discord_config.webhook_url:
                # Map severity string to enum
                severity_map = {
                    "info": AlertSeverity.INFO,
                    "warning": AlertSeverity.WARNING,
                    "error": AlertSeverity.ERROR,
                    "critical": AlertSeverity.CRITICAL,
                }
                min_severity = severity_map.get(
                    discord_config.min_severity.lower(),
                    AlertSeverity.WARNING
                )
                
                discord_handler = DiscordAlertHandler(
                    webhook_url=discord_config.webhook_url,
                    min_severity=min_severity,
                    username=discord_config.username,
                )
                _global_alert_manager.add_handler(discord_handler)
                logger.info("Discord alert handler configured")
        except Exception as e:
            logger.debug(f"Discord handler not configured: {e}")
    
    return _global_alert_manager


def send_alert(
    severity: AlertSeverity,
    title: str,
    message: str,
    source: str,
    **kwargs: Any,
) -> bool:
    """Convenience function to send alert via global manager."""
    return get_alert_manager().send_alert(
        severity=severity,
        title=title,
        message=message,
        source=source,
        **kwargs,
    )
