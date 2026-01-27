"""
Communication module for MT5 EA <-> Python prediction service.

Provides ZeroMQ-based request/response communication with support for
trading signals, health checks, and heartbeat monitoring.

Includes error handling and resilience patterns:
- RetryManager: Exponential backoff retry logic
- CircuitBreaker: Prevents cascading failures
- ConnectionMonitor: Tracks connection health
- AlertManager: Error notification with rate limiting
"""

from .alerting import (
    Alert,
    AlertHandler,
    AlertManager,
    AlertRateLimiter,
    AlertSeverity,
    FileAlertHandler,
    LogAlertHandler,
    RateLimitConfig,
    get_alert_manager,
    send_alert,
)
from .error_handling import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
    ConnectionHealth,
    ConnectionMonitor,
    ResilientExecutor,
    RetryConfig,
    RetryManager,
)
from .signal_service import SignalService
from .zmq_server import ZMQServer

__all__ = [
    # Core services
    "SignalService",
    "ZMQServer",
    # Error handling
    "RetryManager",
    "RetryConfig",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitState",
    "ConnectionMonitor",
    "ConnectionHealth",
    "ResilientExecutor",
    # Alerting
    "Alert",
    "AlertSeverity",
    "AlertHandler",
    "AlertManager",
    "AlertRateLimiter",
    "RateLimitConfig",
    "LogAlertHandler",
    "FileAlertHandler",
    "get_alert_manager",
    "send_alert",
]
