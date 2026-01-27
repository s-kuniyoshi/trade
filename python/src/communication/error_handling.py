"""
Error handling and resilience patterns for MT5 communication.

Implements:
- RetryManager: Exponential backoff retry logic
- CircuitBreaker: Prevents cascading failures
- ConnectionMonitor: Tracks connection health
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any, Callable, TypeVar

from ..utils.logger import get_logger

logger = get_logger("communication.error_handling")


# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar("T")


# =============================================================================
# Retry Configuration
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_retries: int = 3
    """Maximum number of retry attempts."""
    
    base_delay_ms: int = 100
    """Base delay between retries in milliseconds."""
    
    max_delay_ms: int = 5000
    """Maximum delay between retries in milliseconds."""
    
    exponential_base: float = 2.0
    """Base for exponential backoff calculation."""
    
    jitter_factor: float = 0.1
    """Random jitter factor (0.0 - 1.0) to prevent thundering herd."""


# =============================================================================
# Retry Manager
# =============================================================================

class RetryManager:
    """
    Manages retry logic with exponential backoff.
    
    Supports:
    - Configurable max retries
    - Exponential backoff with jitter
    - Retry-specific exception filtering
    """
    
    def __init__(
        self,
        config: RetryConfig | None = None,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
    ):
        """
        Initialize retry manager.
        
        Args:
            config: Retry configuration (uses defaults if None)
            retryable_exceptions: Tuple of exception types to retry
        """
        self.config = config or RetryConfig()
        self.retryable_exceptions = retryable_exceptions or (Exception,)
        
        # Statistics
        self.total_retries = 0
        self.successful_retries = 0
        self.failed_retries = 0
        self.stats_lock = Lock()
        
        logger.debug(
            f"RetryManager initialized: max_retries={self.config.max_retries}, "
            f"base_delay={self.config.base_delay_ms}ms"
        )
    
    def execute(
        self,
        operation: Callable[[], T],
        operation_name: str = "operation",
    ) -> T:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Callable to execute
            operation_name: Name for logging purposes
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: Last exception if all retries exhausted
        """
        last_exception: Exception | None = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = operation()
                
                # Track successful retry
                if attempt > 0:
                    with self.stats_lock:
                        self.successful_retries += 1
                    logger.info(
                        f"{operation_name} succeeded after {attempt} retries"
                    )
                
                return result
                
            except self.retryable_exceptions as e:
                last_exception = e
                
                with self.stats_lock:
                    self.total_retries += 1
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/"
                        f"{self.config.max_retries + 1}): {e}. "
                        f"Retrying in {delay}ms..."
                    )
                    time.sleep(delay / 1000.0)
                else:
                    logger.error(
                        f"{operation_name} failed after "
                        f"{self.config.max_retries + 1} attempts: {e}"
                    )
                    with self.stats_lock:
                        self.failed_retries += 1
        
        if last_exception is not None:
            raise last_exception
        
        # Should never reach here
        raise RuntimeError(f"{operation_name} failed with unknown error")
    
    def _calculate_delay(self, attempt: int) -> int:
        """
        Calculate delay for retry attempt using exponential backoff with jitter.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in milliseconds
        """
        import random
        
        # Exponential backoff
        delay = self.config.base_delay_ms * (
            self.config.exponential_base ** attempt
        )
        
        # Apply jitter
        jitter = delay * self.config.jitter_factor * random.random()
        delay = delay + jitter
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay_ms)
        
        return int(delay)
    
    def get_stats(self) -> dict[str, int]:
        """Get retry statistics."""
        with self.stats_lock:
            return {
                "total_retries": self.total_retries,
                "successful_retries": self.successful_retries,
                "failed_retries": self.failed_retries,
            }


# =============================================================================
# Circuit Breaker States
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject all requests
    HALF_OPEN = "half_open"  # Testing if service recovered


# =============================================================================
# Circuit Breaker
# =============================================================================

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    failure_threshold: int = 5
    """Number of failures before opening circuit."""
    
    success_threshold: int = 2
    """Number of successes in half-open to close circuit."""
    
    timeout_seconds: float = 30.0
    """Time to wait before transitioning from open to half-open."""
    
    half_open_max_calls: int = 3
    """Maximum calls allowed in half-open state."""


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Implements the Circuit Breaker pattern.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests are rejected
    - HALF_OPEN: Testing if service recovered
    
    Transitions:
    - CLOSED -> OPEN: When failures >= threshold
    - OPEN -> HALF_OPEN: After timeout period
    - HALF_OPEN -> CLOSED: When successes >= threshold
    - HALF_OPEN -> OPEN: When any failure occurs
    """
    
    def __init__(
        self,
        name: str = "default",
        config: CircuitBreakerConfig | None = None,
        on_state_change: Callable[[CircuitState, CircuitState], None] | None = None,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name for logging purposes
            config: Circuit breaker configuration
            on_state_change: Callback when state changes (old_state, new_state)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change
        
        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = Lock()
        
        logger.info(
            f"CircuitBreaker '{name}' initialized: "
            f"failure_threshold={self.config.failure_threshold}, "
            f"timeout={self.config.timeout_seconds}s"
        )
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        return self.state == CircuitState.OPEN
    
    def execute(
        self,
        operation: Callable[[], T],
        operation_name: str = "operation",
    ) -> T:
        """
        Execute operation through circuit breaker.
        
        Args:
            operation: Callable to execute
            operation_name: Name for logging
            
        Returns:
            Result of the operation
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Exception: From the operation
        """
        with self._lock:
            self._check_state_transition()
            
            if self._state == CircuitState.OPEN:
                logger.warning(
                    f"Circuit '{self.name}' is OPEN, rejecting {operation_name}"
                )
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open"
                )
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    logger.warning(
                        f"Circuit '{self.name}' HALF_OPEN max calls reached"
                    )
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' half-open limit reached"
                    )
                self._half_open_calls += 1
        
        try:
            result = operation()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    f"Circuit '{self.name}' HALF_OPEN success "
                    f"({self._success_count}/{self.config.success_threshold})"
                )
                
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            else:
                # Reset failure count on success in closed state
                self._failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"Circuit '{self.name}' HALF_OPEN failure, reopening"
                )
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                logger.debug(
                    f"Circuit '{self.name}' failure count: "
                    f"{self._failure_count}/{self.config.failure_threshold}"
                )
                
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def _check_state_transition(self) -> None:
        """Check if state should transition (called with lock held)."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    self._transition_to(CircuitState.HALF_OPEN)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state (called with lock held)."""
        old_state = self._state
        self._state = new_state
        
        # Reset counters based on new state
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.OPEN:
            self._last_failure_time = time.time()
        
        logger.info(
            f"Circuit '{self.name}' state: {old_state.value} -> {new_state.value}"
        )
        
        # Notify callback
        if self.on_state_change is not None:
            try:
                self.on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None
            
            logger.info(f"Circuit '{self.name}' manually reset from {old_state.value}")
    
    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "half_open_calls": self._half_open_calls,
                "last_failure_time": self._last_failure_time,
            }


# =============================================================================
# Connection Monitor
# =============================================================================

@dataclass
class ConnectionHealth:
    """Connection health metrics."""
    
    connected: bool = False
    last_success_time: datetime | None = None
    last_failure_time: datetime | None = None
    consecutive_failures: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    
    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return (
            self.connected
            and self.consecutive_failures < 3
            and self.success_rate >= 0.9
        )


class ConnectionMonitor:
    """
    Monitors connection health and tracks metrics.
    
    Provides:
    - Connection state tracking
    - Success/failure metrics
    - Health status reporting
    """
    
    def __init__(
        self,
        name: str = "default",
        unhealthy_threshold: int = 3,
    ):
        """
        Initialize connection monitor.
        
        Args:
            name: Connection name for logging
            unhealthy_threshold: Consecutive failures before unhealthy
        """
        self.name = name
        self.unhealthy_threshold = unhealthy_threshold
        
        self._health = ConnectionHealth()
        self._lock = Lock()
        
        logger.debug(f"ConnectionMonitor '{name}' initialized")
    
    def record_success(self) -> None:
        """Record successful request."""
        with self._lock:
            self._health.connected = True
            self._health.last_success_time = datetime.now(timezone.utc)
            self._health.consecutive_failures = 0
            self._health.total_requests += 1
            
            logger.debug(
                f"Connection '{self.name}' success "
                f"(total: {self._health.total_requests})"
            )
    
    def record_failure(self, error: Exception | None = None) -> None:
        """Record failed request."""
        with self._lock:
            self._health.last_failure_time = datetime.now(timezone.utc)
            self._health.consecutive_failures += 1
            self._health.total_requests += 1
            self._health.failed_requests += 1
            
            if self._health.consecutive_failures >= self.unhealthy_threshold:
                self._health.connected = False
            
            logger.warning(
                f"Connection '{self.name}' failure "
                f"(consecutive: {self._health.consecutive_failures}, "
                f"error: {error})"
            )
    
    def record_disconnect(self) -> None:
        """Record disconnection."""
        with self._lock:
            self._health.connected = False
            logger.info(f"Connection '{self.name}' disconnected")
    
    def record_connect(self) -> None:
        """Record successful connection."""
        with self._lock:
            self._health.connected = True
            self._health.consecutive_failures = 0
            logger.info(f"Connection '{self.name}' connected")
    
    def get_health(self) -> ConnectionHealth:
        """Get current health status."""
        with self._lock:
            return ConnectionHealth(
                connected=self._health.connected,
                last_success_time=self._health.last_success_time,
                last_failure_time=self._health.last_failure_time,
                consecutive_failures=self._health.consecutive_failures,
                total_requests=self._health.total_requests,
                failed_requests=self._health.failed_requests,
            )
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        with self._lock:
            return self._health.is_healthy
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._health = ConnectionHealth()
            logger.info(f"Connection '{self.name}' metrics reset")


# =============================================================================
# Resilient Operation Executor
# =============================================================================

class ResilientExecutor:
    """
    Combines RetryManager and CircuitBreaker for resilient operations.
    
    Operations are:
    1. First checked against circuit breaker
    2. Then retried on failure (if circuit allows)
    """
    
    def __init__(
        self,
        name: str = "default",
        retry_config: RetryConfig | None = None,
        circuit_config: CircuitBreakerConfig | None = None,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
    ):
        """
        Initialize resilient executor.
        
        Args:
            name: Executor name for logging
            retry_config: Retry configuration
            circuit_config: Circuit breaker configuration
            retryable_exceptions: Exceptions that should trigger retry
        """
        self.name = name
        
        self.retry_manager = RetryManager(
            config=retry_config,
            retryable_exceptions=retryable_exceptions,
        )
        
        self.circuit_breaker = CircuitBreaker(
            name=name,
            config=circuit_config,
        )
        
        self.connection_monitor = ConnectionMonitor(name=name)
        
        logger.info(f"ResilientExecutor '{name}' initialized")
    
    def execute(
        self,
        operation: Callable[[], T],
        operation_name: str = "operation",
    ) -> T:
        """
        Execute operation with retry and circuit breaker protection.
        
        Args:
            operation: Callable to execute
            operation_name: Name for logging
            
        Returns:
            Result of the operation
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Exception: From the operation after all retries exhausted
        """
        def wrapped_operation() -> T:
            return self.circuit_breaker.execute(operation, operation_name)
        
        try:
            result = self.retry_manager.execute(wrapped_operation, operation_name)
            self.connection_monitor.record_success()
            return result
        except CircuitBreakerOpenError:
            # Don't count circuit breaker rejection as connection failure
            raise
        except Exception as e:
            self.connection_monitor.record_failure(e)
            raise
    
    def get_status(self) -> dict[str, Any]:
        """Get executor status including all components."""
        return {
            "name": self.name,
            "circuit_breaker": self.circuit_breaker.get_status(),
            "retry_stats": self.retry_manager.get_stats(),
            "connection_health": {
                "connected": self.connection_monitor.get_health().connected,
                "is_healthy": self.connection_monitor.is_healthy(),
                "success_rate": self.connection_monitor.get_health().success_rate,
                "consecutive_failures": self.connection_monitor.get_health().consecutive_failures,
            },
        }
    
    def reset(self) -> None:
        """Reset all components."""
        self.circuit_breaker.reset()
        self.connection_monitor.reset()
        logger.info(f"ResilientExecutor '{self.name}' reset")
