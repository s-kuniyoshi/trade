"""
Unit tests for error handling and resilience patterns.

Tests cover:
- RetryManager: Exponential backoff retry logic
- CircuitBreaker: Circuit breaker state transitions
- ConnectionMonitor: Connection health tracking
- AlertManager: Alert generation and rate limiting
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime, timezone

# Import error handling modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.communication.error_handling import (
    RetryManager,
    RetryConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
    ConnectionMonitor,
    ConnectionHealth,
    ResilientExecutor,
)
from src.communication.alerting import (
    Alert,
    AlertSeverity,
    AlertManager,
    AlertRateLimiter,
    RateLimitConfig,
    LogAlertHandler,
    FileAlertHandler,
)


# =============================================================================
# RetryManager Tests
# =============================================================================

class TestRetryManager:
    """Tests for RetryManager class."""
    
    def test_successful_operation_no_retry(self):
        """Test operation succeeds on first attempt."""
        manager = RetryManager(config=RetryConfig(max_retries=3))
        
        call_count = 0
        def operation():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = manager.execute(operation, "test_op")
        
        assert result == "success"
        assert call_count == 1
    
    def test_retry_on_failure(self):
        """Test operation retries on failure and eventually succeeds."""
        manager = RetryManager(
            config=RetryConfig(
                max_retries=3,
                base_delay_ms=10,  # Short delay for testing
            )
        )
        
        call_count = 0
        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = manager.execute(operation, "test_op")
        
        assert result == "success"
        assert call_count == 3
    
    def test_max_retries_exceeded(self):
        """Test exception raised when max retries exceeded."""
        manager = RetryManager(
            config=RetryConfig(
                max_retries=2,
                base_delay_ms=10,
            )
        )
        
        def operation():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            manager.execute(operation, "test_op")
        
        stats = manager.get_stats()
        assert stats["failed_retries"] == 1
    
    def test_retryable_exceptions_filter(self):
        """Test only specified exceptions trigger retry."""
        manager = RetryManager(
            config=RetryConfig(max_retries=3, base_delay_ms=10),
            retryable_exceptions=(ValueError,),
        )
        
        def operation():
            raise TypeError("Not retryable")
        
        with pytest.raises(TypeError):
            manager.execute(operation, "test_op")
    
    def test_exponential_backoff_delay(self):
        """Test delay increases exponentially."""
        manager = RetryManager(
            config=RetryConfig(
                base_delay_ms=100,
                exponential_base=2.0,
                jitter_factor=0.0,  # No jitter for deterministic test
                max_delay_ms=10000,
            )
        )
        
        # Calculate delays for attempts 0, 1, 2
        delay_0 = manager._calculate_delay(0)
        delay_1 = manager._calculate_delay(1)
        delay_2 = manager._calculate_delay(2)
        
        assert delay_0 == 100
        assert delay_1 == 200
        assert delay_2 == 400
    
    def test_max_delay_cap(self):
        """Test delay is capped at max_delay_ms."""
        manager = RetryManager(
            config=RetryConfig(
                base_delay_ms=1000,
                exponential_base=2.0,
                jitter_factor=0.0,
                max_delay_ms=2000,
            )
        )
        
        delay = manager._calculate_delay(10)  # Would be 1024000 without cap
        assert delay == 2000


# =============================================================================
# CircuitBreaker Tests
# =============================================================================

class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""
    
    def test_initial_state_closed(self):
        """Test circuit breaker starts in closed state."""
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed
        assert not cb.is_open
    
    def test_opens_after_failure_threshold(self):
        """Test circuit opens after reaching failure threshold."""
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=3),
        )
        
        def failing_operation():
            raise ValueError("Failure")
        
        # Trigger failures up to threshold
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.execute(failing_operation)
        
        # Circuit should now be open
        assert cb.state == CircuitState.OPEN
        assert cb.is_open
    
    def test_open_circuit_rejects_requests(self):
        """Test open circuit rejects all requests."""
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        
        # Open the circuit
        def failing_op():
            raise ValueError("Failure")
        
        with pytest.raises(ValueError):
            cb.execute(failing_op)
        
        # Now requests should be rejected
        def good_op():
            return "success"
        
        with pytest.raises(CircuitBreakerOpenError):
            cb.execute(good_op)
    
    def test_half_open_after_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                timeout_seconds=0.1,  # 100ms timeout
            ),
        )
        
        # Open the circuit
        def failing_op():
            raise ValueError("Failure")
        
        with pytest.raises(ValueError):
            cb.execute(failing_op)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Should now be half-open
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_closes_after_success_in_half_open(self):
        """Test circuit closes after successes in half-open state."""
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                success_threshold=2,
                timeout_seconds=0.1,
            ),
        )
        
        # Open the circuit
        def failing_op():
            raise ValueError("Failure")
        
        with pytest.raises(ValueError):
            cb.execute(failing_op)
        
        # Wait for timeout to enter half-open
        time.sleep(0.15)
        
        # Success in half-open
        def good_op():
            return "success"
        
        cb.execute(good_op)
        cb.execute(good_op)
        
        # Should be closed now
        assert cb.state == CircuitState.CLOSED
    
    def test_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in half-open state."""
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                timeout_seconds=0.1,
            ),
        )
        
        # Open the circuit
        def failing_op():
            raise ValueError("Failure")
        
        with pytest.raises(ValueError):
            cb.execute(failing_op)
        
        # Wait for timeout to enter half-open
        time.sleep(0.15)
        
        # Failure in half-open should reopen
        with pytest.raises(ValueError):
            cb.execute(failing_op)
        
        assert cb.state == CircuitState.OPEN
    
    def test_state_change_callback(self):
        """Test state change callback is called."""
        state_changes = []
        
        def on_change(old_state, new_state):
            state_changes.append((old_state, new_state))
        
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=1),
            on_state_change=on_change,
        )
        
        def failing_op():
            raise ValueError("Failure")
        
        with pytest.raises(ValueError):
            cb.execute(failing_op)
        
        assert len(state_changes) == 1
        assert state_changes[0] == (CircuitState.CLOSED, CircuitState.OPEN)
    
    def test_manual_reset(self):
        """Test manual reset returns to closed state."""
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        
        def failing_op():
            raise ValueError("Failure")
        
        with pytest.raises(ValueError):
            cb.execute(failing_op)
        
        assert cb.state == CircuitState.OPEN
        
        cb.reset()
        
        assert cb.state == CircuitState.CLOSED


# =============================================================================
# ConnectionMonitor Tests
# =============================================================================

class TestConnectionMonitor:
    """Tests for ConnectionMonitor class."""
    
    def test_initial_state(self):
        """Test initial state is disconnected."""
        monitor = ConnectionMonitor(name="test")
        health = monitor.get_health()
        
        assert not health.connected
        assert health.consecutive_failures == 0
        assert health.total_requests == 0
    
    def test_record_connect(self):
        """Test recording connection."""
        monitor = ConnectionMonitor(name="test")
        monitor.record_connect()
        
        health = monitor.get_health()
        assert health.connected
    
    def test_record_success(self):
        """Test recording successful request."""
        monitor = ConnectionMonitor(name="test")
        monitor.record_connect()
        monitor.record_success()
        
        health = monitor.get_health()
        assert health.connected
        assert health.total_requests == 1
        assert health.failed_requests == 0
        assert health.success_rate == 1.0
    
    def test_record_failure(self):
        """Test recording failed request."""
        monitor = ConnectionMonitor(name="test")
        monitor.record_connect()
        monitor.record_failure(Exception("Test error"))
        
        health = monitor.get_health()
        assert health.consecutive_failures == 1
        assert health.total_requests == 1
        assert health.failed_requests == 1
    
    def test_disconnects_after_threshold(self):
        """Test connection marked as disconnected after threshold failures."""
        monitor = ConnectionMonitor(name="test", unhealthy_threshold=3)
        monitor.record_connect()
        
        for _ in range(3):
            monitor.record_failure(Exception("Test"))
        
        health = monitor.get_health()
        assert not health.connected
    
    def test_success_resets_consecutive_failures(self):
        """Test success resets consecutive failure count."""
        monitor = ConnectionMonitor(name="test")
        monitor.record_connect()
        
        monitor.record_failure(Exception("Test"))
        monitor.record_failure(Exception("Test"))
        assert monitor.get_health().consecutive_failures == 2
        
        monitor.record_success()
        assert monitor.get_health().consecutive_failures == 0
    
    def test_is_healthy(self):
        """Test healthy status calculation."""
        monitor = ConnectionMonitor(name="test", unhealthy_threshold=3)
        monitor.record_connect()
        
        # Initially healthy
        assert monitor.is_healthy()
        
        # Record many successes
        for _ in range(10):
            monitor.record_success()
        
        assert monitor.is_healthy()
        
        # Record failures
        for _ in range(3):
            monitor.record_failure(Exception("Test"))
        
        assert not monitor.is_healthy()


# =============================================================================
# AlertManager Tests
# =============================================================================

class TestAlertManager:
    """Tests for AlertManager class."""
    
    def test_send_alert(self):
        """Test sending an alert."""
        manager = AlertManager()
        
        result = manager.send_alert(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test",
            source="test",
        )
        
        assert result is True
        stats = manager.get_stats()
        assert stats["total_alerts"] == 1
        assert stats["sent"] == 1
    
    def test_convenience_methods(self):
        """Test convenience methods (info, warning, error, critical)."""
        manager = AlertManager()
        
        manager.info("Info", "Info message", "test")
        manager.warning("Warning", "Warning message", "test")
        manager.error("Error", "Error message", "test")
        manager.critical("Critical", "Critical message", "test")
        
        stats = manager.get_stats()
        assert stats["total_alerts"] == 4


# =============================================================================
# AlertRateLimiter Tests
# =============================================================================

class TestAlertRateLimiter:
    """Tests for AlertRateLimiter class."""
    
    def test_allows_initial_alerts(self):
        """Test initial alerts are allowed."""
        limiter = AlertRateLimiter(
            config=RateLimitConfig(max_alerts_per_minute=10)
        )
        
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test message",
            source="test",
        )
        
        assert limiter.should_allow(alert)
    
    def test_rate_limits_after_threshold(self):
        """Test rate limiting after threshold is reached."""
        limiter = AlertRateLimiter(
            config=RateLimitConfig(
                max_alerts_per_minute=3,
                cooldown_seconds=0.1,
                dedup_window_seconds=0.0,  # Disable dedup for this test
            )
        )
        
        # Create unique alerts (different titles to avoid dedup)
        alerts = [
            Alert(
                severity=AlertSeverity.WARNING,
                title=f"Test {i}",
                message="Test message",
                source="test",
            )
            for i in range(4)
        ]
        
        # Should allow first 3
        assert limiter.should_allow(alerts[0])
        assert limiter.should_allow(alerts[1])
        assert limiter.should_allow(alerts[2])
        
        # 4th should be blocked
        assert not limiter.should_allow(alerts[3])
    
    def test_deduplicates_identical_alerts(self):
        """Test identical alerts are deduplicated."""
        limiter = AlertRateLimiter(
            config=RateLimitConfig(
                max_alerts_per_minute=100,
                dedup_window_seconds=1.0,
            )
        )
        
        alert1 = Alert(
            severity=AlertSeverity.WARNING,
            title="Same Alert",
            message="Same message",
            source="test",
            error_type="SAME_ERROR",
        )
        
        alert2 = Alert(
            severity=AlertSeverity.WARNING,
            title="Same Alert",
            message="Same message",
            source="test",
            error_type="SAME_ERROR",
        )
        
        assert limiter.should_allow(alert1)
        assert not limiter.should_allow(alert2)  # Deduplicated
    
    def test_allows_different_sources(self):
        """Test different sources have separate limits."""
        limiter = AlertRateLimiter(
            config=RateLimitConfig(
                max_alerts_per_minute=2,
                dedup_window_seconds=0.0,  # Disable dedup for this test
            )
        )
        
        # Create unique alerts per source (different titles to avoid dedup)
        alerts_source1 = [
            Alert(
                severity=AlertSeverity.WARNING,
                title=f"Test S1-{i}",
                message="Message",
                source="source1",
            )
            for i in range(3)
        ]
        
        alerts_source2 = [
            Alert(
                severity=AlertSeverity.WARNING,
                title=f"Test S2-{i}",
                message="Message",
                source="source2",
            )
            for i in range(3)
        ]
        
        # Both sources should have their own limits
        assert limiter.should_allow(alerts_source1[0])
        assert limiter.should_allow(alerts_source2[0])
        assert limiter.should_allow(alerts_source1[1])
        assert limiter.should_allow(alerts_source2[1])
        
        # Now both should be limited
        assert not limiter.should_allow(alerts_source1[2])
        assert not limiter.should_allow(alerts_source2[2])


# =============================================================================
# ResilientExecutor Tests
# =============================================================================

class TestResilientExecutor:
    """Tests for ResilientExecutor class."""
    
    def test_successful_execution(self):
        """Test successful operation execution."""
        executor = ResilientExecutor(name="test")
        
        def operation():
            return "success"
        
        result = executor.execute(operation, "test_op")
        
        assert result == "success"
        status = executor.get_status()
        assert status["connection_health"]["connected"]
    
    def test_retry_and_circuit_breaker_integration(self):
        """Test retry and circuit breaker work together."""
        executor = ResilientExecutor(
            name="test",
            retry_config=RetryConfig(max_retries=2, base_delay_ms=10),
            circuit_config=CircuitBreakerConfig(failure_threshold=5),
        )
        
        call_count = 0
        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"
        
        result = executor.execute(operation, "test_op")
        
        assert result == "success"
        assert call_count == 2
    
    def test_circuit_breaker_prevents_execution(self):
        """Test circuit breaker prevents execution when open."""
        executor = ResilientExecutor(
            name="test",
            retry_config=RetryConfig(max_retries=0),
            circuit_config=CircuitBreakerConfig(failure_threshold=1),
        )
        
        # Open the circuit
        def failing_op():
            raise ValueError("Failure")
        
        with pytest.raises(ValueError):
            executor.execute(failing_op, "test_op")
        
        # Circuit should be open, good operation should fail
        def good_op():
            return "success"
        
        with pytest.raises(CircuitBreakerOpenError):
            executor.execute(good_op, "test_op")
    
    def test_reset_clears_all_state(self):
        """Test reset clears circuit breaker and connection monitor."""
        executor = ResilientExecutor(
            name="test",
            retry_config=RetryConfig(max_retries=0),  # No retries for this test
            circuit_config=CircuitBreakerConfig(failure_threshold=1),
        )
        
        # Open the circuit
        def failing_op():
            raise ValueError("Failure")
        
        with pytest.raises(ValueError):
            executor.execute(failing_op, "test_op")
        
        assert executor.circuit_breaker.is_open
        
        executor.reset()
        
        assert executor.circuit_breaker.is_closed


# =============================================================================
# Alert Data Tests
# =============================================================================

class TestAlert:
    """Tests for Alert dataclass."""
    
    def test_to_dict(self):
        """Test alert serialization to dict."""
        alert = Alert(
            severity=AlertSeverity.ERROR,
            title="Test Alert",
            message="Test message",
            source="test_source",
            error_type="TEST_ERROR",
            metadata={"key": "value"},
        )
        
        data = alert.to_dict()
        
        assert data["severity"] == "error"
        assert data["title"] == "Test Alert"
        assert data["message"] == "Test message"
        assert data["source"] == "test_source"
        assert data["error_type"] == "TEST_ERROR"
        assert data["metadata"] == {"key": "value"}
        assert "timestamp" in data
    
    def test_str_format(self):
        """Test alert string format."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title="Warning Alert",
            message="Something happened",
            source="module",
        )
        
        alert_str = str(alert)
        
        assert "WARNING" in alert_str
        assert "Warning Alert" in alert_str
        assert "module" in alert_str


# =============================================================================
# LogAlertHandler Tests
# =============================================================================

class TestLogAlertHandler:
    """Tests for LogAlertHandler class."""
    
    def test_supports_severity(self):
        """Test severity filtering."""
        handler = LogAlertHandler(min_severity=AlertSeverity.WARNING)
        
        assert not handler.supports_severity(AlertSeverity.INFO)
        assert handler.supports_severity(AlertSeverity.WARNING)
        assert handler.supports_severity(AlertSeverity.ERROR)
        assert handler.supports_severity(AlertSeverity.CRITICAL)
    
    def test_send_logs_alert(self):
        """Test sending logs the alert."""
        handler = LogAlertHandler(min_severity=AlertSeverity.INFO)
        
        alert = Alert(
            severity=AlertSeverity.ERROR,
            title="Test",
            message="Test message",
            source="test",
        )
        
        result = handler.send(alert)
        assert result is True


# =============================================================================
# FileAlertHandler Tests
# =============================================================================

class TestFileAlertHandler:
    """Tests for FileAlertHandler class."""
    
    def test_writes_to_file(self, tmp_path):
        """Test alert is written to file."""
        file_path = tmp_path / "alerts.json"
        handler = FileAlertHandler(
            file_path=file_path,
            min_severity=AlertSeverity.WARNING,
        )
        
        alert = Alert(
            severity=AlertSeverity.ERROR,
            title="Test Alert",
            message="Test message",
            source="test",
        )
        
        result = handler.send(alert)
        
        assert result is True
        assert file_path.exists()
        
        import json
        with open(file_path) as f:
            alerts = json.load(f)
        
        assert len(alerts) == 1
        assert alerts[0]["title"] == "Test Alert"
    
    def test_respects_max_alerts(self, tmp_path):
        """Test max alerts limit is respected."""
        file_path = tmp_path / "alerts.json"
        handler = FileAlertHandler(
            file_path=file_path,
            min_severity=AlertSeverity.INFO,
            max_alerts=3,
        )
        
        # Send 5 alerts
        for i in range(5):
            alert = Alert(
                severity=AlertSeverity.WARNING,
                title=f"Alert {i}",
                message="Message",
                source="test",
            )
            handler.send(alert)
        
        import json
        with open(file_path) as f:
            alerts = json.load(f)
        
        # Should only keep last 3
        assert len(alerts) == 3
        assert alerts[0]["title"] == "Alert 2"
        assert alerts[2]["title"] == "Alert 4"
