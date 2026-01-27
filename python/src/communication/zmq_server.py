"""
ZeroMQ server for MT5 EA <-> Python prediction service communication.

Implements REP (reply) socket pattern for request/response communication
with support for trading signals, health checks, and heartbeat monitoring.

Features:
- Automatic reconnection on socket failures
- Error rate monitoring with alerts
- Graceful degradation under load
"""

# pyright: reportOptionalMemberAccess=false
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any, Callable, cast

if TYPE_CHECKING:
    import zmq

from ..inference.predictor import PredictionService
from ..utils.config import get_config
from ..utils.logger import get_logger
from .error_handling import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    ConnectionMonitor,
)

logger = get_logger("communication.zmq_server")


# =============================================================================
# ZMQ Server
# =============================================================================

class ZMQServer:
    """
    ZeroMQ server for MT5 EA communication.
    
    Implements REP (reply) socket pattern for request/response communication.
    Handles trading signal requests, health checks, and heartbeat monitoring.
    
    Features:
    - Automatic socket reconnection on failures
    - Error rate monitoring with circuit breaker
    - Connection health tracking
    """
    
    def __init__(
        self,
        port: int | None = None,
        prediction_service: PredictionService | None = None,
        timeout_ms: int | None = None,
        max_reconnect_attempts: int = 5,
        reconnect_delay_ms: int = 1000,
        on_error_callback: Callable[[str, Exception], None] | None = None,
    ):
        """
        Initialize ZMQ server.
        
        Args:
            port: Port number (from config if None)
            prediction_service: PredictionService instance
            timeout_ms: Request timeout in milliseconds (from config if None)
            max_reconnect_attempts: Max reconnection attempts before giving up
            reconnect_delay_ms: Delay between reconnection attempts
            on_error_callback: Optional callback for error notifications
        """
        config = get_config()
        
        # Load configuration
        self.port = port if port is not None else \
            config.trading.communication.zeromq.request_port
        self.timeout_ms = timeout_ms if timeout_ms is not None else \
            config.trading.communication.zeromq.timeout_ms
        self.min_confidence = config.trading.signals.get("min_confidence", 0.65)
        
        # Reconnection settings
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_ms = reconnect_delay_ms
        self.on_error_callback = on_error_callback
        
        # Prediction service
        self.prediction_service = prediction_service
        if self.prediction_service is None:
            raise ValueError("PredictionService is required")
        
        # ZMQ context and socket
        self.context: Any = None
        self.socket: Any = None
        self.running = False
        self._reconnect_count = 0
        
        # Statistics
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        self.stats_lock = Lock()
        
        # Error handling components
        self.connection_monitor = ConnectionMonitor(
            name=f"zmq_server_{self.port}",
            unhealthy_threshold=5,
        )
        
        self.circuit_breaker = CircuitBreaker(
            name=f"zmq_server_{self.port}",
            config=CircuitBreakerConfig(
                failure_threshold=10,
                success_threshold=3,
                timeout_seconds=60.0,
            ),
            on_state_change=self._on_circuit_state_change,
        )
        
        logger.info(
            f"ZMQServer initialized: port={self.port}, "
            f"timeout={self.timeout_ms}ms, "
            f"min_confidence={self.min_confidence}, "
            f"max_reconnect={max_reconnect_attempts}"
        )
    
    def _on_circuit_state_change(
        self,
        old_state: CircuitState,
        new_state: CircuitState,
    ) -> None:
        """Handle circuit breaker state changes."""
        if new_state == CircuitState.OPEN:
            error_msg = (
                f"Circuit breaker OPENED for ZMQ server on port {self.port}. "
                f"Too many errors detected."
            )
            logger.error(error_msg)
            self._notify_error("circuit_breaker_open", Exception(error_msg))
        elif new_state == CircuitState.CLOSED and old_state == CircuitState.HALF_OPEN:
            logger.info(
                f"Circuit breaker CLOSED for ZMQ server on port {self.port}. "
                f"Service recovered."
            )
    
    def _notify_error(self, error_type: str, error: Exception) -> None:
        """Notify error via callback if configured."""
        if self.on_error_callback is not None:
            try:
                self.on_error_callback(error_type, error)
            except Exception as callback_error:
                logger.error(f"Error in error callback: {callback_error}")
    
    def start(self) -> None:
        """
        Start the ZMQ server.
        
        Blocks until stop() is called or an exception occurs.
        Includes automatic reconnection on socket failures.
        """
        self.running = True
        self._reconnect_count = 0
        
        while self.running:
            try:
                self._run_server_loop()
            except Exception as e:
                if not self.running:
                    break
                
                logger.error(f"Server loop failed: {e}")
                self._notify_error("server_loop_error", e)
                
                # Attempt reconnection
                if not self._attempt_reconnect():
                    logger.error("Max reconnection attempts reached, stopping server")
                    self.running = False
                    break
        
        self._cleanup_socket()
    
    def _run_server_loop(self) -> None:
        """Run the main server loop."""
        # Import zmq at runtime
        import zmq
        
        # Create ZMQ context and socket
        self._setup_socket()
        self.connection_monitor.record_connect()
        
        logger.info(f"ZMQ Server started on port {self.port}")
        
        # Reset reconnection counter on successful start
        self._reconnect_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        # Main server loop
        while self.running:
            try:
                # Check if circuit breaker allows processing
                if self.circuit_breaker.is_open:
                    logger.warning("Circuit breaker is OPEN, waiting...")
                    time.sleep(1.0)
                    continue
                
                # Receive request
                socket_ref: Any = self.socket
                try:
                    message_bytes = socket_ref.recv(zmq.NOBLOCK)
                except zmq.Again:
                    # Timeout - no message received, continue
                    time.sleep(0.01)
                    continue
                
                message = json.loads(message_bytes.decode("utf-8"))
                
                # Log request
                logger.debug(f"Received request: {message.get('type')}")
                
                # Handle request
                response = self.handle_request(message)
                
                # Send response
                socket_ref.send_json(response)
                
                # Update statistics
                with self.stats_lock:
                    self.request_count += 1
                
                # Record success
                self.connection_monitor.record_success()
                consecutive_errors = 0
                
            except zmq.ZMQError as e:
                consecutive_errors += 1
                self.connection_monitor.record_failure(e)
                logger.error(f"ZMQ error: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        f"Too many consecutive ZMQ errors ({consecutive_errors}), "
                        f"triggering reconnection"
                    )
                    raise  # Exit to reconnection logic
                
                if self.running:
                    time.sleep(0.1)
                    
            except json.JSONDecodeError as e:
                # Invalid JSON - respond with error but don't count as connection failure
                with self.stats_lock:
                    self.error_count += 1
                logger.error(f"Invalid JSON received: {e}")
                try:
                    socket_ref = self.socket
                    if socket_ref is not None:
                        socket_ref.send_json({
                            "status": "error",
                            "error": "Invalid JSON format",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                except Exception:
                    pass
                    
            except Exception as e:
                consecutive_errors += 1
                self.connection_monitor.record_failure(e)
                logger.error(f"Unexpected error in server loop: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    raise  # Exit to reconnection logic
                
                if self.running:
                    time.sleep(0.1)
    
    def _setup_socket(self) -> None:
        """Create and configure ZMQ socket."""
        import zmq
        
        # Clean up existing socket if any
        self._cleanup_socket()
        
        # Create ZMQ context and socket
        context = zmq.Context()
        self.context = context
        socket: Any = context.socket(zmq.REP)
        self.socket = socket
        
        # Set socket options
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close
        
        # Bind to port
        socket.bind(f"tcp://*:{self.port}")
    
    def _cleanup_socket(self) -> None:
        """Clean up ZMQ socket and context."""
        if self.socket is not None:
            try:
                self.socket.close(linger=0)
            except Exception as e:
                logger.warning(f"Error closing socket: {e}")
            self.socket = None
        
        if self.context is not None:
            try:
                self.context.term()
            except Exception as e:
                logger.warning(f"Error terminating context: {e}")
            self.context = None
    
    def _attempt_reconnect(self) -> bool:
        """
        Attempt to reconnect after failure.
        
        Returns:
            True if should retry, False if max attempts reached
        """
        self._reconnect_count += 1
        
        if self._reconnect_count > self.max_reconnect_attempts:
            self._notify_error(
                "max_reconnect_exceeded",
                Exception(f"Max reconnection attempts ({self.max_reconnect_attempts}) exceeded"),
            )
            return False
        
        logger.warning(
            f"Attempting reconnection {self._reconnect_count}/"
            f"{self.max_reconnect_attempts} in {self.reconnect_delay_ms}ms..."
        )
        
        self._cleanup_socket()
        self.connection_monitor.record_disconnect()
        
        time.sleep(self.reconnect_delay_ms / 1000.0)
        return True
    
    def stop(self) -> None:
        """Gracefully shutdown the server."""
        logger.info("Stopping ZMQ Server...")
        self.running = False
        
        self._cleanup_socket()
        self.connection_monitor.record_disconnect()
        
        logger.info("ZMQ Server stopped")
    
    def handle_request(self, message: dict[str, Any]) -> dict[str, Any]:
        """
        Handle incoming request.
        
        Args:
            message: Request message dict
            
        Returns:
            Response message dict
        """
        try:
            msg_type = message.get("type")
            
            if msg_type == "PING":
                return self._handle_ping()
            elif msg_type == "GET_SIGNAL":
                return self._handle_get_signal(message.get("data"))
            elif msg_type == "HEARTBEAT":
                return self._handle_heartbeat()
            else:
                return {
                    "status": "error",
                    "error": f"Unknown message type: {msg_type}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            with self.stats_lock:
                self.error_count += 1
            
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    
    def _handle_ping(self) -> dict[str, Any]:
        """
        Handle PING request.
        
        Returns:
            PONG response
        """
        logger.debug("Handling PING request")
        return {
            "status": "success",
            "data": "PONG",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def _handle_heartbeat(self) -> dict[str, Any]:
        """
        Handle HEARTBEAT request.
        
        Returns:
            Server status response
        """
        logger.debug("Handling HEARTBEAT request")
        
        with self.stats_lock:
            uptime_seconds = int(time.time() - self.start_time)
            request_count = self.request_count
            error_count = self.error_count
        
        service = cast(Any, self.prediction_service)
        return {
            "status": "success",
            "data": {
                "uptime_seconds": uptime_seconds,
                "requests_processed": request_count,
                "errors": error_count,
                "model_info": service.get_model_info(),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def _handle_get_signal(self, data: dict[str, Any] | None) -> dict[str, Any]:
        """
        Handle GET_SIGNAL request.
        
        Args:
            data: Request data containing features and metadata
            
        Returns:
            Trading signal response
        """
        logger.debug("Handling GET_SIGNAL request")
        
        try:
            if data is None:
                raise ValueError("GET_SIGNAL requires data field")
            
            # Extract features
            features_dict = data.get("features")
            if not features_dict:
                raise ValueError("Missing features in request")
            
            # Convert to DataFrame
            import pandas as pd
            features_df = pd.DataFrame([features_dict])
            
            # Get prediction
            service = cast(Any, self.prediction_service)
            signal = service.predict(
                features_df,
                min_confidence=self.min_confidence
            )
            
            # Extract metadata
            symbol = data.get("symbol", "UNKNOWN")
            current_price = data.get("current_price", 0.0)
            atr = data.get("features", {}).get("atr_14", 0.01)
            
            # Calculate stop loss and take profit
            sl = None
            tp = None
            
            if signal["direction"] == "buy":
                sl = current_price - 2 * atr
                tp = current_price + 3 * atr
            elif signal["direction"] == "sell":
                sl = current_price + 2 * atr
                tp = current_price - 3 * atr
            
            # Build response
            response = {
                "status": "success",
                "data": {
                    "direction": signal["direction"],
                    "confidence": signal["confidence"],
                    "expected_return": signal["expected_return"],
                    "stop_loss": sl,
                    "take_profit": tp,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            logger.debug(
                f"Signal generated for {symbol}: "
                f"direction={signal['direction']}, "
                f"confidence={signal['confidence']:.3f}"
            )
            
            return response
        
        except ValueError as e:
            logger.error(f"Invalid request data: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {
                "status": "error",
                "error": f"Signal generation failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get server statistics.
        
        Returns:
            Statistics dict
        """
        with self.stats_lock:
            uptime_seconds = int(time.time() - self.start_time)
            health = self.connection_monitor.get_health()
            
            return {
                "uptime_seconds": uptime_seconds,
                "requests_processed": self.request_count,
                "errors": self.error_count,
                "running": self.running,
                "reconnect_count": self._reconnect_count,
                "circuit_breaker_state": self.circuit_breaker.state.value,
                "connection_health": {
                    "connected": health.connected,
                    "success_rate": health.success_rate,
                    "consecutive_failures": health.consecutive_failures,
                    "is_healthy": health.is_healthy,
                },
            }
    
    def get_health_status(self) -> dict[str, Any]:
        """
        Get detailed health status for monitoring.
        
        Returns:
            Health status dict
        """
        health = self.connection_monitor.get_health()
        circuit_status = self.circuit_breaker.get_status()
        
        return {
            "healthy": health.is_healthy and not self.circuit_breaker.is_open,
            "connection": {
                "connected": health.connected,
                "success_rate": health.success_rate,
                "consecutive_failures": health.consecutive_failures,
                "total_requests": health.total_requests,
                "failed_requests": health.failed_requests,
            },
            "circuit_breaker": circuit_status,
            "server": {
                "running": self.running,
                "port": self.port,
                "reconnect_count": self._reconnect_count,
            },
        }
