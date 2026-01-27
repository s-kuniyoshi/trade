"""
Integration tests for MT5 communication with ZeroMQ server.

Tests simulate MT5 client communication with the Python ZeroMQ server using
actual ZeroMQ sockets. Covers PING, GET_SIGNAL, HEARTBEAT flows, timeouts,
sequential requests, and concurrent requests.
"""

import pytest
import zmq
import time
import threading
import json
from datetime import datetime, timezone
from unittest.mock import Mock
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.communication.zmq_server import ZMQServer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def mock_prediction_service():
    """
    Create mock prediction service for integration tests.
    
    Returns a Mock object with predict() and get_model_info() methods.
    """
    service = Mock()
    
    # Default prediction response
    service.predict.return_value = {
        "direction": "buy",
        "confidence": 0.75,
        "expected_return": 0.001
    }
    
    # Model info response
    service.get_model_info.return_value = {
        "model_type": "LightGBM",
        "version": "1.0",
        "last_trained": "2026-01-20T15:00:00Z"
    }
    
    return service


@pytest.fixture(scope="module")
def zmq_server(mock_prediction_service):
    """
    Start ZMQ server in background thread.
    
    Uses test port 5556 to avoid conflicts with production server.
    Server runs in daemon thread and is stopped after all tests complete.
    """
    server = ZMQServer(
        port=5556,
        prediction_service=mock_prediction_service,
        timeout_ms=1000
    )
    
    def start_server():
        """Run server in background thread."""
        try:
            server.start()
        except Exception as e:
            print(f"Server error: {e}")
    
    # Start server in daemon thread
    thread = threading.Thread(target=start_server, daemon=True)
    thread.daemon = True
    thread.start()
    
    # Wait for server to start and be ready
    time.sleep(1.0)
    
    # Verify server is running by attempting a ping
    max_retries = 5
    for attempt in range(max_retries):
        try:
            context = zmq.Context()
            test_socket = context.socket(zmq.REQ)
            test_socket.setsockopt(zmq.RCVTIMEO, 2000)
            test_socket.setsockopt(zmq.SNDTIMEO, 2000)
            test_socket.connect("tcp://localhost:5556")
            test_socket.send_json({"type": "PING"})
            response = test_socket.recv_json()
            test_socket.close()
            context.term()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                raise RuntimeError(f"Server failed to start after {max_retries} attempts: {e}")
    
    yield server
    
    # Cleanup
    server.stop()
    time.sleep(0.2)


@pytest.fixture
def zmq_client():
    """
    Create ZMQ REQ client socket.
    
    Connects to test server on port 5556 with 5-second timeout.
    Socket is automatically closed after test completes.
    """
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")
    
    # Set timeouts (5 seconds)
    socket.setsockopt(zmq.RCVTIMEO, 5000)
    socket.setsockopt(zmq.SNDTIMEO, 5000)
    
    yield socket
    
    # Cleanup
    socket.close()
    context.term()


# =============================================================================
# PING Flow Tests
# =============================================================================

class TestPingFlow:
    """Test complete PING request/response cycle."""
    
    def test_full_ping_flow(self, zmq_client):
        """Test complete PING request/response cycle."""
        # Send PING request
        request = {"type": "PING"}
        zmq_client.send_json(request)
        
        # Receive response
        response = zmq_client.recv_json()
        
        # Verify response structure
        assert response["status"] == "success"
        assert response["data"] == "PONG"
        assert "timestamp" in response
        
        # Verify timestamp is ISO 8601
        timestamp_str = response["timestamp"]
        assert "T" in timestamp_str
        assert "+" in timestamp_str or "Z" in timestamp_str
    
    def test_ping_response_is_json_serializable(self, zmq_client):
        """Test PING response is valid JSON."""
        request = {"type": "PING"}
        zmq_client.send_json(request)
        response = zmq_client.recv_json()
        
        # Verify response can be re-serialized
        json_str = json.dumps(response)
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        
        # Verify it can be deserialized
        reparsed = json.loads(json_str)
        assert reparsed["status"] == "success"
    
    def test_ping_response_has_all_required_fields(self, zmq_client):
        """Test PING response includes all required fields."""
        request = {"type": "PING"}
        zmq_client.send_json(request)
        response = zmq_client.recv_json()
        
        # Verify required fields
        assert "status" in response
        assert "data" in response
        assert "timestamp" in response
        
        # Verify no error field in success response
        assert "error" not in response


# =============================================================================
# GET_SIGNAL Flow Tests
# =============================================================================

class TestGetSignalFlow:
    """Test complete GET_SIGNAL request/response cycle."""
    
    def test_full_get_signal_flow(self, zmq_client, mock_prediction_service):
        """Test complete GET_SIGNAL request/response cycle."""
        # Configure mock to return buy signal
        mock_prediction_service.predict.return_value = {
            "direction": "buy",
            "confidence": 0.82,
            "expected_return": 0.0025
        }
        
        # Send GET_SIGNAL request
        request = {
            "type": "GET_SIGNAL",
            "data": {
                "symbol": "EURUSD",
                "current_price": 1.08505,
                "features": {
                    "open": 1.08450,
                    "high": 1.08550,
                    "low": 1.08400,
                    "close": 1.08505,
                    "volume": 1250,
                    "atr_14": 0.0015
                }
            }
        }
        zmq_client.send_json(request)
        
        # Receive response
        response = zmq_client.recv_json()
        
        # Verify response structure
        assert response["status"] == "success"
        assert "data" in response
        assert "timestamp" in response
        
        # Verify signal data
        data = response["data"]
        assert data["direction"] == "buy"
        assert data["confidence"] == 0.82
        assert data["expected_return"] == 0.0025
        assert "stop_loss" in data
        assert "take_profit" in data
    
    def test_get_signal_json_serialization(self, zmq_client):
        """Test GET_SIGNAL request/response JSON serialization."""
        # Send GET_SIGNAL request
        request = {
            "type": "GET_SIGNAL",
            "data": {
                "symbol": "EURUSD",
                "current_price": 1.08505,
                "features": {
                    "open": 1.08450,
                    "high": 1.08550,
                    "low": 1.08400,
                    "close": 1.08505,
                    "volume": 1250,
                    "atr_14": 0.0015
                }
            }
        }
        zmq_client.send_json(request)
        response = zmq_client.recv_json()
        
        # Verify response is valid JSON
        json_str = json.dumps(response)
        reparsed = json.loads(json_str)
        
        assert reparsed["status"] == "success"
        assert "data" in reparsed
    
    def test_get_signal_with_sell_signal(self, zmq_client, mock_prediction_service):
        """Test GET_SIGNAL with sell signal."""
        # Configure mock to return sell signal
        mock_prediction_service.predict.return_value = {
            "direction": "sell",
            "confidence": 0.78,
            "expected_return": -0.0015
        }
        
        request = {
            "type": "GET_SIGNAL",
            "data": {
                "symbol": "EURUSD",
                "current_price": 1.08505,
                "features": {
                    "open": 1.08450,
                    "high": 1.08550,
                    "low": 1.08400,
                    "close": 1.08505,
                    "volume": 1250,
                    "atr_14": 0.0015
                }
            }
        }
        zmq_client.send_json(request)
        response = zmq_client.recv_json()
        
        assert response["status"] == "success"
        assert response["data"]["direction"] == "sell"
    
    def test_get_signal_calculates_stop_loss_and_take_profit(self, zmq_client):
        """Test GET_SIGNAL calculates SL/TP correctly."""
        current_price = 150.0
        atr = 0.1
        
        request = {
            "type": "GET_SIGNAL",
            "data": {
                "symbol": "TEST",
                "current_price": current_price,
                "features": {
                    "open": 149.9,
                    "high": 150.1,
                    "low": 149.8,
                    "close": 150.0,
                    "volume": 1000,
                    "atr_14": atr
                }
            }
        }
        zmq_client.send_json(request)
        response = zmq_client.recv_json()
        
        # For buy: SL = current_price - 2*atr, TP = current_price + 3*atr
        data = response["data"]
        expected_sl = current_price - 2 * atr
        expected_tp = current_price + 3 * atr
        
        assert data["stop_loss"] == expected_sl
        assert data["take_profit"] == expected_tp


# =============================================================================
# Connection Timeout Tests
# =============================================================================

class TestConnectionTimeout:
    """Test timeout handling."""
    
    def test_client_timeout_on_no_response(self):
        """Test client timeout when server doesn't respond."""
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:9999")  # Non-existent server
        
        # Set short timeout (1 second)
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        
        request = {"type": "PING"}
        
        # Send should succeed (no server needed)
        socket.send_json(request)
        
        # Receive should timeout
        with pytest.raises(zmq.Again):
            socket.recv_json()
        
        socket.close()
        context.term()
    
    def test_socket_timeout_configuration(self, zmq_client):
        """Test socket timeout is properly configured."""
        # Verify socket has timeout set
        rcv_timeout = zmq_client.getsockopt(zmq.RCVTIMEO)
        snd_timeout = zmq_client.getsockopt(zmq.SNDTIMEO)
        
        assert rcv_timeout == 5000
        assert snd_timeout == 5000


# =============================================================================
# Sequential Requests Tests
# =============================================================================

class TestSequentialRequests:
    """Test multiple sequential requests."""
    
    def test_multiple_sequential_ping_requests(self, zmq_client):
        """Test multiple PING requests in sequence."""
        for i in range(5):
            request = {"type": "PING"}
            zmq_client.send_json(request)
            response = zmq_client.recv_json()
            
            assert response["status"] == "success"
            assert response["data"] == "PONG"
    
    def test_multiple_sequential_get_signal_requests(self, zmq_client):
        """Test multiple GET_SIGNAL requests in sequence."""
        for i in range(3):
            request = {
                "type": "GET_SIGNAL",
                "data": {
                    "symbol": f"PAIR{i}",
                    "current_price": 100.0 + i,
                    "features": {
                        "open": 99.9 + i,
                        "high": 100.1 + i,
                        "low": 99.8 + i,
                        "close": 100.0 + i,
                        "volume": 1000,
                        "atr_14": 0.01
                    }
                }
            }
            zmq_client.send_json(request)
            response = zmq_client.recv_json()
            
            assert response["status"] == "success"
            assert "data" in response
    
    def test_mixed_request_types_in_sequence(self, zmq_client):
        """Test different request types in sequence."""
        # PING
        zmq_client.send_json({"type": "PING"})
        response = zmq_client.recv_json()
        assert response["status"] == "success"
        
        # GET_SIGNAL
        zmq_client.send_json({
            "type": "GET_SIGNAL",
            "data": {
                "symbol": "EURUSD",
                "current_price": 1.08505,
                "features": {
                    "open": 1.08450,
                    "high": 1.08550,
                    "low": 1.08400,
                    "close": 1.08505,
                    "volume": 1250,
                    "atr_14": 0.0015
                }
            }
        })
        response = zmq_client.recv_json()
        assert response["status"] == "success"
        
        # HEARTBEAT
        zmq_client.send_json({"type": "HEARTBEAT"})
        response = zmq_client.recv_json()
        assert response["status"] == "success"
    
    def test_request_response_order_preserved(self, zmq_client):
        """Test request/response order is preserved."""
        # Send 3 PING requests
        for i in range(3):
            zmq_client.send_json({"type": "PING"})
        
        # Receive 3 responses in order
        for i in range(3):
            response = zmq_client.recv_json()
            assert response["status"] == "success"
            assert response["data"] == "PONG"


# =============================================================================
# Concurrent Requests Tests
# =============================================================================

class TestConcurrentRequests:
    """Test concurrent requests from multiple threads."""
    
    def test_concurrent_ping_requests(self, zmq_server):
        """Test concurrent PING requests from multiple threads."""
        results = []
        errors = []
        
        def send_ping():
            """Send PING request from thread."""
            try:
                context = zmq.Context()
                socket = context.socket(zmq.REQ)
                socket.connect("tcp://localhost:5556")
                socket.setsockopt(zmq.RCVTIMEO, 5000)
                socket.setsockopt(zmq.SNDTIMEO, 5000)
                
                request = {"type": "PING"}
                socket.send_json(request)
                response = socket.recv_json()
                
                results.append(response)
                socket.close()
                context.term()
            except Exception as e:
                errors.append(str(e))
        
        # Create 5 threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=send_ping)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=10)
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        
        for response in results:
            assert response["status"] == "success"
            assert response["data"] == "PONG"
    
    def test_concurrent_get_signal_requests(self, zmq_server, mock_prediction_service):
        """Test concurrent GET_SIGNAL requests from multiple threads."""
        results = []
        errors = []
        
        def send_signal(symbol_id):
            """Send GET_SIGNAL request from thread."""
            try:
                context = zmq.Context()
                socket = context.socket(zmq.REQ)
                socket.connect("tcp://localhost:5556")
                socket.setsockopt(zmq.RCVTIMEO, 5000)
                socket.setsockopt(zmq.SNDTIMEO, 5000)
                
                request = {
                    "type": "GET_SIGNAL",
                    "data": {
                        "symbol": f"PAIR{symbol_id}",
                        "current_price": 100.0 + symbol_id,
                        "features": {
                            "open": 99.9 + symbol_id,
                            "high": 100.1 + symbol_id,
                            "low": 99.8 + symbol_id,
                            "close": 100.0 + symbol_id,
                            "volume": 1000,
                            "atr_14": 0.01
                        }
                    }
                }
                socket.send_json(request)
                response = socket.recv_json()
                
                results.append(response)
                socket.close()
                context.term()
            except Exception as e:
                errors.append(str(e))
        
        # Create 3 threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=send_signal, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=10)
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3
        
        for response in results:
            assert response["status"] == "success"
            assert "data" in response


# =============================================================================
# Protocol Compliance Tests
# =============================================================================

class TestProtocolCompliance:
    """Test protocol compliance."""
    
    def test_response_has_status_field(self, zmq_client):
        """Test all responses have status field."""
        zmq_client.send_json({"type": "PING"})
        response = zmq_client.recv_json()
        
        assert "status" in response
        assert response["status"] in ["success", "error"]
    
    def test_response_has_timestamp_field(self, zmq_client):
        """Test all responses have timestamp field."""
        zmq_client.send_json({"type": "PING"})
        response = zmq_client.recv_json()
        
        assert "timestamp" in response
        
        # Verify ISO 8601 format
        timestamp_str = response["timestamp"]
        assert "T" in timestamp_str
    
    def test_success_response_has_data_field(self, zmq_client):
        """Test success responses have data field."""
        zmq_client.send_json({"type": "PING"})
        response = zmq_client.recv_json()
        
        assert response["status"] == "success"
        assert "data" in response
    
    def test_error_response_has_error_field(self, zmq_client):
        """Test error responses have error field."""
        zmq_client.send_json({"type": "INVALID_TYPE"})
        response = zmq_client.recv_json()
        
        assert response["status"] == "error"
        assert "error" in response
        assert isinstance(response["error"], str)
    
    def test_error_response_no_data_field(self, zmq_client):
        """Test error responses don't have data field."""
        zmq_client.send_json({"type": "INVALID_TYPE"})
        response = zmq_client.recv_json()
        
        assert response["status"] == "error"
        assert "data" not in response


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling."""
    
    def test_invalid_message_type_returns_error(self, zmq_client):
        """Test invalid message type returns error."""
        zmq_client.send_json({"type": "INVALID_TYPE"})
        response = zmq_client.recv_json()
        
        assert response["status"] == "error"
        assert "error" in response
        assert "Unknown message type" in response["error"]
    
    def test_get_signal_without_data_returns_error(self, zmq_client):
        """Test GET_SIGNAL without data field returns error."""
        zmq_client.send_json({"type": "GET_SIGNAL"})
        response = zmq_client.recv_json()
        
        assert response["status"] == "error"
        assert "error" in response
    
    def test_get_signal_without_features_returns_error(self, zmq_client):
        """Test GET_SIGNAL without features returns error."""
        zmq_client.send_json({
            "type": "GET_SIGNAL",
            "data": {
                "symbol": "EURUSD",
                "current_price": 1.08505
            }
        })
        response = zmq_client.recv_json()
        
        assert response["status"] == "error"
        assert "error" in response
    
    def test_server_continues_after_error(self, zmq_client):
        """Test server continues after error."""
        # Send invalid request
        zmq_client.send_json({"type": "INVALID"})
        response = zmq_client.recv_json()
        assert response["status"] == "error"
        
        # Server should still handle valid requests
        zmq_client.send_json({"type": "PING"})
        response = zmq_client.recv_json()
        assert response["status"] == "success"


# =============================================================================
# Heartbeat Tests
# =============================================================================

class TestHeartbeat:
    """Test HEARTBEAT request."""
    
    def test_heartbeat_returns_status(self, zmq_client):
        """Test HEARTBEAT returns server status."""
        zmq_client.send_json({"type": "HEARTBEAT"})
        response = zmq_client.recv_json()
        
        assert response["status"] == "success"
        assert "data" in response
        assert "timestamp" in response
    
    def test_heartbeat_includes_uptime(self, zmq_client):
        """Test HEARTBEAT includes uptime_seconds."""
        zmq_client.send_json({"type": "HEARTBEAT"})
        response = zmq_client.recv_json()
        
        data = response["data"]
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], int)
        assert data["uptime_seconds"] >= 0
    
    def test_heartbeat_includes_request_count(self, zmq_client):
        """Test HEARTBEAT includes requests_processed."""
        zmq_client.send_json({"type": "HEARTBEAT"})
        response = zmq_client.recv_json()
        
        data = response["data"]
        assert "requests_processed" in data
        assert isinstance(data["requests_processed"], int)
    
    def test_heartbeat_includes_model_info(self, zmq_client):
        """Test HEARTBEAT includes model_info."""
        zmq_client.send_json({"type": "HEARTBEAT"})
        response = zmq_client.recv_json()
        
        data = response["data"]
        assert "model_info" in data
        assert isinstance(data["model_info"], dict)
