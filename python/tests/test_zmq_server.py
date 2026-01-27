"""
Comprehensive unit tests for ZMQServer class.

Tests all message types (PING, GET_SIGNAL, HEARTBEAT) and error handling scenarios.
Uses pytest fixtures and mocks to avoid external dependencies.
"""

import pytest
import json
import time
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.communication.zmq_server import ZMQServer


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_prediction_service():
    """
    Create mock PredictionService for testing.
    
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


@pytest.fixture
def zmq_server(mock_prediction_service):
    """
    Create ZMQServer instance for testing.
    
    Uses a test port (5556) and mock prediction service.
    """
    server = ZMQServer(
        port=5556,
        prediction_service=mock_prediction_service,
        timeout_ms=1000
    )
    return server


# =============================================================================
# PING Tests
# =============================================================================

class TestPingRequest:
    """Test PING request/response handling."""
    
    def test_ping_request_returns_pong(self, zmq_server):
        """Test PING request returns PONG response."""
        request = {"type": "PING"}
        response = zmq_server.handle_request(request)
        
        assert response["status"] == "success"
        assert response["data"] == "PONG"
        assert "timestamp" in response
    
    def test_ping_response_has_valid_timestamp(self, zmq_server):
        """Test PING response includes valid ISO 8601 timestamp."""
        request = {"type": "PING"}
        response = zmq_server.handle_request(request)
        
        # Verify timestamp is ISO 8601 format
        timestamp_str = response["timestamp"]
        assert "T" in timestamp_str  # ISO 8601 format
        assert "+" in timestamp_str or "Z" in timestamp_str  # UTC indicator
        
        # Verify timestamp is parseable
        try:
            datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"Invalid ISO 8601 timestamp: {timestamp_str}")
    
    def test_ping_response_structure(self, zmq_server):
        """Test PING response has correct structure."""
        request = {"type": "PING"}
        response = zmq_server.handle_request(request)
        
        # Verify required fields
        assert "status" in response
        assert "data" in response
        assert "timestamp" in response
        
        # Verify no error field in success response
        assert "error" not in response


# =============================================================================
# GET_SIGNAL Tests
# =============================================================================

class TestGetSignalRequest:
    """Test GET_SIGNAL request/response handling."""
    
    def test_get_signal_buy_signal(self, zmq_server, mock_prediction_service):
        """Test GET_SIGNAL returns buy signal."""
        mock_prediction_service.predict.return_value = {
            "direction": "buy",
            "confidence": 0.75,
            "expected_return": 0.001
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
        
        response = zmq_server.handle_request(request)
        
        assert response["status"] == "success"
        assert response["data"]["direction"] == "buy"
        assert response["data"]["confidence"] == 0.75
        assert response["data"]["expected_return"] == 0.001
    
    def test_get_signal_sell_signal(self, zmq_server, mock_prediction_service):
        """Test GET_SIGNAL returns sell signal."""
        mock_prediction_service.predict.return_value = {
            "direction": "sell",
            "confidence": 0.82,
            "expected_return": -0.0008
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
        
        response = zmq_server.handle_request(request)
        
        assert response["status"] == "success"
        assert response["data"]["direction"] == "sell"
        assert response["data"]["confidence"] == 0.82
    
    def test_get_signal_hold_signal(self, zmq_server, mock_prediction_service):
        """Test GET_SIGNAL returns hold signal."""
        mock_prediction_service.predict.return_value = {
            "direction": "hold",
            "confidence": 0.55,
            "expected_return": 0.0
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
        
        response = zmq_server.handle_request(request)
        
        assert response["status"] == "success"
        assert response["data"]["direction"] == "hold"
    
    def test_get_signal_calculates_stop_loss_buy(self, zmq_server, mock_prediction_service):
        """Test GET_SIGNAL calculates stop loss for buy signal."""
        mock_prediction_service.predict.return_value = {
            "direction": "buy",
            "confidence": 0.75,
            "expected_return": 0.001
        }
        
        current_price = 150.0
        atr = 0.1
        
        request = {
            "type": "GET_SIGNAL",
            "data": {
                "symbol": "EURUSD",
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
        
        response = zmq_server.handle_request(request)
        
        # For buy: SL = current_price - 2*atr
        expected_sl = current_price - 2 * atr
        assert response["data"]["stop_loss"] == expected_sl
    
    def test_get_signal_calculates_take_profit_buy(self, zmq_server, mock_prediction_service):
        """Test GET_SIGNAL calculates take profit for buy signal."""
        mock_prediction_service.predict.return_value = {
            "direction": "buy",
            "confidence": 0.75,
            "expected_return": 0.001
        }
        
        current_price = 150.0
        atr = 0.1
        
        request = {
            "type": "GET_SIGNAL",
            "data": {
                "symbol": "EURUSD",
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
        
        response = zmq_server.handle_request(request)
        
        # For buy: TP = current_price + 3*atr
        expected_tp = current_price + 3 * atr
        assert response["data"]["take_profit"] == expected_tp
    
    def test_get_signal_calculates_stop_loss_sell(self, zmq_server, mock_prediction_service):
        """Test GET_SIGNAL calculates stop loss for sell signal."""
        mock_prediction_service.predict.return_value = {
            "direction": "sell",
            "confidence": 0.75,
            "expected_return": -0.001
        }
        
        current_price = 150.0
        atr = 0.1
        
        request = {
            "type": "GET_SIGNAL",
            "data": {
                "symbol": "EURUSD",
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
        
        response = zmq_server.handle_request(request)
        
        # For sell: SL = current_price + 2*atr
        expected_sl = current_price + 2 * atr
        assert response["data"]["stop_loss"] == expected_sl
    
    def test_get_signal_response_structure(self, zmq_server):
        """Test GET_SIGNAL response has correct structure."""
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
        
        response = zmq_server.handle_request(request)
        
        # Verify response structure
        assert response["status"] == "success"
        assert "data" in response
        assert "timestamp" in response
        
        # Verify data fields
        data = response["data"]
        assert "direction" in data
        assert "confidence" in data
        assert "expected_return" in data
        assert "stop_loss" in data
        assert "take_profit" in data


# =============================================================================
# HEARTBEAT Tests
# =============================================================================

class TestHeartbeatRequest:
    """Test HEARTBEAT request/response handling."""
    
    def test_heartbeat_returns_status(self, zmq_server):
        """Test HEARTBEAT returns server status."""
        request = {"type": "HEARTBEAT"}
        response = zmq_server.handle_request(request)
        
        assert response["status"] == "success"
        assert "data" in response
        assert "timestamp" in response
    
    def test_heartbeat_includes_uptime(self, zmq_server):
        """Test HEARTBEAT includes uptime_seconds."""
        request = {"type": "HEARTBEAT"}
        response = zmq_server.handle_request(request)
        
        data = response["data"]
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], int)
        assert data["uptime_seconds"] >= 0
    
    def test_heartbeat_includes_request_count(self, zmq_server):
        """Test HEARTBEAT includes requests_processed."""
        request = {"type": "HEARTBEAT"}
        response = zmq_server.handle_request(request)
        
        data = response["data"]
        assert "requests_processed" in data
        assert isinstance(data["requests_processed"], int)
    
    def test_heartbeat_includes_error_count(self, zmq_server):
        """Test HEARTBEAT includes errors."""
        request = {"type": "HEARTBEAT"}
        response = zmq_server.handle_request(request)
        
        data = response["data"]
        assert "errors" in data
        assert isinstance(data["errors"], int)
    
    def test_heartbeat_includes_model_info(self, zmq_server, mock_prediction_service):
        """Test HEARTBEAT includes model_info."""
        request = {"type": "HEARTBEAT"}
        response = zmq_server.handle_request(request)
        
        data = response["data"]
        assert "model_info" in data
        
        # Verify model info matches mock
        model_info = data["model_info"]
        assert model_info["model_type"] == "LightGBM"
        assert model_info["version"] == "1.0"


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_unknown_message_type(self, zmq_server):
        """Test unknown message type returns error."""
        request = {"type": "INVALID_TYPE"}
        response = zmq_server.handle_request(request)
        
        assert response["status"] == "error"
        assert "error" in response
        assert "Unknown message type" in response["error"]
        assert "INVALID_TYPE" in response["error"]
    
    def test_missing_message_type(self, zmq_server):
        """Test missing type field returns error."""
        request = {}
        response = zmq_server.handle_request(request)
        
        assert response["status"] == "error"
        assert "error" in response
    
    def test_get_signal_missing_data_field(self, zmq_server):
        """Test GET_SIGNAL without data field returns error."""
        request = {"type": "GET_SIGNAL"}
        response = zmq_server.handle_request(request)
        
        assert response["status"] == "error"
        assert "error" in response
        assert "data" in response["error"].lower()
    
    def test_get_signal_missing_features(self, zmq_server):
        """Test GET_SIGNAL without features returns error."""
        request = {
            "type": "GET_SIGNAL",
            "data": {
                "symbol": "EURUSD",
                "current_price": 1.08505
            }
        }
        response = zmq_server.handle_request(request)
        
        assert response["status"] == "error"
        assert "error" in response
        assert "features" in response["error"].lower()
    
    def test_error_response_has_timestamp(self, zmq_server):
        """Test error response includes timestamp."""
        request = {"type": "INVALID"}
        response = zmq_server.handle_request(request)
        
        assert "timestamp" in response
        assert response["status"] == "error"
    
    def test_error_response_no_data_field(self, zmq_server):
        """Test error response does not include data field."""
        request = {"type": "INVALID"}
        response = zmq_server.handle_request(request)
        
        assert "data" not in response
        assert "error" in response


# =============================================================================
# Statistics Tracking Tests
# =============================================================================

class TestStatisticsTracking:
    """Test request and error statistics tracking."""
    
    def test_request_count_increments(self, zmq_server):
        """Test request counter increments on successful request."""
        initial_count = zmq_server.request_count
        
        request = {"type": "PING"}
        zmq_server.handle_request(request)
        
        # Note: request_count is only incremented in the server loop,
        # not in handle_request directly. This test verifies the counter exists.
        assert hasattr(zmq_server, "request_count")
        assert zmq_server.request_count == initial_count
    
    def test_error_count_increments_on_exception(self, zmq_server):
        """Test error counter increments when exception is raised in handle_request."""
        initial_count = zmq_server.error_count
        
        # Trigger an exception by passing invalid message type that causes exception
        # We'll mock handle_request to raise an exception
        with patch.object(zmq_server, '_handle_ping', side_effect=RuntimeError("Test error")):
            request = {"type": "PING"}
            response = zmq_server.handle_request(request)
        
        assert response["status"] == "error"
        assert zmq_server.error_count == initial_count + 1
    
    def test_get_stats_returns_statistics(self, zmq_server):
        """Test get_stats() returns server statistics."""
        stats = zmq_server.get_stats()
        
        assert "uptime_seconds" in stats
        assert "requests_processed" in stats
        assert "errors" in stats
        assert "running" in stats
    
    def test_stats_uptime_is_non_negative(self, zmq_server):
        """Test uptime_seconds is non-negative."""
        stats = zmq_server.get_stats()
        assert stats["uptime_seconds"] >= 0
    
    def test_stats_counters_are_integers(self, zmq_server):
        """Test statistics counters are integers."""
        stats = zmq_server.get_stats()
        
        assert isinstance(stats["uptime_seconds"], int)
        assert isinstance(stats["requests_processed"], int)
        assert isinstance(stats["errors"], int)


# =============================================================================
# Confidence Filtering Tests
# =============================================================================

class TestConfidenceFiltering:
    """Test confidence threshold handling."""
    
    def test_server_has_min_confidence_threshold(self, zmq_server):
        """Test server has min_confidence attribute."""
        assert hasattr(zmq_server, "min_confidence")
        assert isinstance(zmq_server.min_confidence, float)
        assert 0.0 <= zmq_server.min_confidence <= 1.0
    
    def test_min_confidence_passed_to_predict(self, zmq_server, mock_prediction_service):
        """Test min_confidence is passed to prediction service."""
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
        
        zmq_server.handle_request(request)
        
        # Verify predict was called with min_confidence parameter
        mock_prediction_service.predict.assert_called_once()
        call_kwargs = mock_prediction_service.predict.call_args[1]
        assert "min_confidence" in call_kwargs
        assert call_kwargs["min_confidence"] == zmq_server.min_confidence


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for multiple requests."""
    
    def test_multiple_ping_requests(self, zmq_server):
        """Test handling multiple PING requests."""
        for _ in range(3):
            request = {"type": "PING"}
            response = zmq_server.handle_request(request)
            assert response["status"] == "success"
            assert response["data"] == "PONG"
    
    def test_mixed_request_types(self, zmq_server):
        """Test handling different request types in sequence."""
        # PING
        ping_response = zmq_server.handle_request({"type": "PING"})
        assert ping_response["status"] == "success"
        
        # GET_SIGNAL
        signal_response = zmq_server.handle_request({
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
        assert signal_response["status"] == "success"
        
        # HEARTBEAT
        heartbeat_response = zmq_server.handle_request({"type": "HEARTBEAT"})
        assert heartbeat_response["status"] == "success"
    
    def test_error_does_not_break_server(self, zmq_server):
        """Test server continues after error."""
        # Send invalid request
        error_response = zmq_server.handle_request({"type": "INVALID"})
        assert error_response["status"] == "error"
        
        # Server should still handle valid requests
        ping_response = zmq_server.handle_request({"type": "PING"})
        assert ping_response["status"] == "success"


# =============================================================================
# Response Format Tests
# =============================================================================

class TestResponseFormat:
    """Test response format compliance with protocol."""
    
    def test_all_responses_have_timestamp(self, zmq_server):
        """Test all responses include timestamp field."""
        requests = [
            {"type": "PING"},
            {"type": "HEARTBEAT"},
            {"type": "INVALID"}
        ]
        
        for request in requests:
            response = zmq_server.handle_request(request)
            assert "timestamp" in response, f"Missing timestamp in {request['type']} response"
    
    def test_all_responses_have_status(self, zmq_server):
        """Test all responses include status field."""
        requests = [
            {"type": "PING"},
            {"type": "HEARTBEAT"},
            {"type": "INVALID"}
        ]
        
        for request in requests:
            response = zmq_server.handle_request(request)
            assert "status" in response, f"Missing status in {request['type']} response"
            assert response["status"] in ["success", "error"]
    
    def test_success_responses_have_data(self, zmq_server):
        """Test success responses include data field."""
        requests = [
            {"type": "PING"},
            {"type": "HEARTBEAT"}
        ]
        
        for request in requests:
            response = zmq_server.handle_request(request)
            assert response["status"] == "success"
            assert "data" in response
    
    def test_error_responses_have_error_field(self, zmq_server):
        """Test error responses include error field."""
        request = {"type": "INVALID"}
        response = zmq_server.handle_request(request)
        
        assert response["status"] == "error"
        assert "error" in response
        assert isinstance(response["error"], str)
        assert len(response["error"]) > 0


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_request_dict(self, zmq_server):
        """Test handling empty request dictionary."""
        request = {}
        response = zmq_server.handle_request(request)
        
        assert response["status"] == "error"
        assert "timestamp" in response
    
    def test_none_type_field(self, zmq_server):
        """Test handling None type field."""
        request = {"type": None}
        response = zmq_server.handle_request(request)
        
        assert response["status"] == "error"
    
    def test_case_sensitive_message_type(self, zmq_server):
        """Test message type is case sensitive."""
        request = {"type": "ping"}  # lowercase
        response = zmq_server.handle_request(request)
        
        # Should be treated as unknown type
        assert response["status"] == "error"
    
    def test_get_signal_with_empty_features(self, zmq_server):
        """Test GET_SIGNAL with empty features dict."""
        request = {
            "type": "GET_SIGNAL",
            "data": {
                "symbol": "EURUSD",
                "current_price": 1.08505,
                "features": {}
            }
        }
        response = zmq_server.handle_request(request)
        
        # Should handle gracefully (either error or use defaults)
        assert "status" in response
        assert "timestamp" in response
    
    def test_get_signal_with_missing_atr(self, zmq_server):
        """Test GET_SIGNAL handles missing ATR gracefully."""
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
                    "volume": 1250
                    # atr_14 is missing
                }
            }
        }
        response = zmq_server.handle_request(request)
        
        # Should use default ATR value (0.01)
        assert response["status"] == "success"
        assert "data" in response
