# ZeroMQ Communication Protocol Specification

This document defines the communication protocol between the MetaTrader 5 (MT5) Expert Advisor (EA) and the Python Prediction Service using ZeroMQ.

## 1. Overview

### Architecture
The system uses a client-server architecture where the MT5 EA acts as the client and the Python Prediction Service acts as the server.

```text
+------------------+           +-------------------------+
|      MT5 EA      |           | Python Prediction Svc   |
|     (Client)     |           |        (Server)         |
|                  |           |                         |
|  [CZeroMQ REQ]   | <-------> |     [ZMQServer REP]     |
+------------------+    TCP    +-------------------------+
      (Port 5555)
```

### Socket Pattern
- **Pattern**: Request-Reply (REQ/REP)
- **Transport**: TCP
- **Format**: JSON-encoded strings
- **Encoding**: UTF-8

### Configuration
- **Default Host**: `127.0.0.1`
- **Default Port**: `5555`
- **Timeout**: 5000ms (configurable in `trading.yaml`)

---

## 2. Message Types

The protocol supports three primary message types:

1. **PING**: A lightweight health check to verify connectivity and server responsiveness.
2. **GET_SIGNAL**: A request for a trading signal based on provided market features.
3. **HEARTBEAT**: A request for server status, uptime, and performance statistics.

---

## 3. Request/Response Formats

### Common Response Structure
All responses follow a consistent envelope format:

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "success" or "error" |
| `data` | object/string | Payload data (present on success) |
| `error` | string | Error message (present on error) |
| `timestamp` | string | ISO 8601 UTC timestamp |

---

### PING
**Purpose**: Health check to verify server is responsive.

**Request Format:**
```json
{
  "type": "PING"
}
```

**Response Format:**
```json
{
  "status": "success",
  "data": "PONG",
  "timestamp": "2026-01-26T10:30:01.123456+00:00"
}
```

---

### GET_SIGNAL
**Purpose**: Request a trading signal for a specific symbol and timeframe.

**Request Format:**
```json
{
  "type": "GET_SIGNAL",
  "data": {
    "symbol": "EURUSD",
    "timeframe": "H1",
    "current_price": 1.08505,
    "features": {
      "open": 1.08450,
      "high": 1.08550,
      "low": 1.08400,
      "close": 1.08505,
      "volume": 1250,
      "atr_14": 0.0015
    }
  },
  "timestamp": "2026-01-26 10:30:00"
}
```

**Response Format:**
```json
{
  "status": "success",
  "data": {
    "direction": "buy",
    "confidence": 0.82,
    "expected_return": 0.0025,
    "stop_loss": 1.08205,
    "take_profit": 1.08955
  },
  "timestamp": "2026-01-26T10:30:00.500000+00:00"
}
```

**Data Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `direction` | string | "buy", "sell", or "hold" |
| `confidence` | float | Model confidence score (0.0 to 1.0) |
| `expected_return` | float | Predicted price change |
| `stop_loss` | float | Suggested stop loss price |
| `take_profit` | float | Suggested take profit price |

---

### HEARTBEAT
**Purpose**: Retrieve server health and performance metrics.

**Request Format:**
```json
{
  "type": "HEARTBEAT"
}
```

**Response Format:**
```json
{
  "status": "success",
  "data": {
    "uptime_seconds": 3600,
    "requests_processed": 150,
    "errors": 2,
    "model_info": {
      "name": "XGBoost-Classifier",
      "version": "1.2.0",
      "last_trained": "2026-01-20T15:00:00Z"
    }
  },
  "timestamp": "2026-01-26T11:00:00.000000+00:00"
}
```

---

## 4. Error Handling

### Error Response Format
When a request fails, the server returns a response with `status: "error"`.

```json
{
  "status": "error",
  "error": "Missing features in request",
  "timestamp": "2026-01-26T10:35:00.000000+00:00"
}
```

### Common Error Scenarios
- **Unknown Message Type**: Sent when the `type` field is invalid.
- **Missing Data**: Sent when `GET_SIGNAL` is called without the `data` object.
- **Invalid Features**: Sent when required features for prediction are missing.
- **Model Error**: Sent if the prediction service fails internally.

---

## 5. Timeouts and Configuration

### Configuration (`trading.yaml`)
The communication parameters are defined in the system configuration:

```yaml
communication:
  zeromq:
    request_port: 5555
    timeout_ms: 5000
```

### Client-Side Behavior (MQL5)
- **Timeout**: The `CZeroMQ` class sets `ZMQ_RCVTIMEO` and `ZMQ_SNDTIMEO` to 5000ms.
- **Retries**: The EA implements a connection failure counter. If 5 consecutive failures occur, it logs a critical error.
- **Blocking**: Requests are synchronous. The EA will wait for the response or timeout before continuing execution.

---

## 6. Example Communication Flows

### Successful Signal Flow
1. **EA**: Sends `GET_SIGNAL` with OHLCV data.
2. **Server**: Receives request, parses features, runs model.
3. **Server**: Calculates SL/TP based on ATR and current price.
4. **Server**: Sends `success` response with signal data.
5. **EA**: Parses JSON, verifies confidence > `MinConfidence`, executes trade.

### Error Flow (Timeout)
1. **EA**: Sends `GET_SIGNAL`.
2. **Server**: Busy or offline.
3. **EA**: Waits for 5000ms.
4. **EA**: `zmq_recv` returns `-1`.
5. **EA**: Logs "Failed to receive response (timeout or error)".

---

## 7. MQL5 Integration Guide

### Using CZeroMQ Class
The `CZeroMQ` class provides a high-level wrapper for the protocol.

```mql5
#include <ZeroMQ.mqh>

CZeroMQ g_zmq;

int OnInit() {
   // 1. Connect
   if(!g_zmq.Connect("127.0.0.1", 5555)) return INIT_FAILED;
   
   // 2. Verify with Ping
   if(!g_zmq.Ping()) return INIT_FAILED;
   
   return INIT_SUCCEEDED;
}

void OnTick() {
   string response;
   string features = "{\"open\": 1.08, \"high\": 1.09, \"low\": 1.07, \"close\": 1.085, \"volume\": 100}";
   
   // 3. Request Signal
   if(g_zmq.GetSignal(_Symbol, "H1", features, response)) {
      // 4. Parse response (using helper functions)
      double confidence = ParseConfidence(response);
      if(confidence > 0.65) {
         // Execute trade...
      }
   }
}
```

### Best Practices
1. **Always Ping on Init**: Ensure the server is reachable before starting the EA.
2. **Validate Confidence**: Never execute signals with confidence below the configured threshold.
3. **Handle Timeouts**: Check the return value of `GetSignal` to handle network issues gracefully.
4. **JSON Formatting**: Ensure the `features` string is valid JSON to avoid server-side parsing errors.
