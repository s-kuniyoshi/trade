
## Task: Create ZeroMQ Communication Protocol Specification (PROTOCOL.md)

### Architectural Decisions

#### 1. Standardized Response Envelope
**Decision**: All responses must include `status`, `data`, `error`, and `timestamp`.
**Rationale**: 
- Simplifies parsing logic in MQL5.
- Provides a consistent way to handle both success and error states.
- Ensures every message is traceable via timestamp.

#### 2. Server-Side SL/TP Calculation
**Decision**: The Python server calculates Stop Loss and Take Profit levels.
**Rationale**:
- Centralizes trading logic in the Python service.
- Allows for more complex risk management (e.g., ATR-based) without bloating the EA code.
- Makes the EA a "thin client" focused on execution.

#### 3. Synchronous REQ/REP Pattern
**Decision**: Use blocking REQ/REP sockets for the initial protocol.
**Rationale**:
- Simplest pattern to implement and debug.
- Ensures the EA has a signal before proceeding with a tick.
- Avoids complex state management required for asynchronous communication.

#### 4. JSON-Only Payload
**Decision**: All communication must be valid JSON strings.
**Rationale**:
- Human-readable and easy to debug.
- Supported by both Python and MQL5 (via string manipulation or libraries).
- Flexible for future additions to the message schema.

#### 5. ISO 8601 UTC Timestamps
**Decision**: Use `YYYY-MM-DDTHH:MM:SS.ffffff+00:00` format.
**Rationale**:
- Industry standard for time representation.
- Avoids timezone ambiguity between MT5 (broker time) and Python (system/UTC time).

## Task: Create Comprehensive Unit Tests for ZMQServer (test_zmq_server.py)

### Testing Strategy Decisions

#### 1. Test Organization by Functionality
**Decision**: Group tests into logical classes (TestPingRequest, TestGetSignalRequest, etc.)
**Rationale**:
- Improves readability and maintainability
- Makes it easy to find tests for specific functionality
- Allows running tests by class: `pytest tests/test_zmq_server.py::TestPingRequest`
- Follows pytest conventions for test organization

#### 2. Mock-Based Testing Approach
**Decision**: Use `unittest.mock.Mock` for PredictionService instead of real implementation
**Rationale**:
- Avoids external dependencies (no need for actual ML model)
- Enables testing of error scenarios easily (side_effect)
- Allows testing with different prediction outputs
- Faster test execution
- Isolates ZMQServer logic from PredictionService logic

#### 3. Fixture-Based Setup
**Decision**: Use pytest fixtures for mock_prediction_service and zmq_server
**Rationale**:
- Reduces code duplication across tests
- Makes fixtures reusable and composable
- Allows easy modification of test setup
- Follows pytest best practices

#### 4. Comprehensive Error Scenario Coverage
**Decision**: Test all error paths: unknown type, missing data, missing features, exceptions
**Rationale**:
- Ensures robustness of error handling
- Validates error response format compliance
- Tests edge cases that might occur in production
- Verifies server doesn't crash on invalid input

#### 5. Protocol Compliance Validation
**Decision**: Create dedicated test class for response format validation
**Rationale**:
- Ensures all responses follow PROTOCOL.md specification
- Validates ISO 8601 timestamp format
- Checks required fields (status, data/error, timestamp)
- Prevents protocol drift over time

#### 6. Integration Testing
**Decision**: Include tests for multiple requests and mixed request types
**Rationale**:
- Verifies server state management across requests
- Tests that errors don't break subsequent requests
- Validates statistics tracking across multiple requests
- Ensures server is stateless between requests

#### 7. Edge Case Testing
**Decision**: Create dedicated test class for boundary conditions
**Rationale**:
- Tests empty dicts, None values, missing fields
- Validates case sensitivity of message types
- Tests default value handling (e.g., missing ATR)
- Ensures graceful degradation

#### 8. Test Naming Convention
**Decision**: Use `test_<function>_<scenario>()` naming pattern
**Rationale**:
- Makes test purpose immediately clear
- Enables easy test discovery and filtering
- Follows pytest conventions
- Improves test documentation

#### 9. Assertion Strategy
**Decision**: Use explicit assert statements with clear conditions
**Rationale**:
- More readable than implicit assertions
- Easier to debug when tests fail
- Clear about what is being tested
- Follows pytest best practices

#### 10. Statistics Testing Approach
**Decision**: Test error_count increments only on exceptions, not on error responses
**Rationale**:
- Matches actual implementation behavior
- Distinguishes between handled errors (error responses) and unexpected errors (exceptions)
- Validates that error_count tracks critical failures
- Prevents false positives in error tracking

### Test Coverage Metrics
- **Total Test Cases**: 40
- **Lines of Test Code**: 738
- **Coverage Areas**: 9 (PING, GET_SIGNAL, HEARTBEAT, Errors, Stats, Confidence, Integration, Format, Edge Cases)
- **Message Types Tested**: 3 (PING, GET_SIGNAL, HEARTBEAT)
- **Error Scenarios**: 6+
- **Edge Cases**: 5+

### Future Testing Considerations
1. **Performance Tests**: Add tests for response time under load
2. **Concurrency Tests**: Test thread-safety of statistics tracking
3. **Integration Tests**: Test with actual ZMQ sockets (separate test suite)
4. **Regression Tests**: Add tests for specific bugs as they're discovered
5. **Parametrized Tests**: Use pytest.mark.parametrize for testing multiple signal directions
