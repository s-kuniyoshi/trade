# Model Registry Implementation - Architectural Decisions

## 1. Filesystem-Based Storage (vs Database)

**Decision**: Use filesystem-only storage with JSON metadata

**Rationale**:
- Simplicity: No database dependency, easier deployment
- Portability: Models and metadata travel together
- Debugging: Easy to inspect version directories
- Performance: Direct file access, no query overhead
- Backward Compatibility: Old models can be imported without registry

**Trade-offs**:
- Limited query capabilities (mitigated by in-memory filtering)
- No transactional guarantees (mitigated by atomic operations)
- Scaling limitations (acceptable for model registry use case)

## 2. Timestamp-Based Versioning

**Decision**: Use `YYYYMMDD_HHMMSS` format for version names

**Rationale**:
- Chronological Sorting: Natural sorting by date
- Human Readable: Easy to identify when model was trained
- Collision Resistant: 1-second granularity sufficient
- No External Dependencies: Pure datetime formatting
- Deterministic: Same training time = same version name

**Alternatives Considered**:
- Semantic versioning (1.0.0): Requires manual management
- UUID: Not human-readable, harder to debug
- Sequential numbers: Requires state management

## 3. Symlink for Production Model

**Decision**: Use symlink `models/production/latest` → `models/{version}`

**Rationale**:
- Atomic Promotion: Single symlink update = production change
- Rollback Support: Easy to point to previous version
- Inference Integration: `PredictionService` loads from fixed path
- Audit Trail: Symlink target shows current production version

**Windows Compatibility**:
- Fallback to directory copy if symlink fails
- Handles permission issues gracefully
- Maintains functionality on all platforms

## 4. Thread-Safe Operations

**Decision**: Use `threading.Lock()` on all public methods

**Rationale**:
- Concurrent Access: Multiple threads may register/promote simultaneously
- Filesystem Safety: Prevents race conditions on file operations
- Metadata Consistency: Ensures atomic read-modify-write cycles
- Production Stability: Prevents concurrent promotion attempts

**Implementation**:
- Single lock per registry instance
- Coarse-grained locking (entire method)
- Acceptable for model registry (not high-frequency operations)

## 5. Metadata Enrichment Strategy

**Decision**: Enhance TrainingResult metadata with registry-specific fields

**Rationale**:
- Separation of Concerns: TrainingResult focuses on training, registry adds deployment info
- Extensibility: Easy to add new metadata fields
- Backward Compatibility: TrainingResult unchanged
- Audit Trail: deployment_status and promoted_at track lifecycle

**Metadata Fields**:
- `version`: Version name (added by registry)
- `training_date`: ISO timestamp (from TrainingResult)
- `data_range`: Optional start/end dates (user-provided)
- `deployment_status`: "registered" or "production"
- `promoted_at`: ISO timestamp when promoted
- `metrics`: From TrainingResult (direction_accuracy, rmse, etc.)
- `feature_names`: From TrainingResult

## 6. Metric Comparison Logic

**Decision**: Classify metrics as "higher is better" or "lower is better"

**Rationale**:
- Objective Comparison: Removes ambiguity about metric direction
- Automated Winner Selection: No manual interpretation needed
- Extensible: Easy to add new metrics to classifications
- Clear Output: Comparison dict shows winner for each metric

**Classifications**:
- Higher is Better: direction_accuracy, profit_factor, sharpe_ratio, r_squared, annual_return, total_return, win_rate
- Lower is Better: rmse, mae, max_drawdown, volatility

## 7. Cleanup Strategy

**Decision**: Keep N latest versions, never delete production

**Rationale**:
- Bounded Storage: Prevents unlimited disk usage
- Production Protection: Current model always available
- Rollback Support: Keep recent versions for quick rollback
- Configurable: `keep_versions` from config

**Implementation**:
- Sort versions by date (newest first)
- Keep first N versions
- Check if version is production (via symlink target)
- Delete entire version directory with `shutil.rmtree()`

## 8. Error Handling Strategy

**Decision**: Raise exceptions with context, log all operations

**Rationale**:
- Fail Fast: Errors propagate immediately
- Debugging: Comprehensive logging for troubleshooting
- Audit Trail: All operations logged (register, promote, cleanup)
- Context: Error messages include version names, paths

**Error Types**:
- `ValueError`: Version not found, invalid input
- `FileNotFoundError`: Metadata missing
- `RuntimeError`: Registration/promotion failures
- Filesystem errors: Logged and re-raised with context

## 9. Cross-Platform Path Handling

**Decision**: Use `pathlib.Path` for all path operations

**Rationale**:
- Platform Agnostic: Handles Windows/Linux path differences
- Type Safe: Path objects prevent string concatenation errors
- Readable: Cleaner syntax than os.path
- Symlink Support: Built-in symlink methods

**Symlink Fallback**:
- Try symlink first (Linux/Mac)
- Fallback to directory copy on Windows
- Handles permission issues gracefully
- Maintains functionality on all platforms

## 10. Configuration Integration

**Decision**: Read registry settings from `config/model.yaml`

**Rationale**:
- Centralized Configuration: All settings in one place
- Environment Specific: Can vary by deployment
- Type Safe: Pydantic validation
- Singleton Pattern: Consistent config access

**Configuration Fields**:
- `registry.path`: Base directory (default: "models/")
- `registry.keep_versions`: Number to keep (default: 5)
- `registry.metadata`: List of metadata fields to store

## 11. Public API Design

**Decision**: Expose 8 public methods + 1 dataclass

**Rationale**:
- Clear Interface: Each method has single responsibility
- Composable: Methods can be chained (register → promote → cleanup)
- Extensible: Easy to add new methods without breaking existing code
- Discoverable: Method names clearly indicate functionality

**Public Methods**:
1. `register()`: Add new model version
2. `promote_to_production()`: Make version production
3. `list_versions()`: Query available versions
4. `load_version()`: Load specific or latest version
5. `compare_versions()`: Compare two versions
6. `cleanup_old_versions()`: Delete old versions
7. `rollback_to_version()`: Revert to previous version
8. `get_version_info()`: Get version metadata

**Public Dataclass**:
- `ModelVersion`: Summary of version (version, training_date, metrics, features, path)

## 12. Logging Strategy

**Decision**: Use `get_logger("training.registry")` with appropriate levels

**Rationale**:
- Consistent Logging: Matches project logging pattern
- Audit Trail: All operations logged
- Debugging: Detailed info for troubleshooting
- Production Ready: Appropriate log levels

**Log Levels**:
- INFO: Normal operations (register, promote, cleanup, load)
- WARNING: Unusual but handled situations (symlink fallback, production skipped)
- ERROR: Failures with context (failed operations)

## 13. Type Hints Coverage

**Decision**: Complete type annotations throughout

**Rationale**:
- IDE Support: Better autocomplete and error detection
- Documentation: Types serve as inline documentation
- Maintainability: Easier to understand code intent
- Testing: Type checkers catch errors early

**Coverage**:
- All method parameters and returns
- Dataclass fields
- Dict/List types with element types
- Optional types with `| None`

## 14. Docstring Format

**Decision**: Follow trainer.py pattern with Args/Returns/Raises

**Rationale**:
- Consistency: Matches existing codebase style
- Completeness: All information in one place
- IDE Integration: Docstrings shown in hover/autocomplete
- Maintainability: Easy to update documentation

**Format**:
- Module-level docstring
- Class docstring with usage example
- Method docstrings with Args, Returns, Raises sections
- Helper method docstrings
