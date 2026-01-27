"""
Model registry system for versioning, storing, comparing, and deploying trained models.

Provides version management with metadata tracking, automatic cleanup of old versions,
and production model promotion via symlinks.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from ..utils.config import get_config
from ..utils.logger import get_logger
from .trainer import TrainingResult

logger = get_logger("training.registry")


# =============================================================================
# Model Version Data Class
# =============================================================================

@dataclass
class ModelVersion:
    """Summary of a model version."""
    version: str
    training_date: str
    metrics: dict[str, float]
    features: list[str]
    path: Path


# =============================================================================
# Model Registry
# =============================================================================

class ModelRegistry:
    """
    Model registry for versioning and managing trained models.
    
    Manages model versions in filesystem with automatic versioning (timestamp-based),
    metadata tracking, version comparison, production promotion, and cleanup.
    
    Usage:
        registry = ModelRegistry()
        version = registry.register(training_result, data_range={...})
        registry.promote_to_production(version)
        versions = registry.list_versions(min_direction_accuracy=0.52)
        comparison = registry.compare_versions(v1, v2)
        registry.cleanup_old_versions()
    """
    
    def __init__(self):
        """Initialize model registry."""
        config = get_config()
        self.base_path = Path(config.model.registry.get("path", "models/"))
        self.keep_versions = config.model.registry.get("keep_versions", 5)
        self.lock = Lock()
        
        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create production directory
        self.production_dir = self.base_path / "production"
        self.production_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"ModelRegistry initialized: base_path={self.base_path}, "
            f"keep_versions={self.keep_versions}"
        )
    
    def _get_version_path(self, version: str) -> Path:
        """Get path for a specific version."""
        return self.base_path / version
    
    def _read_metadata(self, version_path: Path) -> dict[str, Any]:
        """Read metadata.json from version directory."""
        metadata_path = version_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            return json.load(f)
    
    def _write_metadata(self, version_path: Path, metadata: dict[str, Any]) -> None:
        """Write metadata.json to version directory."""
        metadata_path = version_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _generate_version_name(self) -> str:
        """Generate version name from current timestamp."""
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    def _create_symlink(self, target: Path, link: Path) -> bool:
        """
        Create symlink from link to target.
        
        Handles Windows/Linux differences and fallback to copy if symlink fails.
        
        Args:
            target: Target path (absolute or relative)
            link: Symlink path to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove existing symlink/file
            if link.exists() or link.is_symlink():
                if link.is_symlink():
                    link.unlink()
                else:
                    shutil.rmtree(link)
            
            # Try to create symlink
            try:
                link.symlink_to(target)
                logger.info(f"Created symlink: {link} -> {target}")
                return True
            except (OSError, NotImplementedError):
                # Fallback: copy directory on Windows or if symlink fails
                logger.warning(
                    f"Symlink failed, falling back to copy: {link} -> {target}"
                )
                if target.exists():
                    shutil.copytree(target, link, dirs_exist_ok=True)
                return True
        except Exception as e:
            logger.error(f"Failed to create symlink/copy: {e}")
            return False
    
    def register(
        self,
        training_result: TrainingResult,
        data_range: dict[str, str] | None = None,
        metadata_extra: dict[str, Any] | None = None,
    ) -> str:
        """
        Register a new model version.
        
        Args:
            training_result: TrainingResult from model training
            data_range: Optional dict with "start" and "end" dates
            metadata_extra: Optional additional metadata to store
            
        Returns:
            Version name (e.g., "20260126_103000")
            
        Raises:
            RuntimeError: If registration fails
        """
        with self.lock:
            try:
                # Generate version name
                version = self._generate_version_name()
                version_path = self._get_version_path(version)
                
                # Save training result
                training_result.save(version_path)
                
                # Read generated metadata
                metadata = self._read_metadata(version_path)
                
                # Enrich metadata
                metadata["version"] = version
                metadata["training_date"] = training_result.trained_at.isoformat()
                
                if data_range:
                    metadata["data_range"] = data_range
                
                if metadata_extra:
                    metadata.update(metadata_extra)
                
                # Add deployment status
                metadata["deployment_status"] = "registered"
                
                # Write enriched metadata
                self._write_metadata(version_path, metadata)
                
                logger.info(
                    f"Model registered: version={version}, "
                    f"path={version_path}, "
                    f"direction_accuracy={metadata['metrics'].get('direction_accuracy', 0):.3f}"
                )
                
                return version
            except Exception as e:
                logger.error(f"Failed to register model: {e}")
                raise RuntimeError(f"Model registration failed: {e}") from e
    
    def promote_to_production(self, version: str) -> bool:
        """
        Promote model version to production.
        
        Creates/updates symlink at models/production/latest -> models/{version}
        
        Args:
            version: Version name to promote
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                version_path = self._get_version_path(version)
                
                # Verify version exists
                if not version_path.exists():
                    logger.error(f"Version not found: {version}")
                    return False
                
                # Create symlink
                latest_link = self.production_dir / "latest"
                success = self._create_symlink(version_path, latest_link)
                
                if success:
                    # Update metadata
                    metadata = self._read_metadata(version_path)
                    metadata["deployment_status"] = "production"
                    metadata["promoted_at"] = datetime.now(timezone.utc).isoformat()
                    self._write_metadata(version_path, metadata)
                    
                    logger.info(f"Model promoted to production: {version}")
                
                return success
            except Exception as e:
                logger.error(f"Failed to promote model: {e}")
                return False
    
    def list_versions(
        self,
        min_direction_accuracy: float | None = None,
        min_profit_factor: float | None = None,
        limit: int | None = None,
    ) -> list[ModelVersion]:
        """
        List all model versions with optional filtering.
        
        Args:
            min_direction_accuracy: Minimum direction accuracy threshold
            min_profit_factor: Minimum profit factor threshold
            limit: Maximum number of versions to return
            
        Returns:
            List of ModelVersion objects sorted by date (newest first)
        """
        with self.lock:
            try:
                versions = []
                
                # Scan all version directories
                for version_dir in sorted(self.base_path.iterdir(), reverse=True):
                    if not version_dir.is_dir() or version_dir.name == "production":
                        continue
                    
                    try:
                        metadata = self._read_metadata(version_dir)
                        metrics = metadata.get("metrics", {})
                        
                        # Apply filters
                        if min_direction_accuracy is not None:
                            if metrics.get("direction_accuracy", 0) < min_direction_accuracy:
                                continue
                        
                        if min_profit_factor is not None:
                            if metrics.get("profit_factor", 0) < min_profit_factor:
                                continue
                        
                        version_obj = ModelVersion(
                            version=metadata.get("version", version_dir.name),
                            training_date=metadata.get("training_date", ""),
                            metrics=metrics,
                            features=metadata.get("feature_names", []),
                            path=version_dir,
                        )
                        versions.append(version_obj)
                    except Exception as e:
                        logger.warning(f"Failed to read version {version_dir.name}: {e}")
                        continue
                
                # Apply limit
                if limit:
                    versions = versions[:limit]
                
                logger.info(f"Listed {len(versions)} versions")
                return versions
            except Exception as e:
                logger.error(f"Failed to list versions: {e}")
                return []
    
    def load_version(self, version: str = "latest") -> TrainingResult | None:
        """
        Load a specific model version or latest production model.
        
        Args:
            version: Version name or "latest" for production model
            
        Returns:
            TrainingResult or None if not found
        """
        with self.lock:
            try:
                if version == "latest":
                    # Load from production symlink
                    latest_link = self.production_dir / "latest"
                    
                    if not latest_link.exists():
                        logger.warning("No production model available")
                        return None
                    
                    # Resolve symlink
                    if latest_link.is_symlink():
                        version_path = latest_link.resolve()
                    else:
                        version_path = latest_link
                else:
                    version_path = self._get_version_path(version)
                
                if not version_path.exists():
                    logger.error(f"Version not found: {version}")
                    return None
                
                # Load training result
                result = TrainingResult.load(version_path)
                logger.info(f"Loaded model version: {version}")
                return result
            except Exception as e:
                logger.error(f"Failed to load version {version}: {e}")
                return None
    
    def compare_versions(self, version1: str, version2: str) -> dict[str, Any]:
        """
        Compare two model versions by metrics.
        
        Args:
            version1: First version name
            version2: Second version name
            
        Returns:
            Dict comparing metrics with winners indicated
            
        Raises:
            ValueError: If versions not found
        """
        with self.lock:
            try:
                path1 = self._get_version_path(version1)
                path2 = self._get_version_path(version2)
                
                if not path1.exists():
                    raise ValueError(f"Version not found: {version1}")
                if not path2.exists():
                    raise ValueError(f"Version not found: {version2}")
                
                metadata1 = self._read_metadata(path1)
                metadata2 = self._read_metadata(path2)
                
                metrics1 = metadata1.get("metrics", {})
                metrics2 = metadata2.get("metrics", {})
                
                # Metrics where higher is better
                higher_is_better = {
                    "direction_accuracy", "profit_factor", "sharpe_ratio",
                    "r_squared", "annual_return", "total_return", "win_rate"
                }
                
                # Metrics where lower is better
                lower_is_better = {
                    "rmse", "mae", "max_drawdown", "volatility"
                }
                
                comparison = {}
                
                # Compare all metrics
                all_metrics = set(metrics1.keys()) | set(metrics2.keys())
                for metric in all_metrics:
                    val1 = metrics1.get(metric, 0)
                    val2 = metrics2.get(metric, 0)
                    
                    # Determine winner
                    if metric in higher_is_better:
                        winner = version1 if val1 > val2 else version2
                    elif metric in lower_is_better:
                        winner = version1 if val1 < val2 else version2
                    else:
                        winner = None
                    
                    comparison[metric] = {
                        version1: val1,
                        version2: val2,
                        "winner": winner,
                        "difference": abs(val1 - val2),
                    }
                
                logger.info(f"Compared versions: {version1} vs {version2}")
                return comparison
            except Exception as e:
                logger.error(f"Failed to compare versions: {e}")
                raise
    
    def cleanup_old_versions(self) -> list[str]:
        """
        Delete old versions, keeping only N latest.
        
        Never deletes the production version.
        
        Returns:
            List of deleted version names
        """
        with self.lock:
            try:
                deleted = []
                
                # Get all versions sorted by date (newest first)
                versions = self.list_versions(limit=None)
                
                # Get production version
                latest_link = self.production_dir / "latest"
                production_version = None
                if latest_link.exists():
                    if latest_link.is_symlink():
                        production_path = latest_link.resolve()
                    else:
                        production_path = latest_link
                    production_version = production_path.name
                
                # Delete old versions
                for i, version_obj in enumerate(versions):
                    if i < self.keep_versions:
                        continue  # Keep this version
                    
                    if version_obj.version == production_version:
                        logger.info(f"Skipping production version: {version_obj.version}")
                        continue
                    
                    try:
                        version_path = version_obj.path
                        shutil.rmtree(version_path)
                        deleted.append(version_obj.version)
                        logger.info(f"Deleted old version: {version_obj.version}")
                    except Exception as e:
                        logger.error(f"Failed to delete version {version_obj.version}: {e}")
                
                logger.info(f"Cleanup complete: deleted {len(deleted)} versions")
                return deleted
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
                return []
    
    def rollback_to_version(self, version: str) -> bool:
        """
        Rollback production to a previous version.
        
        Args:
            version: Version name to rollback to
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                version_path = self._get_version_path(version)
                
                if not version_path.exists():
                    logger.error(f"Version not found: {version}")
                    return False
                
                # Promote to production
                success = self.promote_to_production(version)
                
                if success:
                    logger.warning(f"Rolled back to version: {version}")
                
                return success
            except Exception as e:
                logger.error(f"Rollback failed: {e}")
                return False
    
    def get_version_info(self, version: str) -> dict[str, Any] | None:
        """
        Get detailed information about a version.
        
        Args:
            version: Version name
            
        Returns:
            Version metadata dict or None if not found
        """
        with self.lock:
            try:
                version_path = self._get_version_path(version)
                
                if not version_path.exists():
                    logger.error(f"Version not found: {version}")
                    return None
                
                metadata = self._read_metadata(version_path)
                return metadata
            except Exception as e:
                logger.error(f"Failed to get version info: {e}")
                return None
