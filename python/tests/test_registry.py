"""
Comprehensive test suite for Phase 2 ML registry module.

Tests for model registry, versioning, and deployment management.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timezone
from threading import Thread
import time

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.registry import ModelRegistry, ModelVersion
from src.training.trainer import TrainingResult
from src.training.evaluation import PerformanceMetrics


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_registry_dir(tmp_path):
    """Create temporary directory for registry testing."""
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    return registry_dir


@pytest.fixture
def sample_training_result():
    """Create sample training result for registration."""
    result = Mock(spec=TrainingResult)
    result.model = Mock()
    result.metrics = Mock(spec=PerformanceMetrics)
    result.metrics.rmse = 0.005
    result.metrics.mae = 0.004
    result.metrics.r_squared = 0.85
    result.metrics.direction_accuracy = 0.55
    result.metrics.to_dict = Mock(return_value={
        'rmse': 0.005,
        'mae': 0.004,
        'r_squared': 0.85,
        'direction_accuracy': 0.55,
    })
    result.feature_importance = pd.DataFrame({
        'feature': ['feature1', 'feature2', 'feature3', 'feature4'],
        'importance': [0.4, 0.3, 0.2, 0.1],
    })
    result.trained_at = datetime.now(timezone.utc)
    result.save = Mock()
    return result


@pytest.fixture
def mock_registry_config():
    """Create mock configuration for registry."""
    config = Mock()
    config.model.registry.get = Mock(side_effect=lambda k, default=None: {
        'path': 'models/',
        'keep_versions': 5,
    }.get(k, default))
    return config


# =============================================================================
# ModelRegistry Tests
# =============================================================================

class TestModelRegistry:
    """Tests for ModelRegistry class."""
    
    def test_init(self, temp_registry_dir):
        """Test registry initialization."""
        with patch('src.training.registry.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.registry.get = Mock(side_effect=lambda k, default=None: {
                'path': str(temp_registry_dir),
                'keep_versions': 5,
            }.get(k, default))
            mock_get_config.return_value = mock_config
            
            registry = ModelRegistry()
            
            assert registry.base_path == temp_registry_dir
            assert registry.keep_versions == 5
            assert (temp_registry_dir / "production").exists()
    
    def test_register_model(self, temp_registry_dir, sample_training_result):
        """Test model registration."""
        with patch('src.training.registry.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.registry.get = Mock(side_effect=lambda k, default=None: {
                'path': str(temp_registry_dir),
                'keep_versions': 5,
            }.get(k, default))
            mock_get_config.return_value = mock_config
            
            registry = ModelRegistry()
            
            # Mock the save method to create metadata
            def mock_save(path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                metadata = {
                    'metrics': {
                        'rmse': 0.005,
                        'mae': 0.004,
                        'r_squared': 0.85,
                        'direction_accuracy': 0.55,
                    },
                    'config': {
                        'model_type': 'lightgbm',
                        'objective': 'regression',
                        'n_estimators': 100,
                    },
                    'trained_at': datetime.now(timezone.utc).isoformat(),
                    'feature_names': ['feature1', 'feature2', 'feature3', 'feature4'],
                }
                with open(path / "metadata.json", "w") as f:
                    json.dump(metadata, f)
            
            sample_training_result.save = mock_save
            
            version = registry.register(sample_training_result)
            
            assert version is not None
            assert len(version) == 15  # YYYYMMdd_HHMMSS format
            assert (temp_registry_dir / version).exists()
    
    def test_register_with_metadata(self, temp_registry_dir, sample_training_result):
        """Test model registration with extra metadata."""
        with patch('src.training.registry.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.registry.get = Mock(side_effect=lambda k, default=None: {
                'path': str(temp_registry_dir),
                'keep_versions': 5,
            }.get(k, default))
            mock_get_config.return_value = mock_config
            
            registry = ModelRegistry()
            
            def mock_save(path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                metadata = {
                    'metrics': {
                        'rmse': 0.005,
                        'mae': 0.004,
                        'r_squared': 0.85,
                        'direction_accuracy': 0.55,
                    },
                    'config': {
                        'model_type': 'lightgbm',
                        'objective': 'regression',
                        'n_estimators': 100,
                    },
                    'trained_at': datetime.now(timezone.utc).isoformat(),
                    'feature_names': ['feature1', 'feature2', 'feature3', 'feature4'],
                }
                with open(path / "metadata.json", "w") as f:
                    json.dump(metadata, f)
            
            sample_training_result.save = mock_save
            
            version = registry.register(
                sample_training_result,
                data_range={'start': '2024-01-01', 'end': '2024-12-31'},
                metadata_extra={'custom_field': 'custom_value'}
            )
            
            assert version is not None
            
            # Check metadata was enriched
            metadata_path = temp_registry_dir / version / "metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            assert 'data_range' in metadata
            assert 'custom_field' in metadata
    
    def test_promote_to_production(self, temp_registry_dir, sample_training_result):
        """Test promoting model to production."""
        with patch('src.training.registry.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.registry.get = Mock(side_effect=lambda k, default=None: {
                'path': str(temp_registry_dir),
                'keep_versions': 5,
            }.get(k, default))
            mock_get_config.return_value = mock_config
            
            registry = ModelRegistry()
            
            def mock_save(path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                metadata = {
                    'metrics': {
                        'rmse': 0.005,
                        'mae': 0.004,
                        'r_squared': 0.85,
                        'direction_accuracy': 0.55,
                    },
                    'config': {
                        'model_type': 'lightgbm',
                        'objective': 'regression',
                        'n_estimators': 100,
                    },
                    'trained_at': datetime.now(timezone.utc).isoformat(),
                    'feature_names': ['feature1', 'feature2', 'feature3', 'feature4'],
                }
                with open(path / "metadata.json", "w") as f:
                    json.dump(metadata, f)
            
            sample_training_result.save = mock_save
            
            version = registry.register(sample_training_result)
            success = registry.promote_to_production(version)
            
            assert success is True
            
            # Check symlink/copy exists
            latest_link = temp_registry_dir / "production" / "latest"
            assert latest_link.exists()
    
    def test_list_versions(self, temp_registry_dir, sample_training_result):
        """Test listing model versions."""
        with patch('src.training.registry.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.registry.get = Mock(side_effect=lambda k, default=None: {
                'path': str(temp_registry_dir),
                'keep_versions': 5,
            }.get(k, default))
            mock_get_config.return_value = mock_config
            
            registry = ModelRegistry()
            
            def mock_save(path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                metadata = {
                    'metrics': {
                        'rmse': 0.005,
                        'mae': 0.004,
                        'r_squared': 0.85,
                        'direction_accuracy': 0.55,
                    },
                    'config': {
                        'model_type': 'lightgbm',
                        'objective': 'regression',
                        'n_estimators': 100,
                    },
                    'trained_at': datetime.now(timezone.utc).isoformat(),
                    'feature_names': ['feature1', 'feature2', 'feature3', 'feature4'],
                }
                with open(path / "metadata.json", "w") as f:
                    json.dump(metadata, f)
            
            sample_training_result.save = mock_save
            
            # Register multiple versions
            v1 = registry.register(sample_training_result)
            time.sleep(1.1)  # Ensure different second for version name
            v2 = registry.register(sample_training_result)
            
            versions = registry.list_versions()
            
            assert len(versions) >= 2
            assert all(isinstance(v, ModelVersion) for v in versions)
    
    def test_list_versions_with_filter(self, temp_registry_dir, sample_training_result):
        """Test listing versions with filters."""
        with patch('src.training.registry.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.registry.get = Mock(side_effect=lambda k, default=None: {
                'path': str(temp_registry_dir),
                'keep_versions': 5,
            }.get(k, default))
            mock_get_config.return_value = mock_config
            
            registry = ModelRegistry()
            
            def mock_save(path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                metadata = {
                    'metrics': {
                        'rmse': 0.005,
                        'mae': 0.004,
                        'r_squared': 0.85,
                        'direction_accuracy': 0.55,
                    },
                    'config': {
                        'model_type': 'lightgbm',
                        'objective': 'regression',
                        'n_estimators': 100,
                    },
                    'trained_at': datetime.now(timezone.utc).isoformat(),
                    'feature_names': ['feature1', 'feature2', 'feature3', 'feature4'],
                }
                with open(path / "metadata.json", "w") as f:
                    json.dump(metadata, f)
            
            sample_training_result.save = mock_save
            
            registry.register(sample_training_result)
            
            # Filter by direction accuracy
            versions = registry.list_versions(min_direction_accuracy=0.50)
            assert len(versions) >= 1
            
            # Filter by high threshold
            versions = registry.list_versions(min_direction_accuracy=0.60)
            assert len(versions) == 0
    
    def test_load_version(self, temp_registry_dir, sample_training_result):
        """Test loading a specific version."""
        with patch('src.training.registry.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.registry.get = Mock(side_effect=lambda k, default=None: {
                'path': str(temp_registry_dir),
                'keep_versions': 5,
            }.get(k, default))
            mock_get_config.return_value = mock_config
            
            registry = ModelRegistry()
            
            def mock_save(path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                metadata = {
                    'metrics': {
                        'rmse': 0.005,
                        'mae': 0.004,
                        'r_squared': 0.85,
                        'direction_accuracy': 0.55,
                    },
                    'config': {
                        'model_type': 'lightgbm',
                        'objective': 'regression',
                        'n_estimators': 100,
                    },
                    'trained_at': datetime.now(timezone.utc).isoformat(),
                    'feature_names': ['feature1', 'feature2', 'feature3', 'feature4'],
                }
                with open(path / "metadata.json", "w") as f:
                    json.dump(metadata, f)
                
                # Create dummy model file
                with open(path / "model.pkl", "wb") as f:
                    f.write(b"dummy_model")
                
                # Create feature importance
                pd.DataFrame({
                    'feature': ['feature1', 'feature2', 'feature3', 'feature4'],
                    'importance': [0.4, 0.3, 0.2, 0.1],
                }).to_csv(path / "feature_importance.csv", index=False)
            
            sample_training_result.save = mock_save
            
            version = registry.register(sample_training_result)
            
            with patch('src.training.registry.TrainingResult.load') as mock_load:
                mock_load.return_value = sample_training_result
                
                loaded = registry.load_version(version)
                assert loaded is not None
    
    def test_load_latest_production(self, temp_registry_dir, sample_training_result):
        """Test loading latest production model."""
        with patch('src.training.registry.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.registry.get = Mock(side_effect=lambda k, default=None: {
                'path': str(temp_registry_dir),
                'keep_versions': 5,
            }.get(k, default))
            mock_get_config.return_value = mock_config
            
            registry = ModelRegistry()
            
            def mock_save(path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                metadata = {
                    'metrics': {
                        'rmse': 0.005,
                        'mae': 0.004,
                        'r_squared': 0.85,
                        'direction_accuracy': 0.55,
                    },
                    'config': {
                        'model_type': 'lightgbm',
                        'objective': 'regression',
                        'n_estimators': 100,
                    },
                    'trained_at': datetime.now(timezone.utc).isoformat(),
                    'feature_names': ['feature1', 'feature2', 'feature3', 'feature4'],
                }
                with open(path / "metadata.json", "w") as f:
                    json.dump(metadata, f)
                
                with open(path / "model.pkl", "wb") as f:
                    f.write(b"dummy_model")
                
                pd.DataFrame({
                    'feature': ['feature1', 'feature2', 'feature3', 'feature4'],
                    'importance': [0.4, 0.3, 0.2, 0.1],
                }).to_csv(path / "feature_importance.csv", index=False)
            
            sample_training_result.save = mock_save
            
            version = registry.register(sample_training_result)
            registry.promote_to_production(version)
            
            with patch('src.training.registry.TrainingResult.load') as mock_load:
                mock_load.return_value = sample_training_result
                
                loaded = registry.load_version("latest")
                assert loaded is not None
    
    def test_compare_versions(self, temp_registry_dir, sample_training_result):
        """Test comparing two versions."""
        with patch('src.training.registry.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.registry.get = Mock(side_effect=lambda k, default=None: {
                'path': str(temp_registry_dir),
                'keep_versions': 5,
            }.get(k, default))
            mock_get_config.return_value = mock_config
            
            registry = ModelRegistry()
            
            def mock_save(path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                metadata = {
                    'metrics': {
                        'rmse': 0.005,
                        'mae': 0.004,
                        'r_squared': 0.85,
                        'direction_accuracy': 0.55,
                    },
                    'config': {
                        'model_type': 'lightgbm',
                        'objective': 'regression',
                        'n_estimators': 100,
                    },
                    'trained_at': datetime.now(timezone.utc).isoformat(),
                    'feature_names': ['feature1', 'feature2', 'feature3', 'feature4'],
                }
                with open(path / "metadata.json", "w") as f:
                    json.dump(metadata, f)
            
            sample_training_result.save = mock_save
            
            v1 = registry.register(sample_training_result)
            time.sleep(0.1)
            v2 = registry.register(sample_training_result)
            
            comparison = registry.compare_versions(v1, v2)
            
            assert isinstance(comparison, dict)
            assert 'rmse' in comparison
            assert 'direction_accuracy' in comparison
    
    def test_cleanup_old_versions(self, temp_registry_dir, sample_training_result):
        """Test cleanup of old versions."""
        with patch('src.training.registry.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.registry.get = Mock(side_effect=lambda k, default=None: {
                'path': str(temp_registry_dir),
                'keep_versions': 2,
            }.get(k, default))
            mock_get_config.return_value = mock_config
            
            registry = ModelRegistry()
            
            def mock_save(path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                metadata = {
                    'metrics': {
                        'rmse': 0.005,
                        'mae': 0.004,
                        'r_squared': 0.85,
                        'direction_accuracy': 0.55,
                    },
                    'config': {
                        'model_type': 'lightgbm',
                        'objective': 'regression',
                        'n_estimators': 100,
                    },
                    'trained_at': datetime.now(timezone.utc).isoformat(),
                    'feature_names': ['feature1', 'feature2', 'feature3', 'feature4'],
                }
                with open(path / "metadata.json", "w") as f:
                    json.dump(metadata, f)
            
            sample_training_result.save = mock_save
            
            # Register 4 versions
            versions = []
            for i in range(4):
                v = registry.register(sample_training_result)
                versions.append(v)
                time.sleep(0.05)
            
            # Cleanup
            deleted = registry.cleanup_old_versions()
            
            # Should have deleted 2 versions (keep 2)
            assert len(deleted) >= 1
    
    def test_rollback_to_version(self, temp_registry_dir, sample_training_result):
        """Test rolling back to a previous version."""
        with patch('src.training.registry.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.registry.get = Mock(side_effect=lambda k, default=None: {
                'path': str(temp_registry_dir),
                'keep_versions': 5,
            }.get(k, default))
            mock_get_config.return_value = mock_config
            
            registry = ModelRegistry()
            
            def mock_save(path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                metadata = {
                    'metrics': {
                        'rmse': 0.005,
                        'mae': 0.004,
                        'r_squared': 0.85,
                        'direction_accuracy': 0.55,
                    },
                    'config': {
                        'model_type': 'lightgbm',
                        'objective': 'regression',
                        'n_estimators': 100,
                    },
                    'trained_at': datetime.now(timezone.utc).isoformat(),
                    'feature_names': ['feature1', 'feature2', 'feature3', 'feature4'],
                }
                with open(path / "metadata.json", "w") as f:
                    json.dump(metadata, f)
            
            sample_training_result.save = mock_save
            
            v1 = registry.register(sample_training_result)
            time.sleep(0.1)
            v2 = registry.register(sample_training_result)
            
            # Promote v2
            registry.promote_to_production(v2)
            
            # Rollback to v1
            success = registry.rollback_to_version(v1)
            assert success is True
    
    def test_get_version_info(self, temp_registry_dir, sample_training_result):
        """Test getting version metadata."""
        with patch('src.training.registry.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.registry.get = Mock(side_effect=lambda k, default=None: {
                'path': str(temp_registry_dir),
                'keep_versions': 5,
            }.get(k, default))
            mock_get_config.return_value = mock_config
            
            registry = ModelRegistry()
            
            def mock_save(path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                metadata = {
                    'metrics': {
                        'rmse': 0.005,
                        'mae': 0.004,
                        'r_squared': 0.85,
                        'direction_accuracy': 0.55,
                    },
                    'config': {
                        'model_type': 'lightgbm',
                        'objective': 'regression',
                        'n_estimators': 100,
                    },
                    'trained_at': datetime.now(timezone.utc).isoformat(),
                    'feature_names': ['feature1', 'feature2', 'feature3', 'feature4'],
                }
                with open(path / "metadata.json", "w") as f:
                    json.dump(metadata, f)
            
            sample_training_result.save = mock_save
            
            version = registry.register(sample_training_result)
            info = registry.get_version_info(version)
            
            assert info is not None
            assert 'metrics' in info
            assert 'config' in info
            assert info['metrics']['direction_accuracy'] == 0.55
    
    def test_concurrent_access(self, temp_registry_dir, sample_training_result):
        """Test thread safety of registry operations."""
        with patch('src.training.registry.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.registry.get = Mock(side_effect=lambda k, default=None: {
                'path': str(temp_registry_dir),
                'keep_versions': 5,
            }.get(k, default))
            mock_get_config.return_value = mock_config
            
            registry = ModelRegistry()
            
            def mock_save(path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                metadata = {
                    'metrics': {
                        'rmse': 0.005,
                        'mae': 0.004,
                        'r_squared': 0.85,
                        'direction_accuracy': 0.55,
                    },
                    'config': {
                        'model_type': 'lightgbm',
                        'objective': 'regression',
                        'n_estimators': 100,
                    },
                    'trained_at': datetime.now(timezone.utc).isoformat(),
                    'feature_names': ['feature1', 'feature2', 'feature3', 'feature4'],
                }
                with open(path / "metadata.json", "w") as f:
                    json.dump(metadata, f)
            
            sample_training_result.save = mock_save
            
            versions = []
            
            def register_version():
                v = registry.register(sample_training_result)
                versions.append(v)
            
            # Run multiple registrations concurrently
            threads = [Thread(target=register_version) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # All should succeed
            assert len(versions) == 3
            assert len(set(versions)) == 3  # All unique


# =============================================================================
# ModelVersion Tests
# =============================================================================

class TestModelVersion:
    """Tests for ModelVersion dataclass."""
    
    def test_model_version_creation(self, temp_registry_dir):
        """Test creating ModelVersion instance."""
        version = ModelVersion(
            version="20240101_120000",
            training_date="2024-01-01T12:00:00",
            metrics={'rmse': 0.005, 'direction_accuracy': 0.55},
            features=['feature1', 'feature2'],
            path=temp_registry_dir
        )
        
        assert version.version == "20240101_120000"
        assert version.metrics['rmse'] == 0.005
        assert len(version.features) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
