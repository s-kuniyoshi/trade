"""
Comprehensive test suite for Phase 2 ML inference modules.

Tests for predictor, cache, and prediction service.
"""

import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import PredictionService, TTLCache, CacheEntry
from src.training.trainer import TrainingResult
from src.training.evaluation import PerformanceMetrics


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_features():
    """Create single row of features for prediction."""
    return pd.DataFrame({
        'feature1': [0.5],
        'feature2': [-0.3],
        'feature3': [0.1],
        'feature4': [0.2],
    })


@pytest.fixture
def sample_features_batch():
    """Create multiple rows of features for batch prediction."""
    return pd.DataFrame({
        'feature1': [0.5, -0.2, 0.1],
        'feature2': [-0.3, 0.4, -0.1],
        'feature3': [0.1, 0.2, -0.3],
        'feature4': [0.2, -0.1, 0.3],
    })


@pytest.fixture
def mock_model():
    """Create mock trained model."""
    model = Mock()
    model.predict = Mock(side_effect=lambda X: np.random.randn(len(X)) * 0.01)
    return model


@pytest.fixture
def mock_training_result(mock_model):
    """Create mock training result."""
    result = Mock(spec=TrainingResult)
    result.model = mock_model
    result.metrics = Mock(spec=PerformanceMetrics)
    result.metrics.rmse = 0.005
    result.metrics.mae = 0.004
    result.metrics.r_squared = 0.85
    result.metrics.direction_accuracy = 0.55
    result.feature_importance = pd.DataFrame({
        'feature': ['feature1', 'feature2', 'feature3', 'feature4'],
        'importance': [0.4, 0.3, 0.2, 0.1],
    })
    result.trained_at = datetime.now(timezone.utc)
    return result


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock()
    config.model.inference.get = Mock(side_effect=lambda key, default=None: {
        'cache': {'enabled': True, 'ttl_seconds': 300},
        'warmup': {'enabled': True, 'n_samples': 100},
        'batch_size': 32,
    }.get(key, default))
    config.model.target.direction_threshold = 0.001
    config.model.model_dir = "models/production/latest"
    return config


# =============================================================================
# TTLCache Tests
# =============================================================================

class TestTTLCache:
    """Tests for TTLCache class."""
    
    def test_init(self):
        """Test cache initialization."""
        cache = TTLCache(max_size=100, ttl_seconds=300)
        assert cache.max_size == 100
        assert cache.ttl_seconds == 300
        assert cache.size() == 0
    
    def test_set_get(self, sample_features):
        """Test basic cache set/get operations."""
        cache = TTLCache(max_size=100, ttl_seconds=300)
        
        value = {'direction': 'buy', 'confidence': 0.8}
        cache.set(sample_features, value)
        
        retrieved = cache.get(sample_features)
        assert retrieved is not None
        assert retrieved['direction'] == 'buy'
        assert retrieved['confidence'] == 0.8
    
    def test_get_nonexistent(self, sample_features):
        """Test getting non-existent key."""
        cache = TTLCache(max_size=100, ttl_seconds=300)
        
        retrieved = cache.get(sample_features)
        assert retrieved is None
    
    def test_cache_expiry(self, sample_features):
        """Test cache entry expiration."""
        cache = TTLCache(max_size=100, ttl_seconds=1)
        
        value = {'direction': 'buy', 'confidence': 0.8}
        cache.set(sample_features, value)
        
        # Should be available immediately
        assert cache.get(sample_features) is not None
        
        # Wait for expiry
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get(sample_features) is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = TTLCache(max_size=3, ttl_seconds=300)
        
        # Add 3 entries
        for i in range(3):
            features = pd.DataFrame({'feature1': [float(i)]})
            cache.set(features, {'value': i})
        
        assert cache.size() == 3
        
        # Add 4th entry - should evict oldest
        features_4 = pd.DataFrame({'feature1': [3.0]})
        cache.set(features_4, {'value': 3})
        
        assert cache.size() == 3
    
    def test_clear(self, sample_features):
        """Test clearing cache."""
        cache = TTLCache(max_size=100, ttl_seconds=300)
        
        cache.set(sample_features, {'value': 1})
        assert cache.size() == 1
        
        cache.clear()
        assert cache.size() == 0
    
    def test_hash_consistency(self, sample_features):
        """Test that same features produce same hash."""
        cache = TTLCache(max_size=100, ttl_seconds=300)
        
        value1 = {'value': 1}
        cache.set(sample_features, value1)
        
        # Same features should retrieve same value
        retrieved = cache.get(sample_features)
        assert retrieved['value'] == 1
    
    def test_different_features_different_hash(self):
        """Test that different features produce different hashes."""
        cache = TTLCache(max_size=100, ttl_seconds=300)
        
        features1 = pd.DataFrame({'feature1': [0.5]})
        features2 = pd.DataFrame({'feature1': [0.6]})
        
        cache.set(features1, {'value': 1})
        cache.set(features2, {'value': 2})
        
        assert cache.get(features1)['value'] == 1
        assert cache.get(features2)['value'] == 2


# =============================================================================
# PredictionService Tests
# =============================================================================

class TestPredictionService:
    """Tests for PredictionService class."""
    
    def test_init_with_mock_model(self, mock_training_result, tmp_path):
        """Test service initialization with mock model."""
        # Save mock model
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': True, 'ttl_seconds': 300},
                'warmup': {'enabled': False, 'n_samples': 100},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                assert service.model is not None
                assert service.feature_names == ['feature1', 'feature2', 'feature3', 'feature4']
                assert service.cache_enabled is True
    
    def test_validate_features_valid(self, sample_features, mock_training_result, tmp_path):
        """Test feature validation with valid features."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': False},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                # Should not raise
                service._validate_features(sample_features)
    
    def test_validate_features_empty(self, mock_training_result, tmp_path):
        """Test feature validation with empty DataFrame."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': False},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                empty_features = pd.DataFrame()
                with pytest.raises(ValueError):
                    service._validate_features(empty_features)
    
    def test_validate_features_multiple_rows(self, sample_features_batch, mock_training_result, tmp_path):
        """Test feature validation with multiple rows."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': False},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                with pytest.raises(ValueError):
                    service._validate_features(sample_features_batch)
    
    def test_validate_features_missing_columns(self, mock_training_result, tmp_path):
        """Test feature validation with missing columns."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': False},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                incomplete_features = pd.DataFrame({'feature1': [0.5]})
                with pytest.raises(ValueError):
                    service._validate_features(incomplete_features)
    
    def test_validate_features_nan_values(self, mock_training_result, tmp_path):
        """Test feature validation with NaN values."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': False},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                nan_features = pd.DataFrame({
                    'feature1': [np.nan],
                    'feature2': [0.5],
                    'feature3': [0.1],
                    'feature4': [0.2],
                })
                with pytest.raises(ValueError):
                    service._validate_features(nan_features)
    
    def test_validate_features_inf_values(self, mock_training_result, tmp_path):
        """Test feature validation with infinite values."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': False},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                inf_features = pd.DataFrame({
                    'feature1': [np.inf],
                    'feature2': [0.5],
                    'feature3': [0.1],
                    'feature4': [0.2],
                })
                with pytest.raises(ValueError):
                    service._validate_features(inf_features)
    
    def test_estimate_confidence(self, mock_training_result, tmp_path):
        """Test confidence estimation."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': False},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                # Test various predictions
                conf_small = service._estimate_confidence(0.001)
                conf_medium = service._estimate_confidence(0.01)
                conf_large = service._estimate_confidence(0.1)
                
                # Smaller predictions should have lower confidence
                assert conf_small < conf_medium
                assert conf_small == 0.1  # 0.001 / 0.01 = 0.1
                
                # Confidence should be capped at 1.0
                assert conf_medium == 1.0  # 0.01 / 0.01 = 1.0 (at cap)
                assert conf_large == 1.0   # 0.1 / 0.01 = 10.0 → capped to 1.0
    
    def test_map_direction_buy(self, mock_training_result, tmp_path):
        """Test direction mapping - buy signal."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': False},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                direction = service._map_direction(0.01)
                assert direction == "buy"
    
    def test_map_direction_sell(self, mock_training_result, tmp_path):
        """Test direction mapping - sell signal."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': False},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                direction = service._map_direction(-0.01)
                assert direction == "sell"
    
    def test_map_direction_neutral(self, mock_training_result, tmp_path):
        """Test direction mapping - neutral signal."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': False},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                direction = service._map_direction(0.0001)
                assert direction == "neutral"
    
    def test_predict_single(self, sample_features, mock_training_result, tmp_path):
        """Test single prediction."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': False},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                result = service.predict(sample_features)
                
                assert 'direction' in result
                assert 'expected_return' in result
                assert 'confidence' in result
                assert 'timestamp' in result
                assert result['direction'] in ['buy', 'sell', 'neutral']
                assert 0 <= result['confidence'] <= 1
    
    def test_predict_with_quantiles(self, sample_features, mock_training_result, tmp_path):
        """Test prediction with quantiles."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': False},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                result = service.predict(sample_features, include_quantiles=True)
                
                assert 'quantile_5' in result
                assert 'quantile_50' in result
                assert 'quantile_95' in result
    
    def test_predict_batch(self, sample_features_batch, mock_training_result, tmp_path):
        """Test batch prediction."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': False},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                results = service.predict_batch(sample_features_batch)
                
                assert len(results) == 3
                for result in results:
                    assert 'direction' in result
                    assert 'expected_return' in result
                    assert 'confidence' in result
    
    def test_cache_behavior(self, sample_features, mock_training_result, tmp_path):
        """Test caching behavior."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': True, 'ttl_seconds': 300},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                # First prediction
                result1 = service.predict(sample_features)
                
                # Second prediction (should be cached)
                result2 = service.predict(sample_features)
                
                # Results should be identical (same cache entry)
                assert result1['expected_return'] == result2['expected_return']
    
    def test_clear_cache(self, sample_features, mock_training_result, tmp_path):
        """Test cache clearing."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': True, 'ttl_seconds': 300},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                # Add to cache
                service.predict(sample_features)
                assert service.cache.size() > 0
                
                # Clear
                service.clear_cache()
                assert service.cache.size() == 0
    
    def test_get_cache_stats(self, mock_training_result, tmp_path):
        """Test cache statistics."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': True, 'ttl_seconds': 300},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                stats = service.get_cache_stats()
                
                assert 'enabled' in stats
                assert stats['enabled'] is True
                assert 'size' in stats
                assert 'max_size' in stats
                assert 'ttl_seconds' in stats
    
    def test_get_model_info(self, mock_training_result, tmp_path):
        """Test getting model information."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        
        with patch('src.inference.predictor.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.inference.get = Mock(side_effect=lambda key, default=None: {
                'cache': {'enabled': False},
                'warmup': {'enabled': False},
                'batch_size': 32,
            }.get(key, default))
            mock_config.model.target.direction_threshold = 0.001
            mock_config.model.model_dir = str(model_path)
            mock_get_config.return_value = mock_config
            
            with patch('src.inference.predictor.TrainingResult.load') as mock_load:
                mock_load.return_value = mock_training_result
                
                service = PredictionService(model_path=model_path, warmup_enabled=False)
                
                info = service.get_model_info()
                
                assert 'n_features' in info
                assert 'feature_names' in info
                assert 'trained_at' in info
                assert 'metrics' in info
                assert info['n_features'] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
