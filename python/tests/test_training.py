"""
Comprehensive test suite for Phase 2 ML training modules.

Tests for trainer, hyperopt, calibration, and model_comparison modules.
"""

import json
import pickle
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import ModelTrainer, TrainingConfig, TrainingResult
from src.training.hyperopt import OptunaOptimizer, OptimizationResult, LightGBMSearchSpace
from src.training.calibration import ConfidenceCalibrator
from src.training.model_comparison import ModelComparator, MetricComparison, ComparisonResult
from src.training.evaluation import PerformanceMetrics


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_training_data():
    """Create sample training data with features and target."""
    np.random.seed(42)
    n = 500
    
    features = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n),
        'feature4': np.random.randn(n),
    })
    
    # Create target with some correlation to features
    target = (
        0.5 * features['feature1'] +
        0.3 * features['feature2'] +
        0.2 * features['feature3'] +
        np.random.randn(n) * 0.01
    )
    
    df = pd.concat([features, target.rename('target')], axis=1)
    return df


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
def mock_config():
    """Create mock configuration."""
    config = Mock()
    config.model.lightgbm.learning_rate = 0.1
    config.model.lightgbm.n_estimators = 100
    config.model.lightgbm.early_stopping_rounds = 10
    config.model.lightgbm.verbose = -1
    config.model.lightgbm.objective = "regression"
    config.model.lightgbm.metric = "rmse"
    config.model.lightgbm.to_params = Mock(return_value={
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'regression',
        'metric': 'rmse',
        'verbose': -1,
    })
    config.model.training.train_ratio = 0.7
    config.model.training.validation_ratio = 0.15
    config.model.training.test_ratio = 0.15
    config.model.training.walk_forward.get = Mock(side_effect=lambda k, default=None: {
        'enabled': True,
        'train_window_bars': 350,
        'test_window_bars': 70,
        'step_bars': 70,
    }.get(k, default))
    config.model.training.cv.get = Mock(return_value=24)
    config.model.active_model = "lightgbm"
    config.model.calibration.adjustment = {'high_vol_penalty': 0.1, 'counter_trend_penalty': 0.05}
    config.model.calibration.min_confidence = 0.3
    config.model.registry.get = Mock(side_effect=lambda k, default=None: {
        'path': 'models/',
        'keep_versions': 5,
    }.get(k, default))
    return config


@pytest.fixture
def trained_model(sample_training_data):
    """Create a pre-trained model for testing."""
    with patch('src.training.trainer.get_config') as mock_get_config:
        mock_config = Mock()
        mock_config.model.lightgbm.to_params = Mock(return_value={
            'learning_rate': 0.1,
            'n_estimators': 50,
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
        })
        mock_get_config.return_value = mock_config
        
        trainer = ModelTrainer(TrainingConfig(
            n_estimators=50,
            early_stopping_rounds=5,
        ))
        
        result = trainer.train(
            sample_training_data,
            ['feature1', 'feature2', 'feature3', 'feature4'],
            'target'
        )
        return result


@pytest.fixture
def temp_registry_dir(tmp_path):
    """Create temporary directory for registry testing."""
    return tmp_path / "registry"


# =============================================================================
# ModelTrainer Tests
# =============================================================================

class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
    def test_init_default_config(self):
        """Test trainer initialization with default config."""
        with patch('src.training.trainer.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.training.train_ratio = 0.7
            mock_config.model.training.validation_ratio = 0.15
            mock_config.model.training.test_ratio = 0.15
            mock_config.model.training.walk_forward.get = Mock(return_value=True)
            mock_config.model.training.cv.get = Mock(return_value=24)
            mock_config.model.active_model = "lightgbm"
            mock_config.model.lightgbm.objective = "regression"
            mock_config.model.lightgbm.metric = "rmse"
            mock_config.model.lightgbm.n_estimators = 1000
            mock_config.model.lightgbm.early_stopping_rounds = 50
            mock_config.model.lightgbm.verbose = -1
            mock_get_config.return_value = mock_config
            
            trainer = ModelTrainer()
            assert trainer.config is not None
            assert trainer.evaluator is not None
    
    def test_init_custom_config(self):
        """Test trainer initialization with custom config."""
        config = TrainingConfig(n_estimators=100)
        trainer = ModelTrainer(config)
        assert trainer.config.n_estimators == 100
    
    def test_prepare_data(self, sample_training_data):
        """Test data preparation and splitting."""
        trainer = ModelTrainer(TrainingConfig())
        X_train, X_val, y_train, y_val = trainer.prepare_data(
            sample_training_data,
            ['feature1', 'feature2', 'feature3', 'feature4'],
            'target'
        )
        
        # Check splits exist
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(y_train) > 0
        assert len(y_val) > 0
        
        # Check no overlap
        assert len(X_train) + len(X_val) < len(sample_training_data)
        
        # Check indices don't overlap
        assert len(set(X_train.index) & set(X_val.index)) == 0
    
    def test_prepare_data_with_missing_values(self, sample_training_data):
        """Test data preparation handles missing values."""
        df = sample_training_data.copy()
        df.loc[0, 'feature1'] = np.nan
        df.loc[1, 'target'] = np.nan
        
        trainer = ModelTrainer(TrainingConfig())
        X_train, X_val, y_train, y_val = trainer.prepare_data(
            df,
            ['feature1', 'feature2', 'feature3', 'feature4'],
            'target'
        )
        
        # Should have removed rows with NaN
        assert len(X_train) + len(X_val) < len(df)
        assert not X_train.isna().any().any()
        assert not y_train.isna().any()
    
    def test_train_lightgbm(self, sample_training_data):
        """Test LightGBM model training."""
        trainer = ModelTrainer(TrainingConfig(n_estimators=50, early_stopping_rounds=5))
        
        X_train, X_val, y_train, y_val = trainer.prepare_data(
            sample_training_data,
            ['feature1', 'feature2', 'feature3', 'feature4'],
            'target'
        )
        
        model, history = trainer.train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Check model is trained
        assert model is not None
        assert hasattr(model, 'predict')
        
        # Check history
        assert 'train' in history
        assert 'valid' in history
        assert len(history['train']) > 0
        assert len(history['valid']) > 0
    
    def test_train_full_pipeline(self, sample_training_data):
        """Test full training pipeline."""
        trainer = ModelTrainer(TrainingConfig(n_estimators=50, early_stopping_rounds=5))
        
        result = trainer.train(
            sample_training_data,
            ['feature1', 'feature2', 'feature3', 'feature4'],
            'target'
        )
        
        # Check result
        assert isinstance(result, TrainingResult)
        assert result.model is not None
        assert result.metrics is not None
        assert result.feature_importance is not None
        
        # Check metrics
        assert result.metrics.rmse > 0
        assert 0 <= result.metrics.direction_accuracy <= 1
        
        # Check feature importance
        assert len(result.feature_importance) == 4
        assert 'feature' in result.feature_importance.columns
        assert 'importance' in result.feature_importance.columns
    
    def test_should_deploy_pass(self, trained_model):
        """Test deployment criteria - passing case."""
        trainer = ModelTrainer(TrainingConfig(
            min_direction_accuracy=0.51,
            min_profit_factor=1.0
        ))
        
        # Mock metrics to pass criteria
        trained_model.metrics.direction_accuracy = 0.55
        trained_model.metrics.rmse = 0.005
        
        should_deploy, reason = trainer.should_deploy(trained_model)
        assert should_deploy is True
    
    def test_should_deploy_fail_low_accuracy(self, trained_model):
        """Test deployment criteria - low accuracy."""
        trainer = ModelTrainer(TrainingConfig(min_direction_accuracy=0.55))
        
        trained_model.metrics.direction_accuracy = 0.50
        
        should_deploy, reason = trainer.should_deploy(trained_model)
        assert should_deploy is False
        assert "Direction accuracy" in reason
    
    def test_should_deploy_fail_high_rmse(self, trained_model):
        """Test deployment criteria - high RMSE."""
        trainer = ModelTrainer(TrainingConfig(min_direction_accuracy=0.51))
        
        trained_model.metrics.direction_accuracy = 0.55
        trained_model.metrics.rmse = 0.15
        
        should_deploy, reason = trainer.should_deploy(trained_model)
        assert should_deploy is False
        assert "RMSE" in reason


# =============================================================================
# OptunaOptimizer Tests
# =============================================================================

class TestOptunaOptimizer:
    """Tests for OptunaOptimizer class."""
    
    def test_init_default(self):
        """Test optimizer initialization with defaults."""
        optimizer = OptunaOptimizer()
        assert optimizer.n_trials == 50
        assert optimizer.sampler_type == "tpe"
        assert optimizer.pruner_type == "median"
        assert optimizer.direction == "minimize"
    
    def test_init_custom(self):
        """Test optimizer initialization with custom params."""
        optimizer = OptunaOptimizer(
            n_trials=10,
            sampler="random",
            pruner="percentile",
            direction="maximize"
        )
        assert optimizer.n_trials == 10
        assert optimizer.sampler_type == "random"
        assert optimizer.pruner_type == "percentile"
        assert optimizer.direction == "maximize"
    
    def test_create_sampler_tpe(self):
        """Test TPE sampler creation."""
        optimizer = OptunaOptimizer(sampler="tpe")
        sampler = optimizer._create_sampler()
        assert sampler is not None
    
    def test_create_sampler_random(self):
        """Test random sampler creation."""
        optimizer = OptunaOptimizer(sampler="random")
        sampler = optimizer._create_sampler()
        assert sampler is not None
    
    def test_create_sampler_invalid(self):
        """Test invalid sampler raises error."""
        optimizer = OptunaOptimizer(sampler="invalid")
        with pytest.raises(ValueError):
            optimizer._create_sampler()
    
    def test_create_pruner_median(self):
        """Test median pruner creation."""
        optimizer = OptunaOptimizer(pruner="median")
        pruner = optimizer._create_pruner()
        assert pruner is not None
    
    def test_create_pruner_percentile(self):
        """Test percentile pruner creation."""
        optimizer = OptunaOptimizer(pruner="percentile")
        pruner = optimizer._create_pruner()
        assert pruner is not None
    
    def test_search_space(self):
        """Test LightGBM search space definition."""
        from optuna.trial import Trial
        
        # Create a mock trial
        trial = Mock(spec=Trial)
        trial.suggest_int = Mock(side_effect=lambda name, low, high, **kwargs: (low + high) // 2)
        trial.suggest_float = Mock(side_effect=lambda name, low, high, **kwargs: (low + high) / 2)
        
        params = LightGBMSearchSpace.suggest_params(trial)
        
        # Check all expected parameters are present
        assert 'num_leaves' in params
        assert 'max_depth' in params
        assert 'learning_rate' in params
        assert 'lambda_l1' in params
        assert 'lambda_l2' in params
        assert 'feature_fraction' in params
        assert 'bagging_fraction' in params
    
    def test_optimize_simple(self, sample_training_data):
        """Test simple optimization."""
        optimizer = OptunaOptimizer(n_trials=2, timeout_seconds=10)
        
        with patch('src.training.hyperopt.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.training.walk_forward.get = Mock(return_value=100)
            mock_config.model.training.cv.get = Mock(return_value=24)
            mock_config.model.active_model = "lightgbm"
            mock_config.model.lightgbm.objective = "regression"
            mock_config.model.lightgbm.metric = "rmse"
            mock_config.model.lightgbm.early_stopping_rounds = 5
            mock_get_config.return_value = mock_config
            
            result = optimizer.optimize(
                sample_training_data,
                ['feature1', 'feature2', 'feature3', 'feature4'],
                'target',
                use_walk_forward=False
            )
            
            assert isinstance(result, OptimizationResult)
            assert result.best_params is not None
            assert result.best_value > 0
            assert result.n_trials >= 1


# =============================================================================
# ConfidenceCalibrator Tests
# =============================================================================

class TestConfidenceCalibrator:
    """Tests for ConfidenceCalibrator class."""
    
    def test_init_isotonic(self):
        """Test calibrator initialization with isotonic method."""
        calibrator = ConfidenceCalibrator(method="isotonic")
        assert calibrator.method == "isotonic"
        assert calibrator.is_fitted is False
    
    def test_init_platt(self):
        """Test calibrator initialization with Platt method."""
        calibrator = ConfidenceCalibrator(method="platt")
        assert calibrator.method == "platt"
        assert calibrator.is_fitted is False
    
    def test_init_invalid_method(self):
        """Test invalid calibration method raises error."""
        with pytest.raises(ValueError):
            ConfidenceCalibrator(method="invalid")
    
    def test_fit_isotonic(self):
        """Test fitting isotonic calibrator."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        y_proba = np.clip(np.abs(y_pred) / 0.01, 0, 1)
        
        calibrator = ConfidenceCalibrator(method="isotonic")
        calibrator.fit(y_true, y_pred, y_proba)
        
        assert calibrator.is_fitted is True
        assert calibrator.calibrator is not None
    
    def test_fit_platt(self):
        """Test fitting Platt calibrator."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        y_proba = np.clip(np.abs(y_pred) / 0.01, 0, 1)
        
        calibrator = ConfidenceCalibrator(method="platt")
        calibrator.fit(y_true, y_pred, y_proba)
        
        assert calibrator.is_fitted is True
        assert calibrator.calibrator is not None
    
    def test_fit_insufficient_data(self):
        """Test fitting with insufficient data."""
        y_true = np.array([0.1, 0.2])
        y_pred = np.array([0.15, 0.25])
        y_proba = np.array([0.5, 0.6])
        
        calibrator = ConfidenceCalibrator(min_samples=50)
        calibrator.fit(y_true, y_pred, y_proba)
        
        # Should not be fitted due to insufficient data
        assert calibrator.is_fitted is False
    
    def test_transform_not_fitted(self):
        """Test transform on unfitted calibrator."""
        calibrator = ConfidenceCalibrator()
        y_pred = np.array([0.1, 0.2])
        y_proba = np.array([0.5, 0.6])
        
        result = calibrator.transform(y_pred, y_proba)
        
        # Should return raw probabilities
        np.testing.assert_array_equal(result, y_proba)
    
    def test_transform_fitted(self):
        """Test transform on fitted calibrator."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        y_proba = np.clip(np.abs(y_pred) / 0.01, 0, 1)
        
        calibrator = ConfidenceCalibrator(method="isotonic")
        calibrator.fit(y_true, y_pred, y_proba)
        
        result = calibrator.transform(y_pred, y_proba)
        
        # Result should be in [0, 1]
        assert np.all(result >= 0)
        assert np.all(result <= 1)
    
    def test_regime_adjustments(self):
        """Test market regime adjustments."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        y_proba = np.clip(np.abs(y_pred) / 0.01, 0, 1)
        
        calibrator = ConfidenceCalibrator(method="isotonic")
        calibrator.fit(y_true, y_pred, y_proba)
        
        with patch('src.training.calibration.get_config') as mock_get_config:
            mock_config = Mock()
            mock_config.model.calibration.adjustment = {
                'high_vol_penalty': 0.1,
                'counter_trend_penalty': 0.05
            }
            mock_config.model.calibration.min_confidence = 0.3
            mock_get_config.return_value = mock_config
            
            # Test high volatility adjustment
            result_high_vol = calibrator.transform(
                y_pred, y_proba, is_high_vol=True
            )
            
            # Test counter-trend adjustment
            result_counter = calibrator.transform(
                y_pred, y_proba, is_counter_trend=True
            )
            
            # Both should be lower than base
            result_base = calibrator.transform(y_pred, y_proba)
            assert np.mean(result_high_vol) <= np.mean(result_base)
            assert np.mean(result_counter) <= np.mean(result_base)
    
    def test_save_load(self, tmp_path):
        """Test saving and loading calibrator."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        y_proba = np.clip(np.abs(y_pred) / 0.01, 0, 1)
        
        calibrator = ConfidenceCalibrator(method="isotonic")
        calibrator.fit(y_true, y_pred, y_proba)
        
        # Save
        save_path = tmp_path / "calibrator"
        calibrator.save(save_path)
        
        # Load
        loaded = ConfidenceCalibrator.load(save_path)
        
        assert loaded.is_fitted is True
        assert loaded.method == "isotonic"
        
        # Check predictions are same
        result_orig = calibrator.transform(y_pred, y_proba)
        result_loaded = loaded.transform(y_pred, y_proba)
        np.testing.assert_array_almost_equal(result_orig, result_loaded)


# =============================================================================
# ModelComparator Tests
# =============================================================================

class TestModelComparator:
    """Tests for ModelComparator class."""
    
    def test_init(self):
        """Test comparator initialization."""
        with patch('src.training.model_comparison.ModelRegistry'):
            comparator = ModelComparator()
            assert comparator.models == {}
            assert comparator.baseline_model is None
    
    def test_add_model_success(self, trained_model, tmp_path):
        """Test adding model to comparison."""
        # Save model first
        model_path = tmp_path / "model_v1"
        trained_model.save(model_path)
        
        with patch('src.training.model_comparison.ModelRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.load_version = Mock(return_value=trained_model)
            mock_registry.get_version_info = Mock(return_value={
                'version': 'v1',
                'trained_at': datetime.utcnow().isoformat()
            })
            mock_registry_class.return_value = mock_registry
            
            comparator = ModelComparator()
            success = comparator.add_model('v1', name='model_v1', is_baseline=True)
            
            assert success is True
            assert 'model_v1' in comparator.models
            assert comparator.baseline_model == 'model_v1'
    
    def test_compare_metrics(self, trained_model):
        """Test metric comparison."""
        with patch('src.training.model_comparison.ModelRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.load_version = Mock(return_value=trained_model)
            mock_registry.get_version_info = Mock(return_value={
                'version': 'v1',
                'trained_at': datetime.utcnow().isoformat()
            })
            mock_registry_class.return_value = mock_registry
            
            comparator = ModelComparator()
            comparator.add_model('v1', name='model_v1', is_baseline=True)
            
            result = comparator.compare_metrics()
            
            assert isinstance(result, ComparisonResult)
            assert len(result.metrics) > 0
            assert result.baseline_model == 'model_v1'
    
    def test_generate_report_text(self, trained_model):
        """Test text report generation."""
        with patch('src.training.model_comparison.ModelRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.load_version = Mock(return_value=trained_model)
            mock_registry.get_version_info = Mock(return_value={
                'version': 'v1',
                'trained_at': datetime.utcnow().isoformat()
            })
            mock_registry_class.return_value = mock_registry
            
            comparator = ModelComparator()
            comparator.add_model('v1', name='model_v1', is_baseline=True)
            comparator.compare_metrics()
            
            report = comparator.generate_report(format="text")
            
            assert isinstance(report, str)
            assert "MODEL COMPARISON REPORT" in report
            assert "model_v1" in report
    
    def test_generate_report_json(self, trained_model):
        """Test JSON report generation."""
        with patch('src.training.model_comparison.ModelRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.load_version = Mock(return_value=trained_model)
            mock_registry.get_version_info = Mock(return_value={
                'version': 'v1',
                'trained_at': datetime.utcnow().isoformat()
            })
            mock_registry_class.return_value = mock_registry
            
            comparator = ModelComparator()
            comparator.add_model('v1', name='model_v1', is_baseline=True)
            comparator.compare_metrics()
            
            report = comparator.generate_report(format="json")
            
            # Should be valid JSON
            data = json.loads(report)
            assert 'models' in data
            assert 'metrics' in data
    
    def test_generate_report_html(self, trained_model):
        """Test HTML report generation."""
        with patch('src.training.model_comparison.ModelRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.load_version = Mock(return_value=trained_model)
            mock_registry.get_version_info = Mock(return_value={
                'version': 'v1',
                'trained_at': datetime.utcnow().isoformat()
            })
            mock_registry_class.return_value = mock_registry
            
            comparator = ModelComparator()
            comparator.add_model('v1', name='model_v1', is_baseline=True)
            comparator.compare_metrics()
            
            report = comparator.generate_report(format="html")
            
            assert isinstance(report, str)
            assert "<!DOCTYPE html>" in report
            assert "<table>" in report
    
    def test_export_results(self, trained_model, tmp_path):
        """Test exporting comparison results."""
        with patch('src.training.model_comparison.ModelRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.load_version = Mock(return_value=trained_model)
            mock_registry.get_version_info = Mock(return_value={
                'version': 'v1',
                'trained_at': datetime.utcnow().isoformat()
            })
            mock_registry_class.return_value = mock_registry
            
            comparator = ModelComparator()
            comparator.add_model('v1', name='model_v1', is_baseline=True)
            comparator.compare_metrics()
            
            export_path = tmp_path / "report.json"
            success = comparator.export_results(export_path, format="json")
            
            assert success is True
            assert export_path.exists()


# =============================================================================
# TrainingResult Tests
# =============================================================================

class TestTrainingResult:
    """Tests for TrainingResult class."""
    
    def test_save_load(self, trained_model, tmp_path):
        """Test saving and loading training result."""
        save_path = tmp_path / "result"
        
        # Save
        trained_model.save(save_path)
        
        # Check files exist
        assert (save_path / "model.pkl").exists()
        assert (save_path / "metadata.json").exists()
        assert (save_path / "feature_importance.csv").exists()
        
        # Load
        loaded = TrainingResult.load(save_path)
        
        assert loaded.model is not None
        assert loaded.metrics is not None
        assert loaded.feature_importance is not None
        assert loaded.metrics.rmse == trained_model.metrics.rmse


# =============================================================================
# OptimizationResult Tests
# =============================================================================

class TestOptimizationResult:
    """Tests for OptimizationResult class."""
    
    def test_save_load(self, tmp_path):
        """Test saving and loading optimization result."""
        result = OptimizationResult(
            best_params={'learning_rate': 0.1, 'n_estimators': 100},
            best_value=0.005,
            best_trial=5,
            n_trials=10,
            study_name='test_study',
            optimization_time=60.0,
            trial_history=[
                {'trial_number': 0, 'value': 0.01},
                {'trial_number': 1, 'value': 0.008},
            ]
        )
        
        save_path = tmp_path / "optim_result"
        
        # Save
        result.save(save_path)
        
        # Check files exist
        assert (save_path / "best_params.json").exists()
        assert (save_path / "metadata.json").exists()
        assert (save_path / "trial_history.csv").exists()
        
        # Load
        loaded = OptimizationResult.load(save_path)
        
        assert loaded.best_value == result.best_value
        assert loaded.best_trial == result.best_trial
        assert loaded.n_trials == result.n_trials


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
