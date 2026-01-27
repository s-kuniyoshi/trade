"""
Model training module.

Provides training pipeline for LightGBM and other ML models.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.config import get_config
from ..utils.logger import get_logger
from .evaluation import ModelEvaluator, PerformanceMetrics
from .walk_forward import WalkForwardValidator

logger = get_logger("training.trainer")


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Data split
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Walk-forward
    use_walk_forward: bool = True
    wf_train_window: int = 4380  # ~6 months H1
    wf_test_window: int = 730    # ~1 month H1
    wf_step: int = 730
    wf_embargo: int = 24
    
    # Model parameters
    model_type: str = "lightgbm"
    objective: str = "regression"
    metric: str = "rmse"
    
    # Training
    n_estimators: int = 1000
    early_stopping_rounds: int = 50
    verbose: int = -1
    
    # Evaluation
    min_direction_accuracy: float = 0.52
    min_profit_factor: float = 1.2
    
    @classmethod
    def from_config(cls) -> "TrainingConfig":
        """Load from global config."""
        config = get_config()
        model_config = config.model
        
        return cls(
            train_ratio=model_config.training.train_ratio,
            validation_ratio=model_config.training.validation_ratio,
            test_ratio=model_config.training.test_ratio,
            use_walk_forward=model_config.training.walk_forward.get("enabled", True),
            wf_train_window=model_config.training.walk_forward.get("train_window_bars", 4380),
            wf_test_window=model_config.training.walk_forward.get("test_window_bars", 730),
            wf_step=model_config.training.walk_forward.get("step_bars", 730),
            wf_embargo=model_config.training.cv.get("embargo_bars", 24),
            model_type=model_config.active_model,
            objective=model_config.lightgbm.objective,
            metric=model_config.lightgbm.metric,
            n_estimators=model_config.lightgbm.n_estimators,
            early_stopping_rounds=model_config.lightgbm.early_stopping_rounds,
            verbose=model_config.lightgbm.verbose,
        )


@dataclass
class TrainingResult:
    """Results from model training."""
    model: Any
    metrics: PerformanceMetrics
    feature_importance: pd.DataFrame
    training_history: dict[str, list[float]] = field(default_factory=dict)
    validation_predictions: pd.DataFrame | None = None
    test_predictions: pd.DataFrame | None = None
    config: TrainingConfig = field(default_factory=TrainingConfig)
    trained_at: datetime = field(default_factory=datetime.utcnow)
    
    def save(self, path: Path | str) -> None:
        """Save training result to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            "metrics": self.metrics.to_dict(),
            "config": {
                "model_type": self.config.model_type,
                "objective": self.config.objective,
                "n_estimators": self.config.n_estimators,
            },
            "trained_at": self.trained_at.isoformat(),
            "feature_names": list(self.feature_importance["feature"].values),
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature importance
        self.feature_importance.to_csv(path / "feature_importance.csv", index=False)
        
        # Save predictions if available
        if self.validation_predictions is not None:
            self.validation_predictions.to_csv(path / "validation_predictions.csv")
        if self.test_predictions is not None:
            self.test_predictions.to_csv(path / "test_predictions.csv")
        
        logger.info(f"Training result saved to {path}")
    
    @classmethod
    def load(cls, path: Path | str) -> "TrainingResult":
        """Load training result from disk."""
        path = Path(path)
        
        # Load model
        with open(path / "model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Load feature importance
        feature_importance = pd.read_csv(path / "feature_importance.csv")
        
        # Create metrics from dict
        metrics = PerformanceMetrics(**metadata["metrics"])
        
        # Load predictions if available
        val_pred_path = path / "validation_predictions.csv"
        validation_predictions = pd.read_csv(val_pred_path, index_col=0) if val_pred_path.exists() else None
        
        test_pred_path = path / "test_predictions.csv"
        test_predictions = pd.read_csv(test_pred_path, index_col=0) if test_pred_path.exists() else None
        
        return cls(
            model=model,
            metrics=metrics,
            feature_importance=feature_importance,
            validation_predictions=validation_predictions,
            test_predictions=test_predictions,
            trained_at=datetime.fromisoformat(metadata["trained_at"]),
        )


# =============================================================================
# Model Trainer
# =============================================================================

class ModelTrainer:
    """
    Model training pipeline.
    
    Handles data preparation, model training, evaluation, and persistence.
    """
    
    def __init__(self, config: TrainingConfig | None = None):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig.from_config()
        self.evaluator = ModelEvaluator()
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        test_size: float | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare train/validation/test splits.
        
        Args:
            df: Full dataset
            feature_columns: Feature column names
            target_column: Target column name
            test_size: Test set size (if None, use config)
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        # Remove rows with missing values
        df_clean = df[feature_columns + [target_column]].dropna()
        
        X = df_clean[feature_columns]
        y = df_clean[target_column]
        
        if test_size is None:
            test_size = self.config.test_ratio
        
        # Time-series aware split (no shuffle)
        train_val_size = 1.0 - test_size
        val_size_adjusted = self.config.validation_ratio / train_val_size
        
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, shuffle=False
        )
        
        logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        return X_train, X_val, y_train, y_val
    
    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict[str, Any] | None = None,
    ) -> tuple[lgb.Booster, dict[str, list[float]]]:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            params: Model parameters (if None, use config)
            
        Returns:
            Tuple of (trained model, training history)
        """
        if params is None:
            config = get_config()
            params = config.model.lightgbm.to_params()
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Training history callback
        history = {"train": [], "valid": []}
        
        def record_eval(env):
            history["train"].append(env.evaluation_result_list[0][2])
            if len(env.evaluation_result_list) > 1:
                history["valid"].append(env.evaluation_result_list[1][2])
        
        # Train
        logger.info("Training LightGBM model...")
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds),
                lgb.log_evaluation(period=100),
                record_eval,
            ],
        )
        
        logger.info(f"Training complete. Best iteration: {model.best_iteration}")
        
        return model, history
    
    def train(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        params: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """
        Train model with full pipeline.
        
        Args:
            df: Full dataset with features and target
            feature_columns: Feature column names
            target_column: Target column name
            params: Model parameters
            
        Returns:
            TrainingResult with model and metrics
        """
        # Prepare data
        X_train, X_val, y_train, y_val = self.prepare_data(
            df, feature_columns, target_column
        )
        
        # Train model
        if self.config.model_type == "lightgbm":
            model, history = self.train_lightgbm(
                X_train, y_train, X_val, y_val, params
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Evaluate
        train_metrics = self.evaluator.evaluate_predictions(y_train, y_train_pred)
        val_metrics = self.evaluator.evaluate_predictions(y_val, y_val_pred)
        
        logger.info(f"Train RMSE: {train_metrics['rmse']:.6f}, "
                   f"Val RMSE: {val_metrics['rmse']:.6f}")
        logger.info(f"Train Direction Acc: {train_metrics['direction_accuracy']:.3f}, "
                   f"Val Direction Acc: {val_metrics['direction_accuracy']:.3f}")
        
        # Feature importance
        if hasattr(model, "feature_importance"):
            importance = model.feature_importance(importance_type="gain")
            feature_importance = pd.DataFrame({
                "feature": feature_columns,
                "importance": importance,
            }).sort_values("importance", ascending=False)
        else:
            feature_importance = pd.DataFrame()
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            rmse=val_metrics["rmse"],
            mae=val_metrics["mae"],
            r_squared=val_metrics["r_squared"],
            direction_accuracy=val_metrics["direction_accuracy"],
        )
        
        # Store predictions
        validation_predictions = pd.DataFrame({
            "actual": y_val,
            "predicted": y_val_pred,
        }, index=X_val.index)
        
        result = TrainingResult(
            model=model,
            metrics=metrics,
            feature_importance=feature_importance,
            training_history=history,
            validation_predictions=validation_predictions,
            config=self.config,
        )
        
        return result
    
    def train_walk_forward(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        params: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """
        Train with walk-forward validation.
        
        Args:
            df: Full dataset
            feature_columns: Feature column names
            target_column: Target column name
            params: Model parameters
            
        Returns:
            TrainingResult with best model
        """
        validator = WalkForwardValidator(
            train_window=self.config.wf_train_window,
            test_window=self.config.wf_test_window,
            step=self.config.wf_step,
            embargo=self.config.wf_embargo,
        )
        
        def model_factory():
            if self.config.model_type == "lightgbm":
                config = get_config()
                lgb_params = params or config.model.lightgbm.to_params()
                return lgb.LGBMRegressor(**lgb_params)
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        logger.info("Starting walk-forward validation...")
        wf_result = validator.validate(
            df,
            feature_columns,
            target_column,
            model_factory,
            fit_params={
                "verbose": False,
                "callbacks": [lgb.early_stopping(self.config.early_stopping_rounds)],
            }
        )
        
        logger.info(f"Walk-forward complete: {wf_result.aggregate_metrics.get('n_folds', 0)} folds")
        logger.info(f"Avg Direction Accuracy: {wf_result.aggregate_metrics.get('avg_direction_accuracy', 0):.3f}")
        logger.info(f"Avg RMSE: {wf_result.aggregate_metrics.get('avg_rmse', 0):.6f}")
        
        # Train final model on all data
        logger.info("Training final model on full dataset...")
        X = df[feature_columns].dropna()
        y = df.loc[X.index, target_column]
        
        # Use last 80% for training
        train_size = int(len(X) * 0.8)
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X.iloc[train_size:]
        y_val = y.iloc[train_size:]
        
        if self.config.model_type == "lightgbm":
            model, history = self.train_lightgbm(
                X_train, y_train, X_val, y_val, params
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Evaluate final model
        y_val_pred = model.predict(X_val)
        val_metrics = self.evaluator.evaluate_predictions(y_val, y_val_pred)
        
        # Feature importance
        if hasattr(model, "feature_importance"):
            importance = model.feature_importance(importance_type="gain")
            feature_importance = pd.DataFrame({
                "feature": feature_columns,
                "importance": importance,
            }).sort_values("importance", ascending=False)
        else:
            feature_importance = pd.DataFrame()
        
        # Create metrics
        metrics = PerformanceMetrics(
            rmse=val_metrics["rmse"],
            mae=val_metrics["mae"],
            r_squared=val_metrics["r_squared"],
            direction_accuracy=val_metrics["direction_accuracy"],
        )
        
        result = TrainingResult(
            model=model,
            metrics=metrics,
            feature_importance=feature_importance,
            training_history=history,
            validation_predictions=wf_result.predictions,
            config=self.config,
        )
        
        return result
    
    def should_deploy(self, result: TrainingResult) -> tuple[bool, str]:
        """
        Check if model meets deployment criteria.
        
        Args:
            result: Training result
            
        Returns:
            Tuple of (should_deploy, reason)
        """
        metrics = result.metrics
        
        # Check direction accuracy
        if metrics.direction_accuracy < self.config.min_direction_accuracy:
            return False, f"Direction accuracy {metrics.direction_accuracy:.3f} < {self.config.min_direction_accuracy}"
        
        # Check if model is better than random
        if metrics.direction_accuracy < 0.51:
            return False, "Model not better than random"
        
        # Check RMSE is reasonable
        if metrics.rmse > 0.1:  # Arbitrary threshold
            return False, f"RMSE {metrics.rmse:.6f} too high"
        
        return True, "All criteria met"


# =============================================================================
# Quick Training Function
# =============================================================================

def train_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    use_walk_forward: bool = True,
    save_path: Path | str | None = None,
) -> TrainingResult:
    """
    Quick function to train a model.
    
    Args:
        df: Dataset with features and target
        feature_columns: Feature column names
        target_column: Target column name
        use_walk_forward: Whether to use walk-forward validation
        save_path: Path to save model (optional)
        
    Returns:
        TrainingResult
    """
    trainer = ModelTrainer()
    
    if use_walk_forward:
        result = trainer.train_walk_forward(df, feature_columns, target_column)
    else:
        result = trainer.train(df, feature_columns, target_column)
    
    # Check deployment criteria
    should_deploy, reason = trainer.should_deploy(result)
    logger.info(f"Deployment check: {should_deploy} - {reason}")
    
    # Save if path provided
    if save_path is not None:
        result.save(save_path)
    
    return result
