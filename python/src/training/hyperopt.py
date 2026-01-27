"""
Hyperparameter optimization module using Optuna.

Provides automated hyperparameter tuning for LightGBM models using
Optuna's optimization algorithms with walk-forward validation.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner, PercentilePruner
from optuna.samplers import RandomSampler, TPESampler
from optuna.trial import Trial

from ..utils.config import get_config
from ..utils.logger import get_logger
from .evaluation import ModelEvaluator
from .trainer import TrainingConfig
from .walk_forward import WalkForwardValidator

logger = get_logger("training.hyperopt")


# =============================================================================
# Hyperparameter Search Spaces
# =============================================================================

class LightGBMSearchSpace:
    """Define search spaces for LightGBM hyperparameters."""
    
    @staticmethod
    def suggest_params(trial: Trial) -> dict[str, Any]:
        """
        Suggest hyperparameters for LightGBM.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of suggested parameters
        """
        return {
            # Tree parameters
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
            
            # Learning parameters
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=100),
            
            # Regularization
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.1),
            
            # Sampling
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        }


# =============================================================================
# Optimization Result
# =============================================================================

@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""
    best_params: dict[str, Any]
    best_value: float
    best_trial: int
    n_trials: int
    study_name: str
    optimization_time: float
    trial_history: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def save(self, path: Path | str) -> None:
        """
        Save optimization result to disk.
        
        Args:
            path: Directory to save results
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save best parameters
        with open(path / "best_params.json", "w") as f:
            json.dump(self.best_params, f, indent=2)
        
        # Save metadata
        metadata = {
            "best_value": self.best_value,
            "best_trial": self.best_trial,
            "n_trials": self.n_trials,
            "study_name": self.study_name,
            "optimization_time": self.optimization_time,
            "created_at": self.created_at.isoformat(),
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save trial history
        if self.trial_history:
            trial_df = pd.DataFrame(self.trial_history)
            trial_df.to_csv(path / "trial_history.csv", index=False)
        
        logger.info(f"Optimization result saved to {path}")
    
    @classmethod
    def load(cls, path: Path | str) -> "OptimizationResult":
        """
        Load optimization result from disk.
        
        Args:
            path: Directory containing saved results
            
        Returns:
            OptimizationResult instance
        """
        path = Path(path)
        
        # Load best parameters
        with open(path / "best_params.json", "r") as f:
            best_params = json.load(f)
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Load trial history if available
        trial_history = []
        trial_history_path = path / "trial_history.csv"
        if trial_history_path.exists():
            trial_df = pd.read_csv(trial_history_path)
            trial_history = trial_df.to_dict("records")
        
        return cls(
            best_params=best_params,
            best_value=metadata["best_value"],
            best_trial=metadata["best_trial"],
            n_trials=metadata["n_trials"],
            study_name=metadata["study_name"],
            optimization_time=metadata["optimization_time"],
            trial_history=trial_history,
            created_at=datetime.fromisoformat(metadata["created_at"]),
        )


# =============================================================================
# Optuna Optimizer
# =============================================================================

class OptunaOptimizer:
    """
    Hyperparameter optimizer using Optuna.
    
    Optimizes LightGBM hyperparameters using Optuna's optimization algorithms
    with walk-forward validation for robust parameter selection.
    """
    
    def __init__(
        self,
        n_trials: int = 50,
        timeout_seconds: int | None = None,
        sampler: str = "tpe",
        pruner: str = "median",
        direction: str = "minimize",
        seed: int = 42,
        n_jobs: int = 1,
    ):
        """
        Initialize optimizer.
        
        Args:
            n_trials: Number of trials to run
            timeout_seconds: Timeout for optimization (None = no limit)
            sampler: Sampler type ("tpe" or "random")
            pruner: Pruner type ("median" or "percentile")
            direction: Optimization direction ("minimize" or "maximize")
            seed: Random seed
            n_jobs: Number of parallel jobs
        """
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.sampler_type = sampler
        self.pruner_type = pruner
        self.direction = direction
        self.seed = seed
        self.n_jobs = n_jobs
        self.evaluator = ModelEvaluator()
        self.study: optuna.Study | None = None
    
    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create sampler based on configuration."""
        if self.sampler_type == "tpe":
            return TPESampler(seed=self.seed)
        elif self.sampler_type == "random":
            return RandomSampler(seed=self.seed)
        else:
            raise ValueError(f"Unknown sampler: {self.sampler_type}")
    
    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Create pruner based on configuration."""
        if self.pruner_type == "median":
            return MedianPruner(n_warmup_steps=5)
        elif self.pruner_type == "percentile":
            return PercentilePruner(percentile=25, n_warmup_steps=5)
        else:
            raise ValueError(f"Unknown pruner: {self.pruner_type}")
    
    def _objective(
        self,
        trial: Trial,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        training_config: TrainingConfig,
        use_walk_forward: bool = True,
    ) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial
            df: Dataset
            feature_columns: Feature column names
            target_column: Target column name
            training_config: Training configuration
            use_walk_forward: Whether to use walk-forward validation
            
        Returns:
            Objective value (RMSE to minimize)
        """
        # Suggest parameters
        params = LightGBMSearchSpace.suggest_params(trial)
        
        # Add fixed parameters
        params.update({
            "objective": training_config.objective,
            "metric": training_config.metric,
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_jobs": -1,
            "random_state": self.seed,
        })
        
        try:
            if use_walk_forward:
                # Walk-forward validation
                rmse = self._evaluate_walk_forward(
                    df, feature_columns, target_column, params, training_config, trial
                )
            else:
                # Simple train/val split
                rmse = self._evaluate_simple(
                    df, feature_columns, target_column, params, training_config
                )
            
            return rmse
        
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float("inf")
    
    def _evaluate_simple(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        params: dict[str, Any],
        training_config: TrainingConfig,
    ) -> float:
        """
        Evaluate parameters with simple train/val split.
        
        Args:
            df: Dataset
            feature_columns: Feature column names
            target_column: Target column name
            params: Model parameters
            training_config: Training configuration
            
        Returns:
            RMSE on validation set
        """
        # Clean data
        df_clean = df[feature_columns + [target_column]].dropna()
        X = df_clean[feature_columns]
        y = df_clean[target_column]
        
        # Split
        split_idx = int(len(X) * (1 - training_config.test_ratio))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(training_config.early_stopping_rounds),
                lgb.log_evaluation(period=0),
            ],
        )
        
        # Evaluate
        y_pred = model.predict(X_val)
        metrics = self.evaluator.evaluate_predictions(y_val, y_pred)
        
        return metrics["rmse"]
    
    def _evaluate_walk_forward(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        params: dict[str, Any],
        training_config: TrainingConfig,
        trial: Trial | None = None,
    ) -> float:
        """
        Evaluate parameters with walk-forward validation.
        
        Args:
            df: Dataset
            feature_columns: Feature column names
            target_column: Target column name
            params: Model parameters
            training_config: Training configuration
            trial: Optuna trial (for pruning)
            
        Returns:
            Average RMSE across folds
        """
        validator = WalkForwardValidator(
            train_window=training_config.wf_train_window,
            test_window=training_config.wf_test_window,
            step=training_config.wf_step,
            embargo=training_config.wf_embargo,
        )
        
        splits = validator.get_splits(df)
        rmse_values = []
        
        for fold_idx, (train_df, test_df, split_info) in enumerate(splits):
            # Prepare data
            X_train = train_df[feature_columns].dropna()
            y_train = train_df.loc[X_train.index, target_column]
            
            X_test = test_df[feature_columns].dropna()
            y_test = test_df.loc[X_test.index, target_column]
            
            # Skip if insufficient data
            if len(X_train) < 100 or len(X_test) < 10:
                continue
            
            # Train
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[test_data],
                valid_names=["test"],
                callbacks=[
                    lgb.early_stopping(training_config.early_stopping_rounds),
                    lgb.log_evaluation(period=0),
                ],
            )
            
            # Evaluate
            y_pred = model.predict(X_test)
            metrics = self.evaluator.evaluate_predictions(y_test, y_pred)
            rmse_values.append(metrics["rmse"])
            
            # Report intermediate value for pruning
            if trial is not None:
                trial.report(np.mean(rmse_values), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        if not rmse_values:
            return float("inf")
        
        return np.mean(rmse_values)
    
    def optimize(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        training_config: TrainingConfig | None = None,
        use_walk_forward: bool = True,
        study_name: str | None = None,
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.
        
        Args:
            df: Dataset with features and target
            feature_columns: Feature column names
            target_column: Target column name
            training_config: Training configuration (if None, load from config)
            use_walk_forward: Whether to use walk-forward validation
            study_name: Name for the study (for tracking)
            
        Returns:
            OptimizationResult with best parameters
        """
        if training_config is None:
            training_config = TrainingConfig.from_config()
        
        if study_name is None:
            study_name = f"lightgbm_opt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting hyperparameter optimization: {study_name}")
        logger.info(f"Sampler: {self.sampler_type}, Pruner: {self.pruner_type}")
        logger.info(f"N trials: {self.n_trials}, Timeout: {self.timeout_seconds}s")
        
        # Create study
        sampler = self._create_sampler()
        pruner = self._create_pruner()
        
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
        )
        
        # Optimize
        start_time = datetime.utcnow()
        
        self.study.optimize(
            lambda trial: self._objective(
                trial, df, feature_columns, target_column, training_config, use_walk_forward
            ),
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )
        
        optimization_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Extract results
        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        # Build trial history
        trial_history = []
        for trial in self.study.trials:
            trial_history.append({
                "trial_number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
                "params": json.dumps(trial.params),
            })
        
        result = OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            best_trial=best_trial.number,
            n_trials=len(self.study.trials),
            study_name=study_name,
            optimization_time=optimization_time,
            trial_history=trial_history,
        )
        
        logger.info(f"Optimization complete!")
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best RMSE: {best_value:.6f}")
        logger.info(f"Time: {optimization_time:.1f}s")
        logger.info(f"Best parameters: {best_params}")
        
        return result
    
    def save_study(self, path: Path | str) -> None:
        """
        Save optimization study to disk.
        
        Args:
            path: Path to save study
        """
        if self.study is None:
            raise ValueError("No study to save. Run optimize() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump(self.study, f)
        
        logger.info(f"Study saved to {path}")
    
    def load_study(self, path: Path | str) -> None:
        """
        Load optimization study from disk.
        
        Args:
            path: Path to study file
        """
        path = Path(path)
        
        with open(path, "rb") as f:
            self.study = pickle.load(f)
        
        logger.info(f"Study loaded from {path}")
    
    def get_best_params(self) -> dict[str, Any]:
        """
        Get best parameters from study.
        
        Returns:
            Dictionary of best parameters
        """
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")
        
        return self.study.best_trial.params
    
    def get_study_summary(self) -> dict[str, Any]:
        """
        Get summary of optimization study.
        
        Returns:
            Dictionary with study summary
        """
        if self.study is None:
            raise ValueError("No study available. Run optimize() first.")
        
        trials = self.study.trials
        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
        
        values = [t.value for t in completed_trials if t.value is not None]
        
        return {
            "n_trials": len(trials),
            "n_completed": len(completed_trials),
            "n_pruned": len(pruned_trials),
            "best_value": self.study.best_value,
            "best_trial": self.study.best_trial.number,
            "mean_value": np.mean(values) if values else None,
            "std_value": np.std(values) if values else None,
            "min_value": np.min(values) if values else None,
            "max_value": np.max(values) if values else None,
        }


# =============================================================================
# Quick Optimization Function
# =============================================================================

def optimize_hyperparameters(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    n_trials: int = 50,
    use_walk_forward: bool = True,
    save_path: Path | str | None = None,
) -> OptimizationResult:
    """
    Quick function to optimize hyperparameters.
    
    Args:
        df: Dataset with features and target
        feature_columns: Feature column names
        target_column: Target column name
        n_trials: Number of optimization trials
        use_walk_forward: Whether to use walk-forward validation
        save_path: Path to save results (optional)
        
    Returns:
        OptimizationResult with best parameters
    """
    config = get_config()
    hyperopt_config = config.model.training.hyperopt
    
    optimizer = OptunaOptimizer(
        n_trials=n_trials,
        timeout_seconds=hyperopt_config.get("timeout_seconds", 3600),
        sampler=hyperopt_config.get("method", "optuna"),
    )
    
    result = optimizer.optimize(
        df, feature_columns, target_column,
        use_walk_forward=use_walk_forward,
    )
    
    if save_path is not None:
        result.save(save_path)
    
    return result
