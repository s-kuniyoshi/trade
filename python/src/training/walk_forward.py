"""
Walk-Forward Validation module.

Provides time-series aware cross-validation for ML models.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger("training.walk_forward")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class WalkForwardSplit:
    """Represents a single walk-forward split."""
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size: int
    test_size: int


@dataclass
class WalkForwardResult:
    """Results from walk-forward validation."""
    splits: list[WalkForwardSplit]
    fold_metrics: list[dict[str, float]]
    aggregate_metrics: dict[str, float]
    predictions: pd.DataFrame | None = None
    
    def summary(self) -> str:
        """Get summary string."""
        m = self.aggregate_metrics
        return (
            f"Walk-Forward Validation Results\n"
            f"{'='*50}\n"
            f"Number of Folds: {len(self.splits)}\n"
            f"Avg Direction Accuracy: {m.get('avg_direction_accuracy', 0):.3f}\n"
            f"Avg RMSE: {m.get('avg_rmse', 0):.6f}\n"
            f"Avg Sharpe: {m.get('avg_sharpe', 0):.2f}\n"
            f"Avg Profit Factor: {m.get('avg_profit_factor', 0):.2f}\n"
        )


# =============================================================================
# Walk-Forward Validator
# =============================================================================

class WalkForwardValidator:
    """
    Walk-Forward Cross-Validation for time series.
    
    Implements expanding or rolling window walk-forward analysis
    to validate ML models without lookahead bias.
    """
    
    def __init__(
        self,
        train_window: int = 4380,  # ~6 months of H1 bars
        test_window: int = 730,    # ~1 month of H1 bars
        step: int | None = None,   # Step size (default: test_window)
        embargo: int = 24,         # Gap between train and test
        expanding: bool = False,   # Expanding vs rolling window
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            train_window: Number of bars for training
            test_window: Number of bars for testing
            step: Step size between folds (default: test_window)
            embargo: Gap between train and test to prevent leakage
            expanding: If True, use expanding window (cumulative training data)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step = step or test_window
        self.embargo = embargo
        self.expanding = expanding
    
    def get_splits(
        self,
        df: pd.DataFrame,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, WalkForwardSplit]]:
        """
        Generate walk-forward splits.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            List of (train_df, test_df, split_info) tuples
        """
        n = len(df)
        splits = []
        fold = 0
        
        if self.expanding:
            # Expanding window: training data grows with each fold
            train_start_idx = 0
            test_start_idx = self.train_window + self.embargo
            
            while test_start_idx + self.test_window <= n:
                train_end_idx = test_start_idx - self.embargo
                test_end_idx = test_start_idx + self.test_window
                
                train_df = df.iloc[train_start_idx:train_end_idx].copy()
                test_df = df.iloc[test_start_idx:test_end_idx].copy()
                
                split_info = WalkForwardSplit(
                    fold=fold,
                    train_start=df.index[train_start_idx],
                    train_end=df.index[train_end_idx - 1],
                    test_start=df.index[test_start_idx],
                    test_end=df.index[test_end_idx - 1],
                    train_size=len(train_df),
                    test_size=len(test_df),
                )
                
                splits.append((train_df, test_df, split_info))
                test_start_idx += self.step
                fold += 1
        else:
            # Rolling window: fixed training window size
            start_idx = 0
            
            while start_idx + self.train_window + self.embargo + self.test_window <= n:
                train_start_idx = start_idx
                train_end_idx = start_idx + self.train_window
                test_start_idx = train_end_idx + self.embargo
                test_end_idx = test_start_idx + self.test_window
                
                train_df = df.iloc[train_start_idx:train_end_idx].copy()
                test_df = df.iloc[test_start_idx:test_end_idx].copy()
                
                split_info = WalkForwardSplit(
                    fold=fold,
                    train_start=df.index[train_start_idx],
                    train_end=df.index[train_end_idx - 1],
                    test_start=df.index[test_start_idx],
                    test_end=df.index[test_end_idx - 1],
                    train_size=len(train_df),
                    test_size=len(test_df),
                )
                
                splits.append((train_df, test_df, split_info))
                start_idx += self.step
                fold += 1
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def validate(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        model_factory: Callable[[], Any],
        fit_params: dict[str, Any] | None = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.
        
        Args:
            df: DataFrame with features and target
            feature_columns: List of feature column names
            target_column: Target column name
            model_factory: Function that creates a new model instance
            fit_params: Additional parameters for model.fit()
            
        Returns:
            WalkForwardResult with metrics and predictions
        """
        if fit_params is None:
            fit_params = {}
        
        splits = self.get_splits(df)
        fold_metrics = []
        all_predictions = []
        
        for train_df, test_df, split_info in splits:
            logger.info(f"Fold {split_info.fold}: "
                       f"train={split_info.train_size}, test={split_info.test_size}")
            
            # Prepare data
            X_train = train_df[feature_columns].dropna()
            y_train = train_df.loc[X_train.index, target_column]
            
            X_test = test_df[feature_columns].dropna()
            y_test = test_df.loc[X_test.index, target_column]
            
            # Skip if insufficient data
            if len(X_train) < 100 or len(X_test) < 10:
                logger.warning(f"Fold {split_info.fold}: Insufficient data, skipping")
                continue
            
            # Create and train model
            model = model_factory()
            
            # Handle LightGBM/XGBoost early stopping
            if "eval_set" in fit_params or hasattr(model, "fit"):
                try:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        **fit_params
                    )
                except TypeError:
                    # Model doesn't support eval_set
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_fold_metrics(y_test, y_pred, test_df, target_column)
            metrics["fold"] = split_info.fold
            fold_metrics.append(metrics)
            
            # Store predictions
            pred_df = pd.DataFrame({
                "actual": y_test,
                "predicted": y_pred,
                "fold": split_info.fold,
            }, index=X_test.index)
            all_predictions.append(pred_df)
        
        # Aggregate metrics
        aggregate = self._aggregate_metrics(fold_metrics)
        
        # Combine predictions
        predictions = pd.concat(all_predictions) if all_predictions else None
        
        return WalkForwardResult(
            splits=[s[2] for s in splits],
            fold_metrics=fold_metrics,
            aggregate_metrics=aggregate,
            predictions=predictions,
        )
    
    def _calculate_fold_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        test_df: pd.DataFrame,
        target_column: str,
    ) -> dict[str, float]:
        """Calculate metrics for a single fold."""
        # Remove NaN
        valid_idx = ~(y_true.isna() | pd.isna(y_pred))
        y_true = y_true[valid_idx]
        y_pred = y_pred[valid_idx]
        
        if len(y_true) == 0:
            return {"rmse": 0.0, "mae": 0.0, "direction_accuracy": 0.0}
        
        # Regression metrics
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Direction accuracy
        actual_direction = (y_true > 0).astype(int)
        pred_direction = (y_pred > 0).astype(int)
        direction_accuracy = (actual_direction == pred_direction).mean()
        
        # Trading metrics (simplified)
        # Assume we trade when prediction is positive
        trades = y_pred != 0
        if trades.any():
            returns = np.where(y_pred > 0, y_true, -y_true)
            
            # Only count actual trades
            trade_returns = returns[trades]
            winners = trade_returns[trade_returns > 0]
            losers = trade_returns[trade_returns <= 0]
            
            total_profit = winners.sum() if len(winners) > 0 else 0.0
            total_loss = abs(losers.sum()) if len(losers) > 0 else 0.0
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")
            win_rate = len(winners) / len(trade_returns) if len(trade_returns) > 0 else 0.0
            
            # Sharpe ratio
            if len(trade_returns) > 1 and np.std(trade_returns) > 0:
                sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252 * 24)
            else:
                sharpe = 0.0
        else:
            profit_factor = 0.0
            win_rate = 0.0
            sharpe = 0.0
        
        return {
            "rmse": rmse,
            "mae": mae,
            "direction_accuracy": direction_accuracy,
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "n_samples": len(y_true),
        }
    
    def _aggregate_metrics(
        self,
        fold_metrics: list[dict[str, float]],
    ) -> dict[str, float]:
        """Aggregate metrics across all folds."""
        if not fold_metrics:
            return {}
        
        metrics_df = pd.DataFrame(fold_metrics)
        
        aggregate = {}
        for col in metrics_df.columns:
            if col == "fold":
                continue
            values = metrics_df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) > 0:
                aggregate[f"avg_{col}"] = values.mean()
                aggregate[f"std_{col}"] = values.std()
                aggregate[f"min_{col}"] = values.min()
                aggregate[f"max_{col}"] = values.max()
        
        aggregate["n_folds"] = len(fold_metrics)
        
        return aggregate


# =============================================================================
# Purged K-Fold for Time Series
# =============================================================================

class PurgedKFold:
    """
    Purged K-Fold cross-validation for time series.
    
    Prevents data leakage by adding embargo periods between
    training and test sets.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo_bars: int = 24,
    ):
        """
        Initialize purged k-fold.
        
        Args:
            n_splits: Number of folds
            embargo_bars: Number of bars to skip between train and test
        """
        self.n_splits = n_splits
        self.embargo_bars = embargo_bars
    
    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional)
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        n = len(X)
        fold_size = n // self.n_splits
        
        splits = []
        
        for i in range(self.n_splits):
            # Test fold boundaries
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n
            
            # Train indices (exclude test fold + embargo)
            train_indices = []
            
            # Before test fold
            if test_start > 0:
                train_end = max(0, test_start - self.embargo_bars)
                train_indices.extend(range(0, train_end))
            
            # After test fold
            if test_end < n:
                train_start = min(n, test_end + self.embargo_bars)
                train_indices.extend(range(train_start, n))
            
            test_indices = list(range(test_start, test_end))
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((np.array(train_indices), np.array(test_indices)))
        
        return splits
    
    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits
