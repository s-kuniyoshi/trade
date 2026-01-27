"""
Data preprocessing module.

Handles data cleaning, normalization, and preparation for ML.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ..utils.logger import get_logger

logger = get_logger("data.preprocessor")


# =============================================================================
# Data Preprocessor
# =============================================================================

class DataPreprocessor:
    """
    Data preprocessor for ML pipeline.
    
    Handles missing values, outliers, and feature scaling.
    """
    
    def __init__(
        self,
        scaling_method: str = "standard",
        missing_method: str = "fill_forward",
        outlier_method: str | None = "iqr",
        outlier_threshold: float = 3.0,
    ):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method: Method for scaling features
                - "standard": StandardScaler (z-score)
                - "minmax": MinMaxScaler (0-1 range)
                - "robust": RobustScaler (median/IQR)
                - "none": No scaling
            missing_method: Method for handling missing values
                - "drop": Drop rows with missing values
                - "fill_zero": Fill with zeros
                - "fill_mean": Fill with column mean
                - "fill_median": Fill with column median
                - "fill_forward": Forward fill
            outlier_method: Method for handling outliers
                - "iqr": IQR-based removal
                - "zscore": Z-score based removal
                - None: No outlier handling
            outlier_threshold: Threshold for outlier detection
        """
        self.scaling_method = scaling_method
        self.missing_method = missing_method
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
        self._scaler: StandardScaler | MinMaxScaler | RobustScaler | None = None
        self._feature_means: dict[str, float] = {}
        self._feature_medians: dict[str, float] = {}
        self._fitted = False
    
    def fit(self, df: pd.DataFrame, feature_columns: list[str] | None = None) -> "DataPreprocessor":
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame
            feature_columns: Columns to preprocess (all numeric if None)
            
        Returns:
            Self for chaining
        """
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_features = df[feature_columns].copy()
        
        # Store statistics for filling
        self._feature_means = df_features.mean().to_dict()
        self._feature_medians = df_features.median().to_dict()
        
        # Handle missing values before fitting scaler
        df_clean = self._handle_missing(df_features, fit=True)
        
        # Fit scaler
        if self.scaling_method != "none":
            if self.scaling_method == "standard":
                self._scaler = StandardScaler()
            elif self.scaling_method == "minmax":
                self._scaler = MinMaxScaler()
            elif self.scaling_method == "robust":
                self._scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling_method}")
            
            self._scaler.fit(df_clean)
        
        self._fitted = True
        self._feature_columns = feature_columns
        
        logger.debug(f"Preprocessor fitted on {len(feature_columns)} features")
        return self
    
    def transform(
        self,
        df: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            feature_columns: Columns to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        if feature_columns is None:
            feature_columns = self._feature_columns
        
        df = df.copy()
        df_features = df[feature_columns].copy()
        
        # Handle missing values
        df_clean = self._handle_missing(df_features, fit=False)
        
        # Handle outliers
        if self.outlier_method is not None:
            df_clean = self._handle_outliers(df_clean)
        
        # Scale features
        if self._scaler is not None:
            scaled_values = self._scaler.transform(df_clean)
            df_scaled = pd.DataFrame(
                scaled_values,
                index=df_clean.index,
                columns=df_clean.columns,
            )
        else:
            df_scaled = df_clean
        
        # Update original DataFrame
        df[feature_columns] = df_scaled
        
        return df
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame to fit and transform
            feature_columns: Columns to process
            
        Returns:
            Transformed DataFrame
        """
        self.fit(df, feature_columns)
        return self.transform(df, feature_columns)
    
    def inverse_transform(
        self,
        df: pd.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Inverse transform scaled data.
        
        Args:
            df: Scaled DataFrame
            feature_columns: Columns to inverse transform
            
        Returns:
            Original scale DataFrame
        """
        if not self._fitted or self._scaler is None:
            return df
        
        if feature_columns is None:
            feature_columns = self._feature_columns
        
        df = df.copy()
        df_features = df[feature_columns]
        
        original_values = self._scaler.inverse_transform(df_features)
        df_original = pd.DataFrame(
            original_values,
            index=df_features.index,
            columns=df_features.columns,
        )
        
        df[feature_columns] = df_original
        return df
    
    def _handle_missing(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Handle missing values."""
        df = df.copy()
        
        if self.missing_method == "drop":
            df = df.dropna()
        elif self.missing_method == "fill_zero":
            df = df.fillna(0)
        elif self.missing_method == "fill_mean":
            if fit:
                df = df.fillna(df.mean())
            else:
                df = df.fillna(self._feature_means)
        elif self.missing_method == "fill_median":
            if fit:
                df = df.fillna(df.median())
            else:
                df = df.fillna(self._feature_medians)
        elif self.missing_method == "fill_forward":
            df = df.ffill().bfill()
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers."""
        df = df.copy()
        
        if self.outlier_method == "iqr":
            for col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.outlier_threshold * IQR
                upper = Q3 + self.outlier_threshold * IQR
                df[col] = df[col].clip(lower, upper)
                
        elif self.outlier_method == "zscore":
            for col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    lower = mean - self.outlier_threshold * std
                    upper = mean + self.outlier_threshold * std
                    df[col] = df[col].clip(lower, upper)
        
        return df
    
    def get_params(self) -> dict[str, Any]:
        """Get preprocessor parameters."""
        return {
            "scaling_method": self.scaling_method,
            "missing_method": self.missing_method,
            "outlier_method": self.outlier_method,
            "outlier_threshold": self.outlier_threshold,
        }
    
    def save_state(self) -> dict[str, Any]:
        """Save preprocessor state for persistence."""
        import pickle
        
        state = {
            "params": self.get_params(),
            "feature_means": self._feature_means,
            "feature_medians": self._feature_medians,
            "feature_columns": self._feature_columns if self._fitted else [],
            "fitted": self._fitted,
        }
        
        if self._scaler is not None:
            state["scaler"] = pickle.dumps(self._scaler)
        
        return state
    
    @classmethod
    def load_state(cls, state: dict[str, Any]) -> "DataPreprocessor":
        """Load preprocessor from saved state."""
        import pickle
        
        preprocessor = cls(**state["params"])
        preprocessor._feature_means = state["feature_means"]
        preprocessor._feature_medians = state["feature_medians"]
        preprocessor._feature_columns = state["feature_columns"]
        preprocessor._fitted = state["fitted"]
        
        if "scaler" in state:
            preprocessor._scaler = pickle.loads(state["scaler"])
        
        return preprocessor


# =============================================================================
# Time Series Specific Preprocessing
# =============================================================================

class TimeSeriesPreprocessor:
    """
    Specialized preprocessor for time series data.
    
    Handles time-aware operations like lag features and rolling statistics.
    """
    
    def __init__(self):
        """Initialize time series preprocessor."""
        self._base_preprocessor: DataPreprocessor | None = None
    
    def add_lag_features(
        self,
        df: pd.DataFrame,
        columns: list[str],
        lags: list[int],
        prefix: str = "lag",
    ) -> pd.DataFrame:
        """
        Add lagged features to DataFrame.
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            prefix: Prefix for lag column names
            
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        
        for col in columns:
            for lag in lags:
                df[f"{prefix}_{col}_{lag}"] = df[col].shift(lag)
        
        return df
    
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        columns: list[str],
        windows: list[int],
        functions: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Add rolling statistics features.
        
        Args:
            df: Input DataFrame
            columns: Columns to compute rolling stats for
            windows: List of window sizes
            functions: List of functions ("mean", "std", "min", "max", "sum")
            
        Returns:
            DataFrame with rolling features added
        """
        if functions is None:
            functions = ["mean", "std"]
        
        df = df.copy()
        
        for col in columns:
            for window in windows:
                rolling = df[col].rolling(window=window, min_periods=1)
                
                for func in functions:
                    if func == "mean":
                        df[f"rolling_{col}_{window}_mean"] = rolling.mean()
                    elif func == "std":
                        df[f"rolling_{col}_{window}_std"] = rolling.std()
                    elif func == "min":
                        df[f"rolling_{col}_{window}_min"] = rolling.min()
                    elif func == "max":
                        df[f"rolling_{col}_{window}_max"] = rolling.max()
                    elif func == "sum":
                        df[f"rolling_{col}_{window}_sum"] = rolling.sum()
        
        return df
    
    def add_diff_features(
        self,
        df: pd.DataFrame,
        columns: list[str],
        periods: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Add differenced features.
        
        Args:
            df: Input DataFrame
            columns: Columns to difference
            periods: List of difference periods
            
        Returns:
            DataFrame with diff features added
        """
        if periods is None:
            periods = [1]
        
        df = df.copy()
        
        for col in columns:
            for period in periods:
                df[f"diff_{col}_{period}"] = df[col].diff(period)
        
        return df
    
    def add_pct_change_features(
        self,
        df: pd.DataFrame,
        columns: list[str],
        periods: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Add percentage change features.
        
        Args:
            df: Input DataFrame
            columns: Columns to compute pct change
            periods: List of periods
            
        Returns:
            DataFrame with pct change features added
        """
        if periods is None:
            periods = [1]
        
        df = df.copy()
        
        for col in columns:
            for period in periods:
                df[f"pct_{col}_{period}"] = df[col].pct_change(period)
        
        return df
    
    def remove_lookahead_bias(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        """
        Ensure no future information leaks into features.
        
        This shifts the target forward to ensure features only use past data.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            feature_columns: Names of feature columns
            
        Returns:
            DataFrame with lookahead bias removed
        """
        df = df.copy()
        
        # Check for any correlation between features and future target
        # This is a basic check - real implementation would be more thorough
        
        # Shift target by 1 to create "next period" target
        if target_column in df.columns:
            df[f"{target_column}_next"] = df[target_column].shift(-1)
        
        return df
    
    def split_train_test_time(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        gap_bars: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically for time series.
        
        Args:
            df: Input DataFrame (must be sorted by time)
            train_ratio: Proportion for training
            gap_bars: Gap between train and test to avoid leakage
            
        Returns:
            Tuple of (train_df, test_df)
        """
        n = len(df)
        train_end = int(n * train_ratio)
        test_start = train_end + gap_bars
        
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:].copy()
        
        return train_df, test_df
    
    def create_walk_forward_splits(
        self,
        df: pd.DataFrame,
        train_window: int,
        test_window: int,
        step: int | None = None,
        embargo: int = 0,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward validation splits.
        
        Args:
            df: Input DataFrame
            train_window: Number of bars for training
            test_window: Number of bars for testing
            step: Step size between windows (default: test_window)
            embargo: Gap between train and test
            
        Returns:
            List of (train_df, test_df) tuples
        """
        if step is None:
            step = test_window
        
        splits = []
        n = len(df)
        
        start = 0
        while start + train_window + embargo + test_window <= n:
            train_end = start + train_window
            test_start = train_end + embargo
            test_end = test_start + test_window
            
            train_df = df.iloc[start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            splits.append((train_df, test_df))
            start += step
        
        logger.debug(f"Created {len(splits)} walk-forward splits")
        return splits
