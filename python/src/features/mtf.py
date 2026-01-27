"""
Multi-timeframe feature engineering module.

Provides features from multiple timeframes for enhanced prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from .technical import TechnicalIndicators

logger = get_logger("features.mtf")


# =============================================================================
# Timeframe Utilities
# =============================================================================

TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
    "MN1": 43200,
}


def get_timeframe_minutes(timeframe: str) -> int:
    """Get minutes for a timeframe string."""
    return TIMEFRAME_MINUTES.get(timeframe.upper(), 60)


def resample_ohlcv(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe.
    
    Args:
        df: DataFrame with OHLCV data (datetime index)
        target_timeframe: Target timeframe string
        
    Returns:
        Resampled DataFrame
    """
    minutes = get_timeframe_minutes(target_timeframe)
    rule = f"{minutes}min"
    
    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    
    return resampled


# =============================================================================
# Multi-Timeframe Features
# =============================================================================

class MultiTimeframeFeatures:
    """
    Multi-timeframe feature calculator.
    
    Computes features from multiple timeframes and aligns them
    to the base timeframe without lookahead bias.
    """
    
    def __init__(
        self,
        base_timeframe: str = "H1",
        higher_timeframes: list[str] | None = None,
    ):
        """
        Initialize MTF feature calculator.
        
        Args:
            base_timeframe: Base timeframe for trading decisions
            higher_timeframes: List of higher timeframes for context
        """
        self.base_timeframe = base_timeframe
        self.higher_timeframes = higher_timeframes or ["H4", "D1"]
        
        # Validate timeframes
        base_minutes = get_timeframe_minutes(base_timeframe)
        for tf in self.higher_timeframes:
            tf_minutes = get_timeframe_minutes(tf)
            if tf_minutes <= base_minutes:
                raise ValueError(f"Higher timeframe {tf} must be larger than base {base_timeframe}")
    
    def compute_htf_features(
        self,
        htf_df: pd.DataFrame,
        timeframe: str,
        features: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compute features from a higher timeframe.
        
        Args:
            htf_df: Higher timeframe OHLCV data
            timeframe: Timeframe string (for naming)
            features: List of features to compute
            
        Returns:
            DataFrame with HTF features
        """
        if features is None:
            features = ["ema_slope", "atr", "trend_direction", "range_position", "rsi"]
        
        ti = TechnicalIndicators(htf_df)
        result = pd.DataFrame(index=htf_df.index)
        tf_prefix = timeframe.lower()
        
        for feature in features:
            if feature == "ema_slope":
                result[f"{tf_prefix}_ema_slope_20"] = ti.ema_slope(20)
            elif feature == "atr":
                result[f"{tf_prefix}_atr"] = ti.atr(14)
            elif feature == "trend_direction":
                ema_fast = ti.ema(10)
                ema_slow = ti.ema(20)
                result[f"{tf_prefix}_trend"] = (ema_fast > ema_slow).astype(int) * 2 - 1
            elif feature == "range_position":
                result[f"{tf_prefix}_range_pos"] = ti.range_position(20)
            elif feature == "rsi":
                result[f"{tf_prefix}_rsi"] = ti.rsi(14)
            elif feature == "adx":
                result[f"{tf_prefix}_adx"] = ti.adx(14)
            elif feature == "bb_width":
                result[f"{tf_prefix}_bb_width"] = ti.bollinger_width(20)
            elif feature == "log_return":
                result[f"{tf_prefix}_return_1"] = ti.log_return(1)
                result[f"{tf_prefix}_return_5"] = ti.log_return(5)
        
        return result
    
    def align_htf_to_base(
        self,
        base_df: pd.DataFrame,
        htf_features: pd.DataFrame,
        htf_timeframe: str,
    ) -> pd.DataFrame:
        """
        Align higher timeframe features to base timeframe.
        
        IMPORTANT: Uses only completed HTF bars to avoid lookahead bias.
        Each base bar gets features from the last COMPLETED higher timeframe bar.
        
        Args:
            base_df: Base timeframe DataFrame
            htf_features: Higher timeframe features
            htf_timeframe: Higher timeframe string
            
        Returns:
            HTF features aligned to base timeframe
        """
        htf_minutes = get_timeframe_minutes(htf_timeframe)
        
        # Forward-fill HTF features to base timeframe
        # This ensures we only use information available at the time
        aligned = htf_features.reindex(base_df.index, method="ffill")
        
        # Shift by 1 to use only COMPLETED bars (avoid lookahead)
        # When a new HTF bar starts, we shouldn't use its data until it closes
        aligned = aligned.shift(1)
        
        return aligned
    
    def compute_all_mtf_features(
        self,
        data: dict[str, pd.DataFrame],
        features: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compute all multi-timeframe features.
        
        Args:
            data: Dict mapping timeframe to OHLCV DataFrame
                  Must include base_timeframe and all higher_timeframes
            features: Features to compute for each HTF
            
        Returns:
            DataFrame with base timeframe data + all HTF features
        """
        if self.base_timeframe not in data:
            raise ValueError(f"Base timeframe {self.base_timeframe} not in data")
        
        base_df = data[self.base_timeframe].copy()
        
        for htf in self.higher_timeframes:
            if htf not in data:
                logger.warning(f"Higher timeframe {htf} not in data, skipping")
                continue
            
            htf_df = data[htf]
            
            # Compute HTF features
            htf_features = self.compute_htf_features(htf_df, htf, features)
            
            # Align to base timeframe
            aligned_features = self.align_htf_to_base(base_df, htf_features, htf)
            
            # Merge with base
            for col in aligned_features.columns:
                base_df[col] = aligned_features[col]
        
        logger.debug(f"Computed MTF features from {len(self.higher_timeframes)} higher timeframes")
        return base_df
    
    def compute_regime_features(
        self,
        data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Compute regime detection features from higher timeframes.
        
        Regimes:
        - Trending (strong directional movement)
        - Ranging (consolidation)
        - High Volatility (increased uncertainty)
        
        Args:
            data: Dict mapping timeframe to OHLCV DataFrame
            
        Returns:
            DataFrame with regime features
        """
        base_df = data[self.base_timeframe].copy()
        
        for htf in self.higher_timeframes:
            if htf not in data:
                continue
            
            htf_df = data[htf]
            ti = TechnicalIndicators(htf_df)
            tf_prefix = htf.lower()
            
            # ADX for trend strength
            adx = ti.adx(14)
            
            # Bollinger width for volatility
            bb_width = ti.bollinger_width(20)
            
            # Create regime features
            htf_features = pd.DataFrame(index=htf_df.index)
            
            # Trend regime: ADX > 25
            htf_features[f"{tf_prefix}_is_trending"] = (adx > 25).astype(int)
            
            # High volatility: BB width > 1.5x median
            median_width = bb_width.rolling(window=50, min_periods=20).median()
            htf_features[f"{tf_prefix}_high_vol"] = (bb_width > 1.5 * median_width).astype(int)
            
            # Ranging: Not trending and not high vol
            htf_features[f"{tf_prefix}_is_ranging"] = (
                (htf_features[f"{tf_prefix}_is_trending"] == 0) &
                (htf_features[f"{tf_prefix}_high_vol"] == 0)
            ).astype(int)
            
            # Align to base
            aligned = self.align_htf_to_base(base_df, htf_features, htf)
            for col in aligned.columns:
                base_df[col] = aligned[col]
        
        return base_df


# =============================================================================
# Feature Engine (Main Interface)
# =============================================================================

class FeatureEngine:
    """
    Main feature engineering engine.
    
    Combines technical indicators, multi-timeframe features,
    and temporal features into a unified feature set.
    """
    
    def __init__(
        self,
        base_timeframe: str = "H1",
        higher_timeframes: list[str] | None = None,
    ):
        """
        Initialize feature engine.
        
        Args:
            base_timeframe: Base timeframe for trading
            higher_timeframes: Higher timeframes for MTF features
        """
        self.base_timeframe = base_timeframe
        self.higher_timeframes = higher_timeframes or ["H4", "D1"]
        self.mtf = MultiTimeframeFeatures(base_timeframe, higher_timeframes)
        self._feature_names: list[str] = []
    
    def compute_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute base technical indicator features.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with technical features
        """
        ti = TechnicalIndicators(df)
        return ti.compute_all()
    
    def compute_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute time-based features.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with temporal features
        """
        result = df.copy()
        
        # Hour of day (0-23)
        result["hour"] = df.index.hour
        
        # Day of week (0=Monday, 6=Sunday)
        result["day_of_week"] = df.index.dayofweek
        
        # Trading sessions
        hour = df.index.hour
        result["is_tokyo"] = ((hour >= 0) & (hour < 9)).astype(int)
        result["is_london"] = ((hour >= 8) & (hour < 17)).astype(int)
        result["is_newyork"] = ((hour >= 13) & (hour < 22)).astype(int)
        result["is_overlap"] = (result["is_london"] & result["is_newyork"]).astype(int)
        
        # Encode cyclical features
        result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        result["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        result["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        return result
    
    def compute_cost_features(
        self,
        df: pd.DataFrame,
        spread_data: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Compute cost-related features.
        
        Args:
            df: OHLCV DataFrame
            spread_data: Spread data (optional)
            
        Returns:
            DataFrame with cost features
        """
        result = df.copy()
        
        # If spread data provided, compute spread features
        if spread_data is not None:
            result["spread"] = spread_data
            result["spread_ma_20"] = spread_data.rolling(window=20, min_periods=1).mean()
            result["spread_ratio"] = spread_data / result["atr"] if "atr" in result.columns else spread_data
        
        # Tick volume features
        if "volume" in df.columns:
            result["volume_ma_20"] = df["volume"].rolling(window=20, min_periods=1).mean()
            result["volume_ratio"] = df["volume"] / result["volume_ma_20"]
        
        return result
    
    def compute_target(
        self,
        df: pd.DataFrame,
        horizon: int = 6,
        target_type: str = "return",
    ) -> pd.DataFrame:
        """
        Compute prediction target.
        
        Args:
            df: DataFrame with close prices
            horizon: Prediction horizon (bars ahead)
            target_type: Type of target ("return", "direction")
            
        Returns:
            DataFrame with target column
        """
        result = df.copy()
        
        # Future return
        future_close = df["close"].shift(-horizon)
        future_return = np.log(future_close / df["close"])
        
        if target_type == "return":
            result[f"target_{horizon}"] = future_return
        elif target_type == "direction":
            result[f"target_{horizon}"] = (future_return > 0).astype(int)
        
        return result
    
    def compute_all_features(
        self,
        data: dict[str, pd.DataFrame],
        include_target: bool = True,
        target_horizons: list[int] | None = None,
        spread_data: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Compute all features for ML model.
        
        Args:
            data: Dict mapping timeframe to OHLCV DataFrame
            include_target: Whether to include target variable
            target_horizons: List of prediction horizons
            spread_data: Optional spread data
            
        Returns:
            DataFrame with all features
        """
        if target_horizons is None:
            target_horizons = [3, 6, 12]
        
        base_df = data[self.base_timeframe].copy()
        
        # 1. Base technical features
        result = self.compute_base_features(base_df)
        
        # 2. Multi-timeframe features
        result = self.mtf.compute_all_mtf_features(
            {self.base_timeframe: result, **{tf: data[tf] for tf in self.higher_timeframes if tf in data}}
        )
        
        # 3. Regime features
        regime_df = self.mtf.compute_regime_features(data)
        for col in regime_df.columns:
            if col not in result.columns:
                result[col] = regime_df[col]
        
        # 4. Temporal features
        result = self.compute_temporal_features(result)
        
        # 5. Cost features
        result = self.compute_cost_features(result, spread_data)
        
        # 6. Target (if requested)
        if include_target:
            for horizon in target_horizons:
                result = self.compute_target(result, horizon, "return")
        
        # Store feature names (excluding targets and OHLCV)
        exclude_cols = {"open", "high", "low", "close", "volume", "spread"}
        exclude_cols.update({f"target_{h}" for h in target_horizons})
        self._feature_names = [c for c in result.columns if c not in exclude_cols]
        
        logger.info(f"Computed {len(self._feature_names)} features")
        return result
    
    @property
    def feature_names(self) -> list[str]:
        """Get list of feature names (excluding targets)."""
        return self._feature_names
    
    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        dropna: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """
        Get feature matrix and target for ML.
        
        Args:
            df: DataFrame with all features computed
            dropna: Whether to drop rows with NaN
            
        Returns:
            Tuple of (features DataFrame, target Series or None)
        """
        # Get feature columns
        feature_cols = [c for c in self._feature_names if c in df.columns]
        
        # Get target column (use primary horizon)
        target_col = None
        for col in df.columns:
            if col.startswith("target_"):
                target_col = col
                break
        
        X = df[feature_cols].copy()
        y = df[target_col].copy() if target_col else None
        
        if dropna:
            valid_idx = X.dropna().index
            if y is not None:
                valid_idx = valid_idx.intersection(y.dropna().index)
            X = X.loc[valid_idx]
            if y is not None:
                y = y.loc[valid_idx]
        
        return X, y
