"""
Tests for feature engineering modules.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.features.technical import TechnicalIndicators
from src.features.mtf import MultiTimeframeFeatures, FeatureEngine


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    
    # Generate random walk prices
    returns = np.random.randn(n) * 0.001
    close = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    high = close * (1 + np.abs(np.random.randn(n) * 0.002))
    low = close * (1 - np.abs(np.random.randn(n) * 0.002))
    open_price = close.copy()
    open_price[1:] = close[:-1]
    volume = np.random.randint(1000, 10000, n)
    
    # Create DataFrame
    dates = pd.date_range(start="2024-01-01", periods=n, freq="h", tz="UTC")
    
    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)
    
    return df


@pytest.fixture
def mtf_data(sample_ohlcv):
    """Create multi-timeframe data."""
    h1_data = sample_ohlcv.copy()
    
    # Create H4 data by resampling
    h4_data = h1_data.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    
    # Create D1 data
    d1_data = h1_data.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    
    return {
        "H1": h1_data,
        "H4": h4_data,
        "D1": d1_data,
    }


# =============================================================================
# Technical Indicators Tests
# =============================================================================

class TestTechnicalIndicators:
    """Tests for TechnicalIndicators class."""
    
    def test_sma(self, sample_ohlcv):
        """Test Simple Moving Average calculation."""
        ti = TechnicalIndicators(sample_ohlcv)
        sma = ti.sma(20)
        
        # SMA should have NaN for first 19 values
        assert sma.isna().sum() == 19
        
        # Check manual calculation for last value
        expected = sample_ohlcv["close"].iloc[-20:].mean()
        assert abs(sma.iloc[-1] - expected) < 1e-10
    
    def test_ema(self, sample_ohlcv):
        """Test Exponential Moving Average calculation."""
        ti = TechnicalIndicators(sample_ohlcv)
        ema = ti.ema(20)
        
        # EMA should have fewer NaN than SMA of same period
        assert ema.isna().sum() == 19
        
        # EMA should be closer to recent prices
        close = sample_ohlcv["close"]
        sma = ti.sma(20)
        
        # If price is trending up, EMA should be above SMA
        # If trending down, EMA should be below SMA
        # Just verify they're different
        assert not ema.equals(sma)
    
    def test_rsi(self, sample_ohlcv):
        """Test RSI calculation."""
        ti = TechnicalIndicators(sample_ohlcv)
        rsi = ti.rsi(14)
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd(self, sample_ohlcv):
        """Test MACD calculation."""
        ti = TechnicalIndicators(sample_ohlcv)
        macd_line, signal, hist = ti.macd()
        
        # Histogram should be MACD - Signal
        expected_hist = macd_line - signal
        np.testing.assert_array_almost_equal(
            hist.dropna().values,
            expected_hist.dropna().values,
            decimal=10
        )
    
    def test_atr(self, sample_ohlcv):
        """Test ATR calculation."""
        ti = TechnicalIndicators(sample_ohlcv)
        atr = ti.atr(14)
        
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()
    
    def test_bollinger_bands(self, sample_ohlcv):
        """Test Bollinger Bands calculation."""
        ti = TechnicalIndicators(sample_ohlcv)
        upper, middle, lower = ti.bollinger_bands(20, 2.0)
        
        # Upper should be above middle, middle above lower
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()
    
    def test_adx(self, sample_ohlcv):
        """Test ADX calculation."""
        ti = TechnicalIndicators(sample_ohlcv)
        adx = ti.adx(14)
        
        # ADX should be between 0 and 100
        valid_adx = adx.dropna()
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()
    
    def test_stochastic(self, sample_ohlcv):
        """Test Stochastic Oscillator calculation."""
        ti = TechnicalIndicators(sample_ohlcv)
        stoch_k, stoch_d = ti.stochastic()
        
        # Should be between 0 and 100
        valid_k = stoch_k.dropna()
        valid_d = stoch_d.dropna()
        
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()
        assert (valid_d >= 0).all()
        assert (valid_d <= 100).all()
    
    def test_compute_all(self, sample_ohlcv):
        """Test computing all indicators at once."""
        ti = TechnicalIndicators(sample_ohlcv)
        result = ti.compute_all()
        
        # Should have original columns plus indicators
        assert len(result.columns) > len(sample_ohlcv.columns)
        
        # Check some expected columns exist
        expected_cols = ["ema_10", "rsi", "macd", "atr", "adx"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"


# =============================================================================
# Multi-Timeframe Features Tests
# =============================================================================

class TestMultiTimeframeFeatures:
    """Tests for MultiTimeframeFeatures class."""
    
    def test_htf_alignment(self, mtf_data):
        """Test higher timeframe alignment to base."""
        mtf = MultiTimeframeFeatures("H1", ["H4", "D1"])
        
        h1_df = mtf_data["H1"]
        h4_df = mtf_data["H4"]
        
        # Compute H4 features
        h4_features = mtf.compute_htf_features(h4_df, "H4")
        
        # Align to H1
        aligned = mtf.align_htf_to_base(h1_df, h4_features, "H4")
        
        # Should have same index as H1
        assert len(aligned) == len(h1_df)
        
        # First few values should be NaN (shifted by 1)
        assert aligned.iloc[0].isna().all()
    
    def test_no_lookahead_bias(self, mtf_data):
        """Test that MTF features don't leak future information."""
        mtf = MultiTimeframeFeatures("H1", ["H4"])
        
        result = mtf.compute_all_mtf_features(mtf_data)
        
        # H4 features at any H1 bar should only use H4 bars that
        # completed BEFORE that H1 bar
        # The shift(1) in align_htf_to_base ensures this
        
        # Get an H1 timestamp
        h1_idx = 10  # Some middle index
        h1_time = result.index[h1_idx]
        
        # Get the H4 feature value at this time
        h4_col = [c for c in result.columns if c.startswith("h4_")][0]
        h4_value = result[h4_col].iloc[h1_idx]
        
        # This value should be from an H4 bar that closed before h1_time
        # Due to our shift(1), we're using completed bars only
        # This is hard to verify directly, but we can check values are consistent
        if not pd.isna(h4_value):
            # The value should be forward-filled from previous H4 bar
            assert h4_value == result[h4_col].iloc[h1_idx - 1] or pd.isna(result[h4_col].iloc[h1_idx - 1])
    
    def test_regime_features(self, mtf_data):
        """Test regime detection features."""
        mtf = MultiTimeframeFeatures("H1", ["H4"])
        
        result = mtf.compute_regime_features(mtf_data)
        
        # Should have regime columns
        regime_cols = [c for c in result.columns if "trending" in c or "ranging" in c]
        assert len(regime_cols) > 0
        
        # Regime values should be 0 or 1
        for col in regime_cols:
            valid = result[col].dropna()
            assert set(valid.unique()).issubset({0, 1})


# =============================================================================
# Feature Engine Tests
# =============================================================================

class TestFeatureEngine:
    """Tests for FeatureEngine class."""
    
    def test_compute_all_features(self, mtf_data):
        """Test computing all features."""
        engine = FeatureEngine("H1", ["H4", "D1"])
        
        result = engine.compute_all_features(mtf_data)
        
        # Should have many features
        assert len(result.columns) > 30
        
        # Should have target columns
        target_cols = [c for c in result.columns if c.startswith("target_")]
        assert len(target_cols) > 0
    
    def test_temporal_features(self, sample_ohlcv):
        """Test temporal feature computation."""
        engine = FeatureEngine("H1")
        
        result = engine.compute_temporal_features(sample_ohlcv)
        
        # Check temporal columns
        assert "hour" in result.columns
        assert "day_of_week" in result.columns
        assert "is_london" in result.columns
        assert "is_newyork" in result.columns
        
        # Hour should be 0-23
        assert result["hour"].min() >= 0
        assert result["hour"].max() <= 23
    
    def test_feature_matrix_extraction(self, mtf_data):
        """Test getting feature matrix for ML."""
        engine = FeatureEngine("H1", ["H4"])
        
        result = engine.compute_all_features(mtf_data)
        X, y = engine.get_feature_matrix(result)
        
        # X should not have target columns
        assert not any("target" in c for c in X.columns)
        
        # X and y should have same index
        if y is not None:
            assert len(X) == len(y)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
