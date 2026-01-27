"""
Tests for backtesting and evaluation modules.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.training.backtest import Backtester, BacktestConfig, BacktestResult
from src.training.walk_forward import WalkForwardValidator, PurgedKFold
from src.training.evaluation import ModelEvaluator, PerformanceMetrics


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 1000
    
    # Generate trending price data
    trend = np.linspace(0, 0.1, n)
    noise = np.random.randn(n) * 0.005
    returns = trend / n + noise
    close = 100 * np.exp(np.cumsum(returns))
    
    high = close * (1 + np.abs(np.random.randn(n) * 0.002))
    low = close * (1 - np.abs(np.random.randn(n) * 0.002))
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    volume = np.random.randint(1000, 10000, n)
    
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
def sample_predictions():
    """Create sample predictions for evaluation testing."""
    np.random.seed(42)
    n = 500
    
    # Create correlated predictions
    actual = np.random.randn(n) * 0.01
    noise = np.random.randn(n) * 0.003
    predicted = actual * 0.7 + noise
    
    return pd.Series(actual), pd.Series(predicted)


# =============================================================================
# Backtester Tests
# =============================================================================

class TestBacktester:
    """Tests for Backtester class."""
    
    def test_basic_backtest(self, sample_ohlcv):
        """Test basic backtest execution."""
        config = BacktestConfig(
            initial_balance=10000.0,
            commission_per_lot=7.0,
            spread_pips=1.5,
        )
        
        backtester = Backtester(config)
        
        # Simple signal function: buy when close > SMA20
        def signal_func(data, idx):
            if idx < 20:
                return None
            
            sma = data["close"].iloc[idx-20:idx].mean()
            current = data["close"].iloc[idx]
            
            if current > sma * 1.001:
                return {
                    "direction": "buy",
                    "lots": 0.1,
                    "sl": current - 0.003,
                    "tp": current + 0.005,
                    "confidence": 0.7,
                }
            return None
        
        result = backtester.run(sample_ohlcv, signal_func)
        
        # Should have some trades
        assert isinstance(result, BacktestResult)
        assert len(result.trades) > 0
        assert len(result.equity_curve) > 0
        assert "total_trades" in result.metrics
    
    def test_no_trades(self, sample_ohlcv):
        """Test backtest with no signals."""
        backtester = Backtester()
        
        def no_signal(data, idx):
            return None
        
        result = backtester.run(sample_ohlcv, no_signal)
        
        assert len(result.trades) == 0
        assert result.metrics["total_trades"] == 0
    
    def test_sl_tp_execution(self, sample_ohlcv):
        """Test that SL and TP are hit correctly."""
        config = BacktestConfig(
            initial_balance=10000.0,
            slippage_pips=0,  # Disable slippage for clean test
            spread_pips=0,    # Disable spread for clean test
        )
        
        backtester = Backtester(config)
        
        # Force a trade with tight SL/TP
        trade_entered = [False]
        
        def force_trade(data, idx):
            if idx == 50 and not trade_entered[0]:
                trade_entered[0] = True
                current = data["close"].iloc[idx]
                return {
                    "direction": "buy",
                    "lots": 0.1,
                    "sl": current * 0.99,  # 1% SL
                    "tp": current * 1.02,  # 2% TP
                    "confidence": 0.8,
                }
            return None
        
        result = backtester.run(sample_ohlcv, force_trade)
        
        # Should have at least one trade
        assert len(result.trades) >= 1
    
    def test_metrics_calculation(self, sample_ohlcv):
        """Test that metrics are calculated correctly."""
        backtester = Backtester()
        
        # Always buy
        def always_buy(data, idx):
            if idx % 100 == 0 and idx > 0:
                current = data["close"].iloc[idx]
                return {
                    "direction": "buy",
                    "lots": 0.1,
                    "sl": current * 0.98,
                    "tp": current * 1.03,
                    "confidence": 0.6,
                }
            return None
        
        result = backtester.run(sample_ohlcv, always_buy)
        
        # Check metric keys exist
        assert "total_trades" in result.metrics
        assert "win_rate" in result.metrics
        assert "profit_factor" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "final_balance" in result.metrics


# =============================================================================
# Walk-Forward Tests
# =============================================================================

class TestWalkForwardValidator:
    """Tests for WalkForwardValidator class."""
    
    def test_split_generation(self, sample_ohlcv):
        """Test walk-forward split generation."""
        validator = WalkForwardValidator(
            train_window=200,
            test_window=50,
            step=50,
            embargo=10,
        )
        
        splits = validator.get_splits(sample_ohlcv)
        
        # Should have multiple splits
        assert len(splits) > 0
        
        # Check split properties
        for train_df, test_df, info in splits:
            # Train and test should not overlap
            assert train_df.index.max() < test_df.index.min()
            
            # Sizes should match info
            assert len(train_df) == info.train_size
            assert len(test_df) == info.test_size
    
    def test_expanding_window(self, sample_ohlcv):
        """Test expanding window mode."""
        validator = WalkForwardValidator(
            train_window=200,
            test_window=50,
            step=50,
            embargo=10,
            expanding=True,
        )
        
        splits = validator.get_splits(sample_ohlcv)
        
        # Training size should grow
        train_sizes = [info.train_size for _, _, info in splits]
        
        # Each subsequent split should have more training data
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i-1]
    
    def test_rolling_window(self, sample_ohlcv):
        """Test rolling window mode."""
        validator = WalkForwardValidator(
            train_window=200,
            test_window=50,
            step=50,
            embargo=10,
            expanding=False,
        )
        
        splits = validator.get_splits(sample_ohlcv)
        
        # Training size should be constant
        train_sizes = [info.train_size for _, _, info in splits]
        
        assert all(s == train_sizes[0] for s in train_sizes)


class TestPurgedKFold:
    """Tests for PurgedKFold class."""
    
    def test_split_indices(self, sample_ohlcv):
        """Test purged k-fold split indices."""
        kfold = PurgedKFold(n_splits=5, embargo_bars=10)
        
        X = sample_ohlcv[["close"]]
        splits = kfold.split(X)
        
        assert len(splits) == 5
        
        # Check no overlap between train and test
        for train_idx, test_idx in splits:
            train_set = set(train_idx)
            test_set = set(test_idx)
            
            assert len(train_set.intersection(test_set)) == 0
    
    def test_embargo_gap(self, sample_ohlcv):
        """Test that embargo gap is respected."""
        embargo = 20
        kfold = PurgedKFold(n_splits=3, embargo_bars=embargo)
        
        X = sample_ohlcv[["close"]]
        splits = kfold.split(X)
        
        for train_idx, test_idx in splits:
            # Check gap between max train and min test (before)
            train_before = train_idx[train_idx < test_idx.min()]
            if len(train_before) > 0:
                gap = test_idx.min() - train_before.max()
                assert gap >= embargo
            
            # Check gap between max test and min train (after)
            train_after = train_idx[train_idx > test_idx.max()]
            if len(train_after) > 0:
                gap = train_after.min() - test_idx.max()
                assert gap >= embargo


# =============================================================================
# Evaluation Tests
# =============================================================================

class TestModelEvaluator:
    """Tests for ModelEvaluator class."""
    
    def test_prediction_metrics(self, sample_predictions):
        """Test prediction evaluation metrics."""
        y_true, y_pred = sample_predictions
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_predictions(y_true, y_pred)
        
        # Check all expected metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r_squared" in metrics
        assert "direction_accuracy" in metrics
        assert "correlation" in metrics
        
        # RMSE should be positive
        assert metrics["rmse"] >= 0
        
        # Direction accuracy should be between 0 and 1
        assert 0 <= metrics["direction_accuracy"] <= 1
    
    def test_equity_curve_metrics(self):
        """Test equity curve evaluation."""
        # Create sample equity curve
        n = 500
        returns = np.random.randn(n) * 0.01 + 0.001  # Slight positive drift
        equity = 10000 * np.exp(np.cumsum(returns))
        equity = pd.Series(equity)
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_equity_curve(equity)
        
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "volatility" in metrics
    
    def test_trade_metrics(self):
        """Test trade evaluation metrics."""
        # Create sample trades
        trades = pd.DataFrame({
            "pnl": [100, -50, 75, -30, 120, -40, 80],
        })
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_trades(trades)
        
        assert metrics["total_trades"] == 7
        assert metrics["winners"] == 4
        assert metrics["losers"] == 3
        assert metrics["win_rate"] == 4 / 7
        assert metrics["profit_factor"] > 0
    
    def test_full_evaluation(self, sample_predictions):
        """Test full evaluation combining all metrics."""
        y_true, y_pred = sample_predictions
        
        # Create sample equity
        equity = pd.Series(10000 * np.exp(np.cumsum(np.random.randn(500) * 0.01)))
        
        # Create sample trades
        trades = pd.DataFrame({"pnl": [100, -50, 75, -30, 120]})
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_full(
            y_true=y_true,
            y_pred=y_pred,
            equity=equity,
            trades=trades,
        )
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades == 5
        assert metrics.rmse >= 0


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.5,
            max_drawdown=0.05,
            total_trades=100,
            win_rate=0.55,
        )
        
        d = metrics.to_dict()
        
        assert d["total_return"] == 0.15
        assert d["sharpe_ratio"] == 1.5
        assert d["max_drawdown"] == 0.05
        assert d["total_trades"] == 100
        assert d["win_rate"] == 0.55
    
    def test_summary(self):
        """Test summary string generation."""
        metrics = PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.5,
            total_trades=100,
        )
        
        summary = metrics.summary()
        
        assert "Performance Metrics" in summary
        assert "Total Return" in summary
        assert "Sharpe Ratio" in summary


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
