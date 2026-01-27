"""
Model evaluation module.

Provides metrics and visualization for trading model performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger("training.evaluation")


# =============================================================================
# Performance Metrics
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Trading performance metrics."""
    # Return metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    drawdown_duration: int = 0  # bars
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0
    
    # Prediction metrics
    direction_accuracy: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r_squared: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "monthly_return": self.monthly_return,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "avg_drawdown": self.avg_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "direction_accuracy": self.direction_accuracy,
            "rmse": self.rmse,
            "mae": self.mae,
        }
    
    def summary(self) -> str:
        """Get summary string."""
        return (
            f"Performance Metrics\n"
            f"{'='*40}\n"
            f"Returns:\n"
            f"  Total Return: {self.total_return:.2%}\n"
            f"  Annual Return: {self.annual_return:.2%}\n"
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"\nRisk:\n"
            f"  Volatility: {self.volatility:.2%}\n"
            f"  Max Drawdown: {self.max_drawdown:.2%}\n"
            f"  Calmar Ratio: {self.calmar_ratio:.2f}\n"
            f"\nTrading:\n"
            f"  Total Trades: {self.total_trades}\n"
            f"  Win Rate: {self.win_rate:.2%}\n"
            f"  Profit Factor: {self.profit_factor:.2f}\n"
            f"\nPrediction:\n"
            f"  Direction Accuracy: {self.direction_accuracy:.2%}\n"
            f"  RMSE: {self.rmse:.6f}\n"
        )


# =============================================================================
# Model Evaluator
# =============================================================================

class ModelEvaluator:
    """
    Comprehensive model evaluation.
    
    Evaluates both prediction quality and simulated trading performance.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,  # 2% annual risk-free rate
        periods_per_year: int = 252 * 24,  # Hourly data
    ):
        """
        Initialize evaluator.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            periods_per_year: Number of trading periods per year
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
    
    def evaluate_predictions(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluate prediction quality.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of prediction metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Remove NaN
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        
        if len(y_true) == 0:
            return {
                "rmse": 0.0,
                "mae": 0.0,
                "mse": 0.0,
                "r_squared": 0.0,
                "direction_accuracy": 0.0,
                "correlation": 0.0,
            }
        
        # Regression metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Direction accuracy
        true_direction = (y_true > 0).astype(int)
        pred_direction = (y_pred > 0).astype(int)
        direction_accuracy = np.mean(true_direction == pred_direction)
        
        # Correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
        
        return {
            "rmse": rmse,
            "mae": mae,
            "mse": mse,
            "r_squared": r_squared,
            "direction_accuracy": direction_accuracy,
            "correlation": correlation,
            "n_samples": len(y_true),
        }
    
    def evaluate_equity_curve(
        self,
        equity: pd.Series,
    ) -> dict[str, float]:
        """
        Evaluate equity curve performance.
        
        Args:
            equity: Equity curve series
            
        Returns:
            Dictionary of performance metrics
        """
        if len(equity) < 2:
            return {}
        
        # Returns
        returns = equity.pct_change().dropna()
        
        # Total return
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        
        # Annualized return
        n_periods = len(equity)
        annual_return = (1 + total_return) ** (self.periods_per_year / n_periods) - 1
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(self.periods_per_year)
        
        # Sharpe ratio
        excess_return = annual_return - self.risk_free_rate
        sharpe = excess_return / volatility if volatility > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(self.periods_per_year)
        sortino = excess_return / downside_std if downside_std > 0 else 0.0
        
        # Drawdown analysis
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        avg_drawdown = abs(drawdown.mean())
        
        # Calmar ratio
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Drawdown duration
        underwater = drawdown < 0
        if underwater.any():
            # Find longest drawdown period
            dd_groups = (~underwater).cumsum()
            dd_durations = underwater.groupby(dd_groups).sum()
            max_dd_duration = dd_durations.max()
        else:
            max_dd_duration = 0
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "monthly_return": annual_return / 12,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown": max_drawdown,
            "avg_drawdown": avg_drawdown,
            "max_dd_duration": max_dd_duration,
            "n_periods": n_periods,
        }
    
    def evaluate_trades(
        self,
        trades: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Evaluate trading performance.
        
        Args:
            trades: DataFrame with trade records (must have 'pnl' column)
            
        Returns:
            Dictionary of trading metrics
        """
        if trades.empty or "pnl" not in trades.columns:
            return {}
        
        pnl = trades["pnl"]
        
        winners = pnl[pnl > 0]
        losers = pnl[pnl <= 0]
        
        total_profit = winners.sum() if len(winners) > 0 else 0.0
        total_loss = abs(losers.sum()) if len(losers) > 0 else 0.0
        
        return {
            "total_trades": len(pnl),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(pnl) if len(pnl) > 0 else 0.0,
            "profit_factor": total_profit / total_loss if total_loss > 0 else float("inf"),
            "total_pnl": pnl.sum(),
            "avg_pnl": pnl.mean(),
            "avg_win": winners.mean() if len(winners) > 0 else 0.0,
            "avg_loss": losers.mean() if len(losers) > 0 else 0.0,
            "largest_win": winners.max() if len(winners) > 0 else 0.0,
            "largest_loss": losers.min() if len(losers) > 0 else 0.0,
            "std_pnl": pnl.std(),
            "expectancy": pnl.mean() if len(pnl) > 0 else 0.0,
        }
    
    def evaluate_full(
        self,
        y_true: pd.Series | np.ndarray | None = None,
        y_pred: pd.Series | np.ndarray | None = None,
        equity: pd.Series | None = None,
        trades: pd.DataFrame | None = None,
    ) -> PerformanceMetrics:
        """
        Full evaluation combining all metrics.
        
        Args:
            y_true: Actual target values
            y_pred: Predicted values
            equity: Equity curve
            trades: Trade records
            
        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()
        
        # Prediction metrics
        if y_true is not None and y_pred is not None:
            pred_metrics = self.evaluate_predictions(y_true, y_pred)
            metrics.rmse = pred_metrics.get("rmse", 0.0)
            metrics.mae = pred_metrics.get("mae", 0.0)
            metrics.r_squared = pred_metrics.get("r_squared", 0.0)
            metrics.direction_accuracy = pred_metrics.get("direction_accuracy", 0.0)
        
        # Equity curve metrics
        if equity is not None:
            eq_metrics = self.evaluate_equity_curve(equity)
            metrics.total_return = eq_metrics.get("total_return", 0.0)
            metrics.annual_return = eq_metrics.get("annual_return", 0.0)
            metrics.monthly_return = eq_metrics.get("monthly_return", 0.0)
            metrics.volatility = eq_metrics.get("volatility", 0.0)
            metrics.max_drawdown = eq_metrics.get("max_drawdown", 0.0)
            metrics.avg_drawdown = eq_metrics.get("avg_drawdown", 0.0)
            metrics.sharpe_ratio = eq_metrics.get("sharpe_ratio", 0.0)
            metrics.sortino_ratio = eq_metrics.get("sortino_ratio", 0.0)
            metrics.calmar_ratio = eq_metrics.get("calmar_ratio", 0.0)
        
        # Trading metrics
        if trades is not None:
            trade_metrics = self.evaluate_trades(trades)
            metrics.total_trades = trade_metrics.get("total_trades", 0)
            metrics.win_rate = trade_metrics.get("win_rate", 0.0)
            metrics.profit_factor = trade_metrics.get("profit_factor", 0.0)
            metrics.avg_win = trade_metrics.get("avg_win", 0.0)
            metrics.avg_loss = trade_metrics.get("avg_loss", 0.0)
            metrics.largest_win = trade_metrics.get("largest_win", 0.0)
            metrics.largest_loss = trade_metrics.get("largest_loss", 0.0)
        
        return metrics
    
    def compare_models(
        self,
        results: dict[str, PerformanceMetrics],
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            results: Dictionary mapping model name to metrics
            
        Returns:
            DataFrame with comparison
        """
        comparison = []
        for name, metrics in results.items():
            row = metrics.to_dict()
            row["model"] = name
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        df = df.set_index("model")
        
        return df


# =============================================================================
# Benchmark Comparison
# =============================================================================

class BenchmarkComparison:
    """Compare strategy against benchmarks."""
    
    def __init__(self):
        """Initialize benchmark comparison."""
        self.evaluator = ModelEvaluator()
    
    def compare_to_buy_hold(
        self,
        strategy_equity: pd.Series,
        price_data: pd.Series,
        initial_capital: float = 10000.0,
    ) -> dict[str, dict[str, float]]:
        """
        Compare strategy to buy-and-hold.
        
        Args:
            strategy_equity: Strategy equity curve
            price_data: Price series (close prices)
            initial_capital: Initial capital for buy-and-hold
            
        Returns:
            Dictionary with strategy and benchmark metrics
        """
        # Calculate buy-and-hold equity
        returns = price_data.pct_change().fillna(0)
        bh_equity = initial_capital * (1 + returns).cumprod()
        
        # Align series
        common_idx = strategy_equity.index.intersection(bh_equity.index)
        strategy_equity = strategy_equity.loc[common_idx]
        bh_equity = bh_equity.loc[common_idx]
        
        # Evaluate both
        strategy_metrics = self.evaluator.evaluate_equity_curve(strategy_equity)
        bh_metrics = self.evaluator.evaluate_equity_curve(bh_equity)
        
        # Calculate alpha and beta
        strategy_returns = strategy_equity.pct_change().dropna()
        bh_returns = bh_equity.pct_change().dropna()
        
        if len(strategy_returns) > 1 and len(bh_returns) > 1:
            cov_matrix = np.cov(strategy_returns, bh_returns)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0.0
            alpha = strategy_metrics.get("annual_return", 0) - beta * bh_metrics.get("annual_return", 0)
        else:
            alpha = 0.0
            beta = 0.0
        
        return {
            "strategy": strategy_metrics,
            "buy_and_hold": bh_metrics,
            "comparison": {
                "alpha": alpha,
                "beta": beta,
                "return_diff": strategy_metrics.get("total_return", 0) - bh_metrics.get("total_return", 0),
                "sharpe_diff": strategy_metrics.get("sharpe_ratio", 0) - bh_metrics.get("sharpe_ratio", 0),
            }
        }
