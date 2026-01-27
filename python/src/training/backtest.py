"""
Backtesting module for strategy evaluation.

Provides event-driven backtesting with realistic execution simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger("training.backtest")


# =============================================================================
# Data Classes
# =============================================================================

class OrderType(Enum):
    """Order type enumeration."""
    BUY = "buy"
    SELL = "sell"


class TradeStatus(Enum):
    """Trade status enumeration."""
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class Order:
    """Represents a trading order."""
    timestamp: datetime
    symbol: str
    order_type: OrderType
    lots: float
    entry_price: float
    sl: float | None = None
    tp: float | None = None
    comment: str = ""


@dataclass
class Trade:
    """Represents an executed trade."""
    id: int
    symbol: str
    order_type: OrderType
    lots: float
    entry_time: datetime
    entry_price: float
    sl: float | None = None
    tp: float | None = None
    exit_time: datetime | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    pnl_pips: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    commission: float = 0.0
    swap: float = 0.0
    comment: str = ""
    
    def close(self, exit_time: datetime, exit_price: float, pip_value: float) -> None:
        """Close the trade."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.status = TradeStatus.CLOSED
        
        # Calculate PnL
        if self.order_type == OrderType.BUY:
            price_diff = exit_price - self.entry_price
        else:
            price_diff = self.entry_price - exit_price
        
        self.pnl_pips = price_diff / pip_value
        self.pnl = price_diff * self.lots * (1 / pip_value) * 10  # Simplified PnL


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_balance: float = 10000.0
    commission_per_lot: float = 7.0  # USD per lot
    spread_pips: float = 1.5
    slippage_pips: float = 0.5
    pip_value: float = 0.0001  # For most pairs
    lot_size: float = 100000  # Standard lot
    use_dynamic_spread: bool = True
    max_positions: int = 3
    # Risk management settings
    max_drawdown_pct: float = 0.25  # Max DD before halting
    consecutive_loss_limit: int = 5  # Consecutive losses before risk reduction
    vol_scale_threshold: float = 1.5  # ATR multiplier for volatility adjustment
    risk_per_trade: float = 0.01  # Risk per trade as fraction of equity
    # Symbol restrictions
    long_only_symbols: set[str] = field(default_factory=lambda: {"USDJPY"})


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: list[Trade]
    equity_curve: pd.Series
    metrics: dict[str, float]
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        records = []
        for trade in self.trades:
            records.append({
                "id": trade.id,
                "symbol": trade.symbol,
                "type": trade.order_type.value,
                "lots": trade.lots,
                "entry_time": trade.entry_time,
                "entry_price": trade.entry_price,
                "exit_time": trade.exit_time,
                "exit_price": trade.exit_price,
                "sl": trade.sl,
                "tp": trade.tp,
                "pnl": trade.pnl,
                "pnl_pips": trade.pnl_pips,
                "commission": trade.commission,
                "comment": trade.comment,
            })
        
        return pd.DataFrame(records)
    
    def summary(self) -> str:
        """Get summary string."""
        m = self.metrics
        return (
            f"Backtest Results ({self.start_date.date()} to {self.end_date.date()})\n"
            f"{'='*50}\n"
            f"Total Trades: {m.get('total_trades', 0)}\n"
            f"Win Rate: {m.get('win_rate', 0):.1%}\n"
            f"Profit Factor: {m.get('profit_factor', 0):.2f}\n"
            f"Total PnL: ${m.get('total_pnl', 0):.2f}\n"
            f"Max Drawdown: {m.get('max_drawdown', 0):.1%}\n"
            f"Sharpe Ratio: {m.get('sharpe_ratio', 0):.2f}\n"
            f"Final Balance: ${m.get('final_balance', 0):.2f}\n"
        )


# =============================================================================
# Backtester
# =============================================================================

class Backtester:
    """
    Event-driven backtester.
    
    Simulates trading with realistic execution including spread,
    slippage, and commission.
    """
    
    def __init__(self, config: BacktestConfig | None = None):
        """
        Initialize backtester.
        
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self._reset()
    
    def _reset(self) -> None:
        """Reset backtester state."""
        self._balance = self.config.initial_balance
        self._equity = self.config.initial_balance
        self._trades: list[Trade] = []
        self._open_trades: list[Trade] = []
        self._trade_counter = 0
        self._equity_history: list[tuple[datetime, float]] = []
        # Risk management state
        self._peak_equity = self.config.initial_balance
        self._consecutive_losses: dict[str, int] = {}
        self._trading_halted: dict[str, bool] = {}
        self._avg_atr: dict[str, float] = {}
    
    def run(
        self,
        data: pd.DataFrame,
        signal_func: Callable[[pd.DataFrame, int], dict[str, Any] | None],
        symbol: str = "EURUSD",
    ) -> BacktestResult:
        """
        Run backtest on data.
        
        Args:
            data: OHLCV DataFrame with datetime index
            signal_func: Function that takes (data, current_index) and returns
                        signal dict or None
            symbol: Trading symbol
            
        Returns:
            BacktestResult with all trades and metrics
        """
        self._reset()
        
        # Ensure we have required columns
        required_cols = ["open", "high", "low", "close"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must have columns: {required_cols}")
        
        logger.info(f"Starting backtest on {len(data)} bars")
        
        # Main backtest loop
        for i in range(len(data)):
            current_bar = data.iloc[i]
            current_time = data.index[i]
            
            # Get current price (for position valuation)
            current_price = current_bar["close"]
            
            # Get spread
            if self.config.use_dynamic_spread and "spread" in data.columns:
                spread = data.iloc[i]["spread"]
            else:
                spread = self.config.spread_pips * self.config.pip_value
            
            # 1. Check open positions for SL/TP hits
            self._check_exits(current_bar, current_time, spread)
            
            # 2. Update equity
            self._update_equity(current_time, current_price)
            
            # 3. Get signal
            signal = signal_func(data, i)
            
            # 4. Process signal
            if signal is not None:
                self._process_signal(signal, current_bar, current_time, spread, symbol)
        
        # Close any remaining positions at end
        if self._open_trades:
            final_bar = data.iloc[-1]
            final_time = data.index[-1]
            for trade in self._open_trades.copy():
                self._close_trade(
                    trade,
                    final_time,
                    final_bar["close"],
                    "end_of_backtest"
                )
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Build equity curve
        equity_curve = pd.Series(
            [eq for _, eq in self._equity_history],
            index=[t for t, _ in self._equity_history]
        )
        
        result = BacktestResult(
            trades=self._trades,
            equity_curve=equity_curve,
            metrics=metrics,
            config=self.config,
            start_date=data.index[0].to_pydatetime() if hasattr(data.index[0], 'to_pydatetime') else data.index[0],
            end_date=data.index[-1].to_pydatetime() if hasattr(data.index[-1], 'to_pydatetime') else data.index[-1],
        )
        
        logger.info(f"Backtest complete: {len(self._trades)} trades")
        return result
    
    def _check_exits(
        self,
        bar: pd.Series,
        timestamp: datetime,
        spread: float,
    ) -> None:
        """Check for SL/TP exits on open trades."""
        for trade in self._open_trades.copy():
            exit_price = None
            exit_reason = None
            
            if trade.order_type == OrderType.BUY:
                # Check SL (hit if low <= SL)
                if trade.sl is not None and bar["low"] <= trade.sl:
                    exit_price = trade.sl - self.config.slippage_pips * self.config.pip_value
                    exit_reason = "sl"
                # Check TP (hit if high >= TP)
                elif trade.tp is not None and bar["high"] >= trade.tp:
                    exit_price = trade.tp
                    exit_reason = "tp"
            else:  # SELL
                # Check SL (hit if high >= SL)
                if trade.sl is not None and bar["high"] >= trade.sl:
                    exit_price = trade.sl + self.config.slippage_pips * self.config.pip_value + spread
                    exit_reason = "sl"
                # Check TP (hit if low <= TP)
                elif trade.tp is not None and bar["low"] <= trade.tp:
                    exit_price = trade.tp + spread
                    exit_reason = "tp"
            
            if exit_price is not None:
                self._close_trade(trade, timestamp, exit_price, exit_reason)
    
    def _process_signal(
        self,
        signal: dict[str, Any],
        bar: pd.Series,
        timestamp: datetime,
        spread: float,
        symbol: str,
    ) -> None:
        """Process a trading signal."""
        direction = signal.get("direction")
        if direction not in ["buy", "sell"]:
            return
        
        # Check position limits
        if len(self._open_trades) >= self.config.max_positions:
            return
        
        # Risk management: Long-only symbols
        if symbol in self.config.long_only_symbols and direction == "sell":
            return
        
        # Risk management: Check if trading is halted due to drawdown
        if self._trading_halted.get(symbol, False):
            # Check if DD has recovered to half of threshold
            if self._peak_equity > 0:
                current_dd = (self._peak_equity - self._equity) / self._peak_equity
                if current_dd < self.config.max_drawdown_pct * 0.5:
                    self._trading_halted[symbol] = False
                else:
                    return
        
        # Risk management: Check drawdown limit
        if self._peak_equity > 0:
            current_dd = (self._peak_equity - self._equity) / self._peak_equity
            if current_dd >= self.config.max_drawdown_pct:
                self._trading_halted[symbol] = True
                return
        
        # Get order parameters
        lots = signal.get("lots", 0.1)
        sl = signal.get("sl")
        tp = signal.get("tp")
        confidence = signal.get("confidence", 0.5)
        
        # Risk management: Adjust lots based on consecutive losses
        risk_multiplier = 1.0
        consecutive = self._consecutive_losses.get(symbol, 0)
        if consecutive >= self.config.consecutive_loss_limit:
            risk_multiplier *= 0.5
        
        # Apply risk adjustment to lot size
        lots = lots * risk_multiplier
        
        # Determine entry price with spread and slippage
        if direction == "buy":
            entry_price = bar["close"] + spread + self.config.slippage_pips * self.config.pip_value
            order_type = OrderType.BUY
        else:
            entry_price = bar["close"] - self.config.slippage_pips * self.config.pip_value
            order_type = OrderType.SELL
        
        # Create and execute trade
        self._trade_counter += 1
        trade = Trade(
            id=self._trade_counter,
            symbol=symbol,
            order_type=order_type,
            lots=lots,
            entry_time=timestamp,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            commission=self.config.commission_per_lot * lots,
            comment=f"confidence={confidence:.2f}",
        )
        
        self._open_trades.append(trade)
        self._balance -= trade.commission
        
        logger.debug(f"Opened {direction} {lots} @ {entry_price:.5f}")
    
    def _close_trade(
        self,
        trade: Trade,
        timestamp: datetime,
        exit_price: float,
        reason: str,
    ) -> None:
        """Close a trade."""
        trade.close(timestamp, exit_price, self.config.pip_value)
        trade.comment += f" | exit={reason}"
        
        self._open_trades.remove(trade)
        self._trades.append(trade)
        self._balance += trade.pnl
        
        # Risk management: Update consecutive losses and peak equity
        symbol = trade.symbol
        if trade.pnl > 0:
            # Win: reset consecutive losses, update peak equity
            self._consecutive_losses[symbol] = 0
            if self._equity > self._peak_equity:
                self._peak_equity = self._equity
        else:
            # Loss: increment consecutive losses
            self._consecutive_losses[symbol] = self._consecutive_losses.get(symbol, 0) + 1
        
        logger.debug(f"Closed trade {trade.id}: PnL={trade.pnl:.2f} ({reason})")
    
    def _update_equity(self, timestamp: datetime, current_price: float) -> None:
        """Update equity with open position values."""
        floating_pnl = 0.0
        for trade in self._open_trades:
            if trade.order_type == OrderType.BUY:
                floating_pnl += (current_price - trade.entry_price) * trade.lots * self.config.lot_size
            else:
                floating_pnl += (trade.entry_price - current_price) * trade.lots * self.config.lot_size
        
        self._equity = self._balance + floating_pnl
        self._equity_history.append((timestamp, self._equity))
    
    def _calculate_metrics(self) -> dict[str, float]:
        """Calculate backtest performance metrics."""
        if not self._trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_pnl": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "final_balance": self._balance,
            }
        
        # Basic stats
        closed_trades = [t for t in self._trades if t.status == TradeStatus.CLOSED]
        winners = [t for t in closed_trades if t.pnl > 0]
        losers = [t for t in closed_trades if t.pnl <= 0]
        
        total_profit = sum(t.pnl for t in winners) if winners else 0.0
        total_loss = abs(sum(t.pnl for t in losers)) if losers else 0.0
        
        # Equity curve analysis
        equity_values = [eq for _, eq in self._equity_history]
        if equity_values:
            equity_series = pd.Series(equity_values)
            returns = equity_series.pct_change().dropna()
            
            # Max drawdown
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
            
            # Sharpe ratio (annualized, assuming hourly bars)
            if len(returns) > 1 and returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24)
            else:
                sharpe = 0.0
        else:
            max_drawdown = 0.0
            sharpe = 0.0
        
        return {
            "total_trades": len(closed_trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(closed_trades) if closed_trades else 0.0,
            "profit_factor": total_profit / total_loss if total_loss > 0 else float("inf"),
            "total_pnl": sum(t.pnl for t in closed_trades),
            "avg_pnl": np.mean([t.pnl for t in closed_trades]) if closed_trades else 0.0,
            "avg_winner": np.mean([t.pnl for t in winners]) if winners else 0.0,
            "avg_loser": np.mean([t.pnl for t in losers]) if losers else 0.0,
            "max_winner": max([t.pnl for t in winners]) if winners else 0.0,
            "max_loser": min([t.pnl for t in losers]) if losers else 0.0,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "final_balance": self._balance,
            "total_commission": sum(t.commission for t in closed_trades),
        }


# =============================================================================
# Simple Signal Function Example
# =============================================================================

def create_simple_signal_func(
    model,
    feature_columns: list[str],
    threshold: float = 0.65,
    lot_size: float = 0.1,
    sl_pips: float = 30,
    tp_pips: float = 45,
    pip_value: float = 0.0001,
) -> Callable[[pd.DataFrame, int], dict[str, Any] | None]:
    """
    Create a signal function for backtesting with ML model.
    
    Args:
        model: Trained ML model with predict() method
        feature_columns: List of feature column names
        threshold: Confidence threshold for trading
        lot_size: Position size in lots
        sl_pips: Stop loss in pips
        tp_pips: Take profit in pips
        pip_value: Pip value for the symbol
        
    Returns:
        Signal function for backtester
    """
    def signal_func(data: pd.DataFrame, idx: int) -> dict[str, Any] | None:
        # Need enough history
        if idx < 50:
            return None
        
        # Get features for current bar
        current_features = data.iloc[idx][feature_columns]
        
        # Check for missing values
        if current_features.isna().any():
            return None
        
        # Get prediction
        X = current_features.values.reshape(1, -1)
        prediction = model.predict(X)[0]
        
        # Get confidence if model supports it
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = max(proba)
        else:
            confidence = abs(prediction)
        
        # Check threshold
        if confidence < threshold:
            return None
        
        # Determine direction
        current_price = data.iloc[idx]["close"]
        
        if prediction > 0:
            direction = "buy"
            sl = current_price - sl_pips * pip_value
            tp = current_price + tp_pips * pip_value
        else:
            direction = "sell"
            sl = current_price + sl_pips * pip_value
            tp = current_price - tp_pips * pip_value
        
        return {
            "direction": direction,
            "lots": lot_size,
            "sl": sl,
            "tp": tp,
            "confidence": confidence,
        }
    
    return signal_func
