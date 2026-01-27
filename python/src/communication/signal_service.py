"""
Real-time signal generation service.

Receives OHLCV data from MT5, calculates features using FeatureEngine,
and generates trading signals using PredictionService.
"""

from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any

import pandas as pd

from ..features.mtf import FeatureEngine
from ..inference.predictor import PredictionService
from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger("communication.signal_service")


# =============================================================================
# Signal Service
# =============================================================================

class SignalService:
    """
    Real-time trading signal generation service.
    
    Manages multi-timeframe OHLCV data buffers, calculates features,
    and generates trading signals with confidence filtering.
    
    Incorporates risk management features:
    - USDJPY long-only (no short signals)
    - ADX trend filter
    - SMA200 distance filter
    - Time-of-day filter (avoid low liquidity hours)
    - Consecutive loss tracking
    - Volatility-based position sizing
    """
    
    # Symbol-specific restrictions based on backtesting
    LONG_ONLY_SYMBOLS = {"USDJPY"}  # Symbols that only trade long
    
    def __init__(
        self,
        prediction_service: PredictionService,
        feature_engine: FeatureEngine,
        buffer_sizes: dict[str, int] | None = None,
        min_confidence: float | None = None,
        # Filter settings
        use_filters: bool = True,
        adx_threshold: float = 20.0,
        sma_atr_threshold: float = 0.5,
        # Risk management settings
        max_drawdown_pct: float = 0.25,
        consecutive_loss_limit: int = 5,
        vol_scale_threshold: float = 1.5,
    ):
        """
        Initialize signal service.
        
        Args:
            prediction_service: PredictionService instance for predictions
            feature_engine: FeatureEngine instance for feature calculation
            buffer_sizes: Buffer sizes per timeframe (from config if None)
            min_confidence: Minimum confidence threshold (from config if None)
            use_filters: Whether to apply trading filters
            adx_threshold: Minimum ADX for trend strength
            sma_atr_threshold: Minimum SMA200 distance in ATR units
            max_drawdown_pct: Maximum drawdown before halting trading
            consecutive_loss_limit: Number of consecutive losses before reducing risk
            vol_scale_threshold: ATR multiplier threshold for volatility adjustment
        """
        config = get_config()
        
        # Services
        self.prediction_service = prediction_service
        self.feature_engine = feature_engine
        
        # Configuration
        self.base_timeframe = feature_engine.base_timeframe
        self.higher_timeframes = feature_engine.higher_timeframes
        
        # Buffer sizes (bars to keep per timeframe)
        if buffer_sizes is None:
            buffer_sizes = {
                "H1": 500,
                "H4": 200,
                "D1": 100,
                "M15": 1000,
                "M5": 2000,
                "M1": 5000,
            }
        self.buffer_sizes = buffer_sizes
        
        # Minimum bars before generating signals
        self.min_bars_required = {
            "H1": 100,
            "H4": 50,
            "D1": 30,
            "M15": 100,
            "M5": 100,
            "M1": 100,
        }
        
        # Confidence threshold
        self.min_confidence = min_confidence if min_confidence is not None else \
            config.trading.signals.get("min_confidence", 0.65)
        
        # Filter settings
        self.use_filters = use_filters
        self.adx_threshold = adx_threshold
        self.sma_atr_threshold = sma_atr_threshold
        
        # Risk management settings
        self.max_drawdown_pct = max_drawdown_pct
        self.consecutive_loss_limit = consecutive_loss_limit
        self.vol_scale_threshold = vol_scale_threshold
        
        # Risk management state (per symbol)
        self.consecutive_losses: dict[str, int] = {}
        self.peak_equity: dict[str, float] = {}
        self.current_equity: dict[str, float] = {}
        self.trading_halted: dict[str, bool] = {}
        self.avg_atr: dict[str, float] = {}
        
        # Data buffers: {symbol: {timeframe: DataFrame}}
        self.buffers: dict[str, dict[str, pd.DataFrame]] = {}
        self.buffer_lock = Lock()
        
        logger.info(
            f"SignalService initialized: "
            f"base_timeframe={self.base_timeframe}, "
            f"higher_timeframes={self.higher_timeframes}, "
            f"min_confidence={self.min_confidence}, "
            f"use_filters={self.use_filters}, "
            f"adx_threshold={self.adx_threshold}, "
            f"sma_atr_threshold={self.sma_atr_threshold}"
        )
    
    def add_bar(
        self,
        symbol: str,
        timeframe: str,
        bar_data: dict[str, Any],
    ) -> None:
        """
        Add a new OHLCV bar to the buffer.
        
        Args:
            symbol: Trading symbol (e.g., "USDJPY")
            timeframe: Timeframe string (e.g., "H1")
            bar_data: Bar data dict with keys:
                - time: ISO format timestamp
                - open: Opening price
                - high: High price
                - low: Low price
                - close: Closing price
                - volume: Volume
                
        Raises:
            ValueError: If bar_data is invalid
        """
        # Validate bar data
        required_keys = {"time", "open", "high", "low", "close", "volume"}
        if not all(k in bar_data for k in required_keys):
            raise ValueError(f"Missing required keys in bar_data: {required_keys}")
        
        try:
            # Convert bar to DataFrame row
            bar_time = pd.to_datetime(bar_data["time"])
            bar_row = pd.DataFrame(
                {
                    "open": [float(bar_data["open"])],
                    "high": [float(bar_data["high"])],
                    "low": [float(bar_data["low"])],
                    "close": [float(bar_data["close"])],
                    "volume": [float(bar_data["volume"])],
                },
                index=[bar_time],
            )
            
            # Add to buffer (thread-safe)
            with self.buffer_lock:
                # Initialize symbol buffers if needed
                if symbol not in self.buffers:
                    self.buffers[symbol] = {}
                
                # Initialize timeframe buffer if needed
                if timeframe not in self.buffers[symbol]:
                    self.buffers[symbol][timeframe] = pd.DataFrame()
                
                # Append bar
                df = self.buffers[symbol][timeframe]
                df = pd.concat([df, bar_row])
                
                # Keep only last N bars
                max_bars = self.buffer_sizes.get(timeframe, 500)
                df = df.tail(max_bars)
                
                # Update buffer
                self.buffers[symbol][timeframe] = df
            
            logger.debug(
                f"Added bar: {symbol} {timeframe} "
                f"close={bar_data['close']} "
                f"buffer_size={len(df)}"
            )
        
        except Exception as e:
            logger.error(f"Error adding bar for {symbol} {timeframe}: {e}")
            raise ValueError(f"Invalid bar data: {e}") from e
    
    def generate_signal(self, symbol: str) -> dict[str, Any] | None:
        """
        Generate trading signal for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "USDJPY")
            
        Returns:
            Signal dict with keys:
            - symbol: Trading symbol
            - timeframe: Primary timeframe
            - direction: "buy", "sell", or "neutral"
            - confidence: Confidence score (0.0-1.0)
            - expected_return: Expected return
            - stop_loss: Stop loss price
            - take_profit: Take profit price
            - timestamp: ISO format timestamp
            - features_used: Number of features
            - bars_in_buffer: Number of bars in primary buffer
            
            Returns None if:
            - Insufficient data in buffers
            - Feature calculation fails
            - Confidence below threshold
        """
        try:
            # Check if we have sufficient data
            if not self._has_sufficient_data(symbol):
                logger.debug(f"Insufficient data for {symbol}")
                return None
            
            # Prepare data for FeatureEngine
            with self.buffer_lock:
                data = {}
                data[self.base_timeframe] = self.buffers[symbol][self.base_timeframe].copy()
                
                for htf in self.higher_timeframes:
                    if htf in self.buffers[symbol]:
                        data[htf] = self.buffers[symbol][htf].copy()
            
            # Calculate features
            try:
                features_df = self.feature_engine.compute_all_features(
                    data,
                    include_target=False,
                )
            except Exception as e:
                logger.error(f"Feature calculation failed for {symbol}: {e}")
                return None
            
            # Use only last completed bar (avoid lookahead bias)
            if features_df.empty:
                logger.warning(f"No features computed for {symbol}")
                return None
            
            current_features = features_df.iloc[[-1]]
            
            # Get prediction
            try:
                prediction = self.prediction_service.predict(
                    current_features,
                    min_confidence=0.0,  # We'll filter by confidence ourselves
                )
            except Exception as e:
                logger.error(f"Prediction failed for {symbol}: {e}")
                return None
            
            # Filter by minimum confidence
            if prediction["confidence"] < self.min_confidence:
                logger.debug(
                    f"Signal filtered for {symbol}: "
                    f"confidence={prediction['confidence']:.3f} < {self.min_confidence}"
                )
                return None
            
            # USDJPY long-only filter
            if symbol in self.LONG_ONLY_SYMBOLS and prediction["direction"] == "sell":
                logger.debug(
                    f"Signal filtered for {symbol}: "
                    f"short signals disabled for this symbol (long-only)"
                )
                return None
            
            # Get current price and features for filters
            current_price = self.buffers[symbol][self.base_timeframe].iloc[-1]["close"]
            
            # Apply trading filters
            if self.use_filters:
                filter_result = self._apply_filters(symbol, current_features)
                if filter_result is not None:
                    logger.debug(f"Signal filtered for {symbol}: {filter_result}")
                    return None
            
            # Get ATR from features (if available)
            atr = 0.01  # Default fallback
            if "atr_14" in current_features.columns:
                atr = float(current_features.iloc[-1]["atr_14"])
            elif "atr" in current_features.columns:
                atr = float(current_features.iloc[-1]["atr"])
            
            sl, tp = self._calculate_sl_tp(
                current_price,
                atr,
                prediction["direction"],
            )
            
            # Build signal
            signal = {
                "symbol": symbol,
                "timeframe": self.base_timeframe,
                "direction": prediction["direction"],
                "confidence": prediction["confidence"],
                "expected_return": prediction["expected_return"],
                "stop_loss": sl,
                "take_profit": tp,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "features_used": len(current_features.columns),
                "bars_in_buffer": len(self.buffers[symbol][self.base_timeframe]),
            }
            
            logger.info(
                f"Signal generated for {symbol}: "
                f"direction={signal['direction']}, "
                f"confidence={signal['confidence']:.3f}, "
                f"return={signal['expected_return']:.6f}"
            )
            
            return signal
        
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def get_buffer_status(self, symbol: str) -> dict[str, Any]:
        """
        Get buffer status for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Status dict with buffer information
        """
        with self.buffer_lock:
            if symbol not in self.buffers:
                return {
                    "symbol": symbol,
                    "initialized": False,
                    "timeframes": {},
                }
            
            status = {
                "symbol": symbol,
                "initialized": True,
                "timeframes": {},
            }
            
            for tf, df in self.buffers[symbol].items():
                first_bar = None
                last_bar = None
                if len(df) > 0:
                    first_bar = str(df.index[0])
                    last_bar = str(df.index[-1])
                
                status["timeframes"][tf] = {
                    "bars": len(df),
                    "min_required": self.min_bars_required.get(tf, 100),
                    "sufficient": len(df) >= self.min_bars_required.get(tf, 100),
                    "first_bar": first_bar,
                    "last_bar": last_bar,
                }
            
            return status
    
    def clear_buffer(self, symbol: str) -> None:
        """
        Clear data buffer for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        with self.buffer_lock:
            if symbol in self.buffers:
                del self.buffers[symbol]
                logger.info(f"Buffer cleared for {symbol}")
    
    def _has_sufficient_data(self, symbol: str) -> bool:
        """
        Check if we have sufficient data to generate signals.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if sufficient data available
        """
        with self.buffer_lock:
            if symbol not in self.buffers:
                return False
            
            # Check base timeframe
            if self.base_timeframe not in self.buffers[symbol]:
                return False
            
            base_df = self.buffers[symbol][self.base_timeframe]
            min_bars = self.min_bars_required.get(self.base_timeframe, 100)
            
            if len(base_df) < min_bars:
                return False
            
            # Check higher timeframes (if available)
            for htf in self.higher_timeframes:
                if htf in self.buffers[symbol]:
                    htf_df = self.buffers[symbol][htf]
                    htf_min_bars = self.min_bars_required.get(htf, 50)
                    if len(htf_df) < htf_min_bars:
                        logger.debug(
                            f"Insufficient {htf} data for {symbol}: "
                            f"{len(htf_df)} < {htf_min_bars}"
                        )
                        return False
            
            return True
    
    def _calculate_sl_tp(
        self,
        current_price: float,
        atr: float,
        direction: str,
    ) -> tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            current_price: Current price
            atr: Average True Range
            direction: Trade direction ("buy", "sell", or "neutral")
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        if direction == "buy":
            sl = current_price - 2 * atr
            tp = current_price + 3 * atr
        elif direction == "sell":
            sl = current_price + 2 * atr
            tp = current_price - 3 * atr
        else:
            # Neutral - no SL/TP
            sl = current_price
            tp = current_price
        
        return sl, tp
    
    def _apply_filters(
        self,
        symbol: str,
        features: pd.DataFrame,
    ) -> str | None:
        """
        Apply trading filters based on backtesting insights.
        
        Args:
            symbol: Trading symbol
            features: Current features DataFrame
            
        Returns:
            Filter rejection reason string, or None if all filters pass
        """
        # Get current timestamp for time filter
        last_bar = self.buffers[symbol][self.base_timeframe].index[-1]
        hour = pd.Timestamp(last_bar).hour
        
        # Time filter: Avoid low liquidity hours (UTC 21:00-01:00)
        if hour >= 21 or hour < 1:
            return f"time_filter: hour={hour} in excluded range (21-01 UTC)"
        
        # ADX trend filter
        if "adx_14" in features.columns:
            adx = float(features.iloc[-1]["adx_14"])
            if adx < self.adx_threshold:
                return f"adx_filter: {adx:.1f} < {self.adx_threshold}"
        
        # SMA200 distance filter
        if "sma200_atr_distance" in features.columns:
            sma_dist = float(features.iloc[-1]["sma200_atr_distance"])
            if sma_dist < self.sma_atr_threshold:
                return f"sma_distance_filter: {sma_dist:.2f} < {self.sma_atr_threshold}"
        
        return None
    
    def update_trade_result(
        self,
        symbol: str,
        pnl: float,
        is_win: bool,
    ) -> None:
        """
        Update risk management state after a trade closes.
        
        Args:
            symbol: Trading symbol
            pnl: Profit/loss amount
            is_win: Whether the trade was profitable
        """
        # Initialize if needed
        if symbol not in self.consecutive_losses:
            self.consecutive_losses[symbol] = 0
        if symbol not in self.current_equity:
            self.current_equity[symbol] = 10000.0  # Default
        if symbol not in self.peak_equity:
            self.peak_equity[symbol] = 10000.0
        
        # Update equity
        self.current_equity[symbol] += pnl
        
        if is_win:
            # Reset consecutive losses on win
            self.consecutive_losses[symbol] = 0
            # Update peak equity
            if self.current_equity[symbol] > self.peak_equity[symbol]:
                self.peak_equity[symbol] = self.current_equity[symbol]
        else:
            # Increment consecutive losses
            self.consecutive_losses[symbol] += 1
        
        # Check drawdown
        if self.peak_equity[symbol] > 0:
            current_dd = (self.peak_equity[symbol] - self.current_equity[symbol]) / self.peak_equity[symbol]
            
            if current_dd >= self.max_drawdown_pct:
                self.trading_halted[symbol] = True
                logger.warning(
                    f"Trading halted for {symbol}: "
                    f"drawdown {current_dd*100:.1f}% >= {self.max_drawdown_pct*100:.1f}%"
                )
            elif self.trading_halted.get(symbol, False) and current_dd < self.max_drawdown_pct * 0.5:
                # Resume trading if DD recovers to half of threshold
                self.trading_halted[symbol] = False
                logger.info(f"Trading resumed for {symbol}: drawdown recovered")
        
        logger.debug(
            f"Trade result updated for {symbol}: "
            f"pnl={pnl:.2f}, win={is_win}, "
            f"consecutive_losses={self.consecutive_losses[symbol]}, "
            f"equity={self.current_equity[symbol]:.2f}"
        )
    
    def get_risk_adjustment(self, symbol: str) -> float:
        """
        Get risk adjustment multiplier based on current state.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Risk multiplier (0.0 to 1.0)
        """
        # Check if trading is halted
        if self.trading_halted.get(symbol, False):
            return 0.0
        
        multiplier = 1.0
        
        # Consecutive loss adjustment
        consecutive = self.consecutive_losses.get(symbol, 0)
        if consecutive >= self.consecutive_loss_limit:
            multiplier *= 0.5
            logger.debug(
                f"Risk reduced for {symbol}: "
                f"{consecutive} consecutive losses >= {self.consecutive_loss_limit}"
            )
        
        # Volatility adjustment
        if symbol in self.avg_atr and symbol in self.buffers:
            if self.base_timeframe in self.buffers[symbol]:
                df = self.buffers[symbol][self.base_timeframe]
                if "atr_14" in df.columns and len(df) > 0:
                    current_atr = float(df.iloc[-1].get("atr_14", 0) or 0)
                    avg = float(self.avg_atr.get(symbol, current_atr) or current_atr)
                    if avg > 0 and current_atr > 0:
                        vol_ratio = current_atr / avg
                        if vol_ratio > self.vol_scale_threshold:
                            vol_adj = max(0.5, 1.0 / vol_ratio)
                            multiplier *= vol_adj
                            logger.debug(
                                f"Risk reduced for {symbol}: "
                                f"vol_ratio={vol_ratio:.2f} > {self.vol_scale_threshold}"
                            )
        
        return multiplier
    
    def set_initial_equity(self, symbol: str, equity: float) -> None:
        """Set initial equity for a symbol."""
        self.current_equity[symbol] = equity
        self.peak_equity[symbol] = equity
        self.consecutive_losses[symbol] = 0
        self.trading_halted[symbol] = False
