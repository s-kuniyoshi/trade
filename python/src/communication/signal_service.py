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
    """
    
    def __init__(
        self,
        prediction_service: PredictionService,
        feature_engine: FeatureEngine,
        buffer_sizes: dict[str, int] | None = None,
        min_confidence: float | None = None,
    ):
        """
        Initialize signal service.
        
        Args:
            prediction_service: PredictionService instance for predictions
            feature_engine: FeatureEngine instance for feature calculation
            buffer_sizes: Buffer sizes per timeframe (from config if None)
            min_confidence: Minimum confidence threshold (from config if None)
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
        
        # Data buffers: {symbol: {timeframe: DataFrame}}
        self.buffers: dict[str, dict[str, pd.DataFrame]] = {}
        self.buffer_lock = Lock()
        
        logger.info(
            f"SignalService initialized: "
            f"base_timeframe={self.base_timeframe}, "
            f"higher_timeframes={self.higher_timeframes}, "
            f"min_confidence={self.min_confidence}"
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
            
            # Calculate stop loss and take profit
            current_price = self.buffers[symbol][self.base_timeframe].iloc[-1]["close"]
            
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
