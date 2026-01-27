"""
Technical indicators calculation module.

Provides various technical analysis indicators for feature engineering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger("features.technical")


# =============================================================================
# Technical Indicators Class
# =============================================================================

class TechnicalIndicators:
    """
    Technical indicators calculator.
    
    Computes various technical analysis indicators from OHLCV data.
    All indicators are computed using only past data (no lookahead bias).
    """
    
    def __init__(self, df: pd.DataFrame | None = None):
        """
        Initialize with optional DataFrame.
        
        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)
        """
        self._df = df
    
    def set_data(self, df: pd.DataFrame) -> "TechnicalIndicators":
        """Set the data DataFrame."""
        self._df = df.copy()
        return self
    
    @property
    def df(self) -> pd.DataFrame:
        """Get the current DataFrame."""
        if self._df is None:
            raise ValueError("No data set. Call set_data() first.")
        return self._df
    
    # =========================================================================
    # Moving Averages
    # =========================================================================
    
    def sma(self, period: int, column: str = "close") -> pd.Series:
        """
        Simple Moving Average.
        
        Args:
            period: Number of periods
            column: Column to calculate SMA on
            
        Returns:
            SMA series
        """
        return self.df[column].rolling(window=period, min_periods=period).mean()
    
    def ema(self, period: int, column: str = "close") -> pd.Series:
        """
        Exponential Moving Average.
        
        Args:
            period: Number of periods
            column: Column to calculate EMA on
            
        Returns:
            EMA series
        """
        return self.df[column].ewm(span=period, adjust=False, min_periods=period).mean()
    
    def wma(self, period: int, column: str = "close") -> pd.Series:
        """
        Weighted Moving Average.
        
        Args:
            period: Number of periods
            column: Column to calculate WMA on
            
        Returns:
            WMA series
        """
        weights = np.arange(1, period + 1)
        return self.df[column].rolling(window=period).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True
        )
    
    def ema_slope(self, period: int, column: str = "close") -> pd.Series:
        """
        Slope of EMA (rate of change).
        
        Args:
            period: EMA period
            column: Column to calculate on
            
        Returns:
            EMA slope series
        """
        ema = self.ema(period, column)
        return ema.diff() / ema.shift(1)
    
    # =========================================================================
    # Volatility Indicators
    # =========================================================================
    
    def atr(self, period: int = 14) -> pd.Series:
        """
        Average True Range.
        
        Args:
            period: ATR period
            
        Returns:
            ATR series
        """
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False, min_periods=period).mean()
    
    def bollinger_bands(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = "close",
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        
        Args:
            period: Moving average period
            std_dev: Standard deviation multiplier
            column: Column to calculate on
            
        Returns:
            Tuple of (upper, middle, lower) bands
        """
        middle = self.sma(period, column)
        std = self.df[column].rolling(window=period, min_periods=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def bollinger_width(self, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """
        Bollinger Band Width (volatility indicator).
        
        Args:
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Bollinger width series
        """
        upper, middle, lower = self.bollinger_bands(period, std_dev)
        return (upper - lower) / middle
    
    def realized_volatility(self, period: int = 20, column: str = "close") -> pd.Series:
        """
        Realized (historical) volatility.
        
        Args:
            period: Lookback period
            column: Column to calculate on
            
        Returns:
            Realized volatility series
        """
        log_returns = np.log(self.df[column] / self.df[column].shift(1))
        return log_returns.rolling(window=period, min_periods=period).std() * np.sqrt(252)
    
    # =========================================================================
    # Momentum Indicators
    # =========================================================================
    
    def rsi(self, period: int = 14, column: str = "close") -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            period: RSI period
            column: Column to calculate on
            
        Returns:
            RSI series (0-100)
        """
        delta = self.df[column].diff()
        
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "close",
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            column: Column to calculate on
            
        Returns:
            Tuple of (MACD line, signal line, histogram)
        """
        fast_ema = self.ema(fast_period, column)
        slow_ema = self.ema(slow_period, column)
        
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def stochastic(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.
        
        Args:
            k_period: %K period
            d_period: %D period (signal line)
            smooth_k: %K smoothing period
            
        Returns:
            Tuple of (%K, %D)
        """
        low_min = self.df["low"].rolling(window=k_period, min_periods=k_period).min()
        high_max = self.df["high"].rolling(window=k_period, min_periods=k_period).max()
        
        stoch_k = 100 * (self.df["close"] - low_min) / (high_max - low_min)
        
        # Smooth %K
        if smooth_k > 1:
            stoch_k = stoch_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
        
        # %D is SMA of %K
        stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()
        
        return stoch_k, stoch_d
    
    def cci(self, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index.
        
        Args:
            period: CCI period
            
        Returns:
            CCI series
        """
        typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        sma_tp = typical_price.rolling(window=period, min_periods=period).mean()
        mean_deviation = typical_price.rolling(window=period, min_periods=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    def williams_r(self, period: int = 14) -> pd.Series:
        """
        Williams %R.
        
        Args:
            period: Lookback period
            
        Returns:
            Williams %R series (-100 to 0)
        """
        high_max = self.df["high"].rolling(window=period, min_periods=period).max()
        low_min = self.df["low"].rolling(window=period, min_periods=period).min()
        
        return -100 * (high_max - self.df["close"]) / (high_max - low_min)
    
    # =========================================================================
    # Trend Indicators
    # =========================================================================
    
    def adx(self, period: int = 14) -> pd.Series:
        """
        Average Directional Index (trend strength).
        
        Args:
            period: ADX period
            
        Returns:
            ADX series (0-100)
        """
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        
        # Smoothed TR and DM
        atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False, min_periods=period).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False, min_periods=period).mean()
        
        return adx
    
    def supertrend(
        self,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Supertrend indicator.
        
        Args:
            period: ATR period
            multiplier: ATR multiplier
            
        Returns:
            Tuple of (supertrend line, direction)
        """
        atr = self.atr(period)
        hl2 = (self.df["high"] + self.df["low"]) / 2
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = pd.Series(index=self.df.index, dtype=float)
        direction = pd.Series(index=self.df.index, dtype=int)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(self.df)):
            if self.df["close"].iloc[i] > supertrend.iloc[i - 1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
        
        return supertrend, direction
    
    # =========================================================================
    # Volume Indicators
    # =========================================================================
    
    def obv(self) -> pd.Series:
        """
        On-Balance Volume.
        
        Returns:
            OBV series
        """
        volume = self.df["volume"]
        close = self.df["close"]
        
        direction = np.where(close > close.shift(1), 1, 
                           np.where(close < close.shift(1), -1, 0))
        
        return (volume * direction).cumsum()
    
    def vwap(self) -> pd.Series:
        """
        Volume Weighted Average Price (intraday).
        
        Returns:
            VWAP series
        """
        typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        return (typical_price * self.df["volume"]).cumsum() / self.df["volume"].cumsum()
    
    def mfi(self, period: int = 14) -> pd.Series:
        """
        Money Flow Index.
        
        Args:
            period: MFI period
            
        Returns:
            MFI series (0-100)
        """
        typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        raw_money_flow = typical_price * self.df["volume"]
        
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0.0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0.0)
        
        positive_mf = positive_flow.rolling(window=period, min_periods=period).sum()
        negative_mf = negative_flow.rolling(window=period, min_periods=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    # =========================================================================
    # Price Patterns
    # =========================================================================
    
    def log_return(self, periods: int = 1, column: str = "close") -> pd.Series:
        """
        Log return.
        
        Args:
            periods: Number of periods
            column: Column to calculate on
            
        Returns:
            Log return series
        """
        return np.log(self.df[column] / self.df[column].shift(periods))
    
    def price_range(self) -> pd.Series:
        """
        High-Low range.
        
        Returns:
            Range series
        """
        return self.df["high"] - self.df["low"]
    
    def range_position(self, period: int = 20) -> pd.Series:
        """
        Position within recent range (0-1).
        
        Args:
            period: Lookback period
            
        Returns:
            Range position series
        """
        high_max = self.df["high"].rolling(window=period, min_periods=period).max()
        low_min = self.df["low"].rolling(window=period, min_periods=period).min()
        
        return (self.df["close"] - low_min) / (high_max - low_min)
    
    def gap(self) -> pd.Series:
        """
        Gap between previous close and current open.
        
        Returns:
            Gap series
        """
        return self.df["open"] - self.df["close"].shift(1)
    
    # =========================================================================
    # All Indicators
    # =========================================================================
    
    def compute_all(
        self,
        ema_periods: list[int] | None = None,
        atr_period: int = 14,
        rsi_period: int = 14,
        macd_params: tuple[int, int, int] | None = None,
        bb_params: tuple[int, float] | None = None,
        adx_period: int = 14,
    ) -> pd.DataFrame:
        """
        Compute all common technical indicators.
        
        Args:
            ema_periods: List of EMA periods (default: [10, 20, 50])
            atr_period: ATR period
            rsi_period: RSI period
            macd_params: MACD parameters (fast, slow, signal)
            bb_params: Bollinger Bands parameters (period, std_dev)
            adx_period: ADX period
            
        Returns:
            DataFrame with all indicators
        """
        if ema_periods is None:
            ema_periods = [10, 20, 50]
        if macd_params is None:
            macd_params = (12, 26, 9)
        if bb_params is None:
            bb_params = (20, 2.0)
        
        result = self.df.copy()
        
        # Moving Averages
        for period in ema_periods:
            result[f"ema_{period}"] = self.ema(period)
            result[f"ema_slope_{period}"] = self.ema_slope(period)
        
        # Volatility
        result["atr"] = self.atr(atr_period)
        upper, middle, lower = self.bollinger_bands(bb_params[0], bb_params[1])
        result["bb_upper"] = upper
        result["bb_middle"] = middle
        result["bb_lower"] = lower
        result["bb_width"] = self.bollinger_width(bb_params[0], bb_params[1])
        result["realized_vol"] = self.realized_volatility()
        
        # Momentum
        result["rsi"] = self.rsi(rsi_period)
        macd_line, signal, hist = self.macd(*macd_params)
        result["macd"] = macd_line
        result["macd_signal"] = signal
        result["macd_hist"] = hist
        stoch_k, stoch_d = self.stochastic()
        result["stoch_k"] = stoch_k
        result["stoch_d"] = stoch_d
        
        # Trend
        result["adx"] = self.adx(adx_period)
        
        # Price patterns
        result["log_return_1"] = self.log_return(1)
        result["log_return_5"] = self.log_return(5)
        result["price_range"] = self.price_range()
        result["range_position"] = self.range_position()
        result["gap"] = self.gap()
        
        logger.debug(f"Computed {len(result.columns) - len(self.df.columns)} technical indicators")
        return result
