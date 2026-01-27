"""
Data loader module for fetching market data.

Supports loading data from MetaTrader 5 and cached files.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger("data.loader")


# =============================================================================
# Timeframe Mapping
# =============================================================================

TIMEFRAME_MAP = {
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


def get_mt5_timeframe(timeframe: str) -> int:
    """
    Convert timeframe string to MT5 timeframe constant.
    
    Args:
        timeframe: Timeframe string (e.g., "H1", "D1")
        
    Returns:
        MT5 timeframe constant
    """
    try:
        import MetaTrader5 as mt5
        
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
        return tf_map.get(timeframe.upper(), mt5.TIMEFRAME_H1)
    except ImportError:
        # Fallback for non-MT5 environments
        return TIMEFRAME_MAP.get(timeframe.upper(), 60)


# =============================================================================
# Abstract Data Loader
# =============================================================================

class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        bars: int | None = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Timeframe string (e.g., "H1")
            start: Start datetime
            end: End datetime
            bars: Number of bars to load (alternative to start/end)
            
        Returns:
            DataFrame with columns: open, high, low, close, volume, time
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> dict[str, float]:
        """
        Get current bid/ask prices.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with 'bid', 'ask', 'spread' keys
        """
        pass


# =============================================================================
# MetaTrader 5 Data Loader
# =============================================================================

class MT5DataLoader(DataLoader):
    """
    Data loader for MetaTrader 5.
    
    Loads OHLCV data directly from MT5 terminal.
    """
    
    def __init__(
        self,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
    ):
        """
        Initialize MT5 data loader.
        
        Args:
            login: MT5 account number
            password: MT5 account password
            server: MT5 server name
            path: Path to MT5 terminal
        """
        self._login = login
        self._password = password
        self._server = server
        self._path = path
        self._initialized = False
    
    def _ensure_initialized(self) -> bool:
        """Ensure MT5 is initialized."""
        if self._initialized:
            return True
        
        try:
            import MetaTrader5 as mt5
            
            # Initialize MT5
            init_params = {}
            if self._path:
                init_params["path"] = self._path
            if self._login:
                init_params["login"] = self._login
            if self._password:
                init_params["password"] = self._password
            if self._server:
                init_params["server"] = self._server
            
            if not mt5.initialize(**init_params):
                error = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                return False
            
            self._initialized = True
            logger.info("MT5 initialized successfully")
            return True
            
        except ImportError:
            logger.error("MetaTrader5 package not installed")
            return False
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown MT5 connection."""
        if self._initialized:
            try:
                import MetaTrader5 as mt5
                mt5.shutdown()
                self._initialized = False
                logger.info("MT5 shutdown complete")
            except Exception as e:
                logger.error(f"MT5 shutdown error: {e}")
    
    def load(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        bars: int | None = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from MT5.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Timeframe string (e.g., "H1")
            start: Start datetime (UTC)
            end: End datetime (UTC)
            bars: Number of bars to load
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self._ensure_initialized():
            raise RuntimeError("MT5 not initialized")
        
        import MetaTrader5 as mt5
        
        tf = get_mt5_timeframe(timeframe)
        
        # Determine data range
        if bars is not None:
            # Load N bars from current time
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        elif start is not None and end is not None:
            # Load between dates
            rates = mt5.copy_rates_range(symbol, tf, start, end)
        elif start is not None:
            # Load from start to now
            rates = mt5.copy_rates_range(symbol, tf, start, datetime.utcnow())
        else:
            # Default: load last 1000 bars
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, 1000)
        
        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            logger.warning(f"No data returned for {symbol} {timeframe}: {error}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={
            "time": "time",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "tick_volume": "volume",
            "spread": "spread",
            "real_volume": "real_volume",
        })
        
        # Select and order columns
        columns = ["time", "open", "high", "low", "close", "volume"]
        if "spread" in df.columns:
            columns.append("spread")
        
        df = df[columns].copy()
        df = df.set_index("time")
        
        logger.debug(f"Loaded {len(df)} bars for {symbol} {timeframe}")
        return df
    
    def get_current_price(self, symbol: str) -> dict[str, float]:
        """
        Get current bid/ask prices from MT5.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with bid, ask, spread
        """
        if not self._ensure_initialized():
            raise RuntimeError("MT5 not initialized")
        
        import MetaTrader5 as mt5
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            error = mt5.last_error()
            logger.warning(f"Failed to get tick for {symbol}: {error}")
            return {"bid": 0.0, "ask": 0.0, "spread": 0.0}
        
        spread = tick.ask - tick.bid
        
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": spread,
            "time": datetime.utcfromtimestamp(tick.time),
        }
    
    def get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
        """
        Get symbol information from MT5.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with symbol info or None
        """
        if not self._ensure_initialized():
            return None
        
        import MetaTrader5 as mt5
        
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        
        return {
            "symbol": info.name,
            "digits": info.digits,
            "point": info.point,
            "trade_contract_size": info.trade_contract_size,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "spread": info.spread,
            "trade_mode": info.trade_mode,
        }


# =============================================================================
# CSV/Parquet Data Loader (for backtesting)
# =============================================================================

class FileDataLoader(DataLoader):
    """
    Data loader for historical data files.
    
    Supports CSV and Parquet formats for backtesting.
    """
    
    def __init__(self, data_dir: Path | str):
        """
        Initialize file data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self._cache: dict[str, pd.DataFrame] = {}
    
    def _get_file_path(self, symbol: str, timeframe: str) -> Path | None:
        """Get the file path for a symbol/timeframe combination."""
        # Try different naming conventions
        patterns = [
            f"{symbol}_{timeframe}.parquet",
            f"{symbol}_{timeframe}.csv",
            f"{symbol.lower()}_{timeframe.lower()}.parquet",
            f"{symbol.lower()}_{timeframe.lower()}.csv",
        ]
        
        for pattern in patterns:
            path = self.data_dir / pattern
            if path.exists():
                return path
        
        return None
    
    def load(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        bars: int | None = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from file.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start: Start datetime
            end: End datetime
            bars: Number of bars (from end)
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache
        if cache_key not in self._cache:
            file_path = self._get_file_path(symbol, timeframe)
            if file_path is None:
                logger.warning(f"Data file not found for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Load file
            if file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, parse_dates=["time"])
            
            # Ensure datetime index
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], utc=True)
                df = df.set_index("time")
            
            self._cache[cache_key] = df
            logger.debug(f"Loaded {len(df)} bars from {file_path}")
        
        df = self._cache[cache_key].copy()
        
        # Filter by date range
        if start is not None:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        
        # Limit bars
        if bars is not None:
            df = df.tail(bars)
        
        return df
    
    def get_current_price(self, symbol: str) -> dict[str, float]:
        """
        Get current price (last available from data).
        
        For backtesting, returns the last available price.
        """
        # Try to load H1 data
        df = self.load(symbol, "H1", bars=1)
        if df.empty:
            return {"bid": 0.0, "ask": 0.0, "spread": 0.0}
        
        close = df["close"].iloc[-1]
        spread = df.get("spread", pd.Series([0.0])).iloc[-1]
        
        return {
            "bid": close,
            "ask": close + spread,
            "spread": spread,
            "time": df.index[-1],
        }
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()


# =============================================================================
# Multi-Source Data Loader
# =============================================================================

class MultiSourceLoader(DataLoader):
    """
    Data loader that combines multiple sources.
    
    Tries MT5 first, falls back to files for historical data.
    """
    
    def __init__(
        self,
        mt5_loader: MT5DataLoader | None = None,
        file_loader: FileDataLoader | None = None,
    ):
        """
        Initialize multi-source loader.
        
        Args:
            mt5_loader: MT5 data loader
            file_loader: File data loader for backtesting
        """
        self.mt5_loader = mt5_loader
        self.file_loader = file_loader
    
    def load(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        bars: int | None = None,
        source: str = "auto",
    ) -> pd.DataFrame:
        """
        Load OHLCV data from the best available source.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start: Start datetime
            end: End datetime
            bars: Number of bars
            source: Data source ("mt5", "file", "auto")
            
        Returns:
            DataFrame with OHLCV data
        """
        if source == "mt5" and self.mt5_loader:
            return self.mt5_loader.load(symbol, timeframe, start, end, bars)
        
        if source == "file" and self.file_loader:
            return self.file_loader.load(symbol, timeframe, start, end, bars)
        
        # Auto mode: try MT5 first, then file
        if self.mt5_loader:
            try:
                df = self.mt5_loader.load(symbol, timeframe, start, end, bars)
                if not df.empty:
                    return df
            except Exception as e:
                logger.debug(f"MT5 load failed, trying file: {e}")
        
        if self.file_loader:
            return self.file_loader.load(symbol, timeframe, start, end, bars)
        
        logger.warning(f"No data source available for {symbol} {timeframe}")
        return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> dict[str, float]:
        """Get current price from the best available source."""
        if self.mt5_loader:
            try:
                return self.mt5_loader.get_current_price(symbol)
            except Exception:
                pass
        
        if self.file_loader:
            return self.file_loader.get_current_price(symbol)
        
        return {"bid": 0.0, "ask": 0.0, "spread": 0.0}
