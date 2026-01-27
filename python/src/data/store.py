"""
Data storage module for persisting market data.

Supports Parquet and SQLite storage backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger("data.store")


# =============================================================================
# Abstract Data Store
# =============================================================================

class DataStore(ABC):
    """Abstract base class for data storage."""
    
    @abstractmethod
    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        append: bool = True,
    ) -> None:
        """
        Save OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe string
            append: Whether to append to existing data
        """
        pass
    
    @abstractmethod
    def load(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start: Start datetime
            end: End datetime
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def exists(self, symbol: str, timeframe: str) -> bool:
        """Check if data exists for symbol/timeframe."""
        pass
    
    @abstractmethod
    def get_date_range(
        self,
        symbol: str,
        timeframe: str,
    ) -> tuple[datetime | None, datetime | None]:
        """Get the date range of stored data."""
        pass


# =============================================================================
# Parquet Data Store
# =============================================================================

class ParquetStore(DataStore):
    """
    Parquet-based data store.
    
    Stores each symbol/timeframe combination as a separate Parquet file.
    Efficient for large datasets and supports compression.
    """
    
    def __init__(
        self,
        data_dir: Path | str,
        compression: str = "snappy",
    ):
        """
        Initialize Parquet store.
        
        Args:
            data_dir: Directory for storing Parquet files
            compression: Compression algorithm (snappy, gzip, zstd)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self._cache: dict[str, pd.DataFrame] = {}
    
    def _get_file_path(self, symbol: str, timeframe: str) -> Path:
        """Get the file path for a symbol/timeframe combination."""
        return self.data_dir / f"{symbol}_{timeframe}.parquet"
    
    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        append: bool = True,
    ) -> None:
        """
        Save OHLCV data to Parquet file.
        
        Args:
            df: DataFrame with OHLCV data (must have datetime index)
            symbol: Trading symbol
            timeframe: Timeframe string
            append: Whether to append to existing data
        """
        if df.empty:
            logger.warning(f"Empty DataFrame, nothing to save for {symbol} {timeframe}")
            return
        
        file_path = self._get_file_path(symbol, timeframe)
        cache_key = f"{symbol}_{timeframe}"
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df = df.set_index("time")
            else:
                raise ValueError("DataFrame must have datetime index or 'time' column")
        
        # Ensure UTC timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif df.index.tz != "UTC":
            df.index = df.index.tz_convert("UTC")
        
        if append and file_path.exists():
            # Load existing data
            existing_df = pd.read_parquet(file_path)
            if not isinstance(existing_df.index, pd.DatetimeIndex):
                if "time" in existing_df.columns:
                    existing_df = existing_df.set_index("time")
            
            # Ensure UTC timezone on existing
            if existing_df.index.tz is None:
                existing_df.index = existing_df.index.tz_localize("UTC")
            
            # Combine and deduplicate
            combined = pd.concat([existing_df, df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            df = combined
        
        # Save to Parquet
        df.to_parquet(file_path, compression=self.compression)
        
        # Update cache
        self._cache[cache_key] = df
        
        logger.debug(f"Saved {len(df)} bars to {file_path}")
    
    def load(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from Parquet file.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start: Start datetime
            end: End datetime
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache
        if cache_key not in self._cache:
            file_path = self._get_file_path(symbol, timeframe)
            
            if not file_path.exists():
                logger.debug(f"No data file found for {symbol} {timeframe}")
                return pd.DataFrame()
            
            df = pd.read_parquet(file_path)
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if "time" in df.columns:
                    df = df.set_index("time")
            
            # Ensure UTC timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            
            self._cache[cache_key] = df
        
        df = self._cache[cache_key].copy()
        
        # Filter by date range
        if start is not None:
            start_ts = pd.Timestamp(start, tz="UTC")
            df = df[df.index >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end, tz="UTC")
            df = df[df.index <= end_ts]
        
        return df
    
    def exists(self, symbol: str, timeframe: str) -> bool:
        """Check if data exists for symbol/timeframe."""
        return self._get_file_path(symbol, timeframe).exists()
    
    def get_date_range(
        self,
        symbol: str,
        timeframe: str,
    ) -> tuple[datetime | None, datetime | None]:
        """Get the date range of stored data."""
        df = self.load(symbol, timeframe)
        if df.empty:
            return None, None
        return df.index.min().to_pydatetime(), df.index.max().to_pydatetime()
    
    def list_available_data(self) -> list[dict[str, str]]:
        """List all available symbol/timeframe combinations."""
        available = []
        for file_path in self.data_dir.glob("*.parquet"):
            parts = file_path.stem.split("_")
            if len(parts) >= 2:
                symbol = "_".join(parts[:-1])
                timeframe = parts[-1]
                available.append({"symbol": symbol, "timeframe": timeframe})
        return available
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
    
    def delete(self, symbol: str, timeframe: str) -> bool:
        """Delete data for a symbol/timeframe."""
        file_path = self._get_file_path(symbol, timeframe)
        cache_key = f"{symbol}_{timeframe}"
        
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted data for {symbol} {timeframe}")
            return True
        return False


# =============================================================================
# SQLite Data Store
# =============================================================================

class SQLiteStore(DataStore):
    """
    SQLite-based data store.
    
    Stores all data in a single SQLite database.
    Good for smaller datasets and simpler queries.
    """
    
    def __init__(self, db_path: Path | str):
        """
        Initialize SQLite store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database tables."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create OHLCV table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                time TIMESTAMP NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL,
                spread REAL,
                PRIMARY KEY (symbol, timeframe, time)
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf_time
            ON ohlcv (symbol, timeframe, time)
        """)
        
        conn.commit()
        conn.close()
    
    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        append: bool = True,
    ) -> None:
        """
        Save OHLCV data to SQLite.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe string
            append: Whether to append (always replaces duplicates)
        """
        if df.empty:
            return
        
        import sqlite3
        
        # Prepare data
        df = df.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "time"})
        
        df["symbol"] = symbol
        df["timeframe"] = timeframe
        
        # Ensure required columns
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if "volume" not in df.columns:
            df["volume"] = 0
        if "spread" not in df.columns:
            df["spread"] = None
        
        # Select columns in order
        columns = ["symbol", "timeframe", "time", "open", "high", "low", "close", "volume", "spread"]
        df = df[columns]
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        
        if not append:
            # Delete existing data
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM ohlcv WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe)
            )
        
        # Insert or replace data
        df.to_sql("ohlcv", conn, if_exists="append", index=False, method="multi")
        
        # Remove duplicates (keep latest)
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM ohlcv
            WHERE rowid NOT IN (
                SELECT MAX(rowid)
                FROM ohlcv
                GROUP BY symbol, timeframe, time
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Saved {len(df)} bars to SQLite for {symbol} {timeframe}")
    
    def load(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from SQLite.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start: Start datetime
            end: End datetime
            
        Returns:
            DataFrame with OHLCV data
        """
        import sqlite3
        
        query = "SELECT time, open, high, low, close, volume, spread FROM ohlcv WHERE symbol = ? AND timeframe = ?"
        params: list[Any] = [symbol, timeframe]
        
        if start is not None:
            query += " AND time >= ?"
            params.append(start.isoformat())
        if end is not None:
            query += " AND time <= ?"
            params.append(end.isoformat())
        
        query += " ORDER BY time"
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["time"])
        conn.close()
        
        if df.empty:
            return df
        
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
        
        return df
    
    def exists(self, symbol: str, timeframe: str) -> bool:
        """Check if data exists for symbol/timeframe."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND timeframe = ? LIMIT 1",
            (symbol, timeframe)
        )
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def get_date_range(
        self,
        symbol: str,
        timeframe: str,
    ) -> tuple[datetime | None, datetime | None]:
        """Get the date range of stored data."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MIN(time), MAX(time) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result[0] is None:
            return None, None
        
        return (
            pd.Timestamp(result[0]).to_pydatetime(),
            pd.Timestamp(result[1]).to_pydatetime()
        )


# =============================================================================
# Trade Log Store
# =============================================================================

class TradeLogStore:
    """
    Store for trade execution logs.
    
    Persists trade history for analysis and reporting.
    """
    
    def __init__(self, db_path: Path | str):
        """
        Initialize trade log store.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database tables."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket INTEGER,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                lots REAL NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                entry_price REAL NOT NULL,
                exit_time TIMESTAMP,
                exit_price REAL,
                sl REAL,
                tp REAL,
                pnl REAL,
                pnl_pips REAL,
                commission REAL,
                swap REAL,
                status TEXT DEFAULT 'open',
                strategy TEXT,
                model_version TEXT,
                confidence REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_symbol_time
            ON trades (symbol, entry_time)
        """)
        
        conn.commit()
        conn.close()
    
    def log_trade_open(
        self,
        symbol: str,
        direction: str,
        lots: float,
        entry_price: float,
        sl: float | None = None,
        tp: float | None = None,
        ticket: int | None = None,
        strategy: str | None = None,
        model_version: str | None = None,
        confidence: float | None = None,
    ) -> int:
        """
        Log a trade opening.
        
        Returns:
            Trade ID
        """
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades (
                ticket, symbol, direction, lots, entry_time, entry_price,
                sl, tp, status, strategy, model_version, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
        """, (
            ticket, symbol, direction, lots, datetime.utcnow(), entry_price,
            sl, tp, strategy, model_version, confidence
        ))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return trade_id
    
    def log_trade_close(
        self,
        trade_id: int,
        exit_price: float,
        pnl: float,
        pnl_pips: float,
        commission: float = 0.0,
        swap: float = 0.0,
        notes: str | None = None,
    ) -> None:
        """Log a trade closing."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE trades SET
                exit_time = ?,
                exit_price = ?,
                pnl = ?,
                pnl_pips = ?,
                commission = ?,
                swap = ?,
                status = 'closed',
                notes = ?
            WHERE id = ?
        """, (
            datetime.utcnow(), exit_price, pnl, pnl_pips,
            commission, swap, notes, trade_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_trades(
        self,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        """Get trades matching criteria."""
        import sqlite3
        
        query = "SELECT * FROM trades WHERE 1=1"
        params: list[Any] = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start:
            query += " AND entry_time >= ?"
            params.append(start)
        if end:
            query += " AND entry_time <= ?"
            params.append(end)
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY entry_time DESC"
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn, params=params, parse_dates=["entry_time", "exit_time"])
        conn.close()
        
        return df
    
    def get_open_trades(self) -> pd.DataFrame:
        """Get all open trades."""
        return self.get_trades(status="open")
    
    def get_performance_summary(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        """Get performance summary statistics."""
        df = self.get_trades(start=start, end=end, status="closed")
        
        if df.empty:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
            }
        
        winners = df[df["pnl"] > 0]
        losers = df[df["pnl"] <= 0]
        
        total_profit = winners["pnl"].sum() if not winners.empty else 0.0
        total_loss = abs(losers["pnl"].sum()) if not losers.empty else 0.0
        
        return {
            "total_trades": len(df),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(df) if len(df) > 0 else 0.0,
            "total_pnl": df["pnl"].sum(),
            "avg_pnl": df["pnl"].mean(),
            "avg_win": winners["pnl"].mean() if not winners.empty else 0.0,
            "avg_loss": losers["pnl"].mean() if not losers.empty else 0.0,
            "profit_factor": total_profit / total_loss if total_loss > 0 else float("inf"),
            "max_pnl": df["pnl"].max(),
            "min_pnl": df["pnl"].min(),
        }
