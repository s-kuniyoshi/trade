"""
Logging module for the FX Trading System.

Provides structured logging with file rotation and rich console output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from .config import get_config, get_project_root


# =============================================================================
# Logger Configuration
# =============================================================================

# Remove default handler
logger.remove()

# Global logger instance
_logger_configured = False


def setup_logger(
    name: str = "fx_trading",
    level: str = "INFO",
    log_dir: Path | str | None = None,
    console: bool = True,
    file: bool = True,
    rotation: str = "10 MB",
    retention: str = "30 days",
    compression: str = "zip",
) -> None:
    """
    Set up the logging system.
    
    Args:
        name: Logger name (used for log file naming)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        console: Enable console output
        file: Enable file output
        rotation: Log rotation size or time
        retention: Log retention period
        compression: Compression format for rotated logs
    """
    global _logger_configured
    
    if _logger_configured:
        return
    
    # Determine log directory
    if log_dir is None:
        log_dir = get_project_root() / "logs"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Console handler with rich formatting
    if console:
        logger.add(
            sys.stderr,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level=level,
            colorize=True,
        )
    
    # File handler for general logs
    if file:
        logger.add(
            log_dir / f"{name}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression=compression,
            encoding="utf-8",
        )
        
        # Separate file for errors
        logger.add(
            log_dir / f"{name}_error.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation=rotation,
            retention=retention,
            compression=compression,
            encoding="utf-8",
        )
    
    _logger_configured = True
    logger.info(f"Logger initialized: {name} (level={level})")


def get_logger(name: str | None = None) -> "logger":
    """
    Get a logger instance.
    
    Args:
        name: Optional name to bind to the logger
        
    Returns:
        Logger instance
    """
    global _logger_configured
    
    if not _logger_configured:
        # Auto-configure with defaults
        try:
            config = get_config()
            log_config = config.trading.logging
            setup_logger(
                level=log_config.get("level", "INFO"),
            )
        except Exception:
            # Fallback to basic configuration
            setup_logger()
    
    if name:
        return logger.bind(name=name)
    return logger


# =============================================================================
# Specialized Loggers
# =============================================================================

class TradeLogger:
    """
    Specialized logger for trade events.
    
    Provides structured logging for trades with additional context.
    """
    
    def __init__(self, log_dir: Path | str | None = None):
        """Initialize trade logger."""
        if log_dir is None:
            log_dir = get_project_root() / "logs" / "trading"
        else:
            log_dir = Path(log_dir)
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self._trade_log = log_dir / "trades.log"
        self._signal_log = log_dir / "signals.log"
        self._logger = get_logger("trade")
    
    def log_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        expected_return: float,
        features: dict[str, Any] | None = None,
    ) -> None:
        """Log a trading signal."""
        self._logger.info(
            f"SIGNAL | {symbol} | {direction.upper()} | "
            f"conf={confidence:.3f} | exp_ret={expected_return:.5f}"
        )
        
        # Write to signal log file
        with open(self._signal_log, "a", encoding="utf-8") as f:
            import json
            from datetime import datetime
            
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "direction": direction,
                "confidence": confidence,
                "expected_return": expected_return,
                "features": features or {},
            }
            f.write(json.dumps(record) + "\n")
    
    def log_trade_open(
        self,
        symbol: str,
        direction: str,
        lots: float,
        entry_price: float,
        sl: float,
        tp: float,
        ticket: int | None = None,
    ) -> None:
        """Log trade opening."""
        self._logger.info(
            f"OPEN | {symbol} | {direction.upper()} | "
            f"lots={lots:.2f} | entry={entry_price:.5f} | "
            f"SL={sl:.5f} | TP={tp:.5f} | ticket={ticket}"
        )
        
        # Write to trade log
        with open(self._trade_log, "a", encoding="utf-8") as f:
            import json
            from datetime import datetime
            
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "event": "open",
                "symbol": symbol,
                "direction": direction,
                "lots": lots,
                "entry_price": entry_price,
                "sl": sl,
                "tp": tp,
                "ticket": ticket,
            }
            f.write(json.dumps(record) + "\n")
    
    def log_trade_close(
        self,
        symbol: str,
        direction: str,
        lots: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pips: float,
        ticket: int | None = None,
        reason: str = "manual",
    ) -> None:
        """Log trade closing."""
        pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
        pips_str = f"+{pnl_pips:.1f}" if pnl_pips >= 0 else f"{pnl_pips:.1f}"
        
        self._logger.info(
            f"CLOSE | {symbol} | {direction.upper()} | "
            f"PnL={pnl_str} ({pips_str} pips) | "
            f"entry={entry_price:.5f} | exit={exit_price:.5f} | "
            f"reason={reason}"
        )
        
        # Write to trade log
        with open(self._trade_log, "a", encoding="utf-8") as f:
            import json
            from datetime import datetime
            
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "event": "close",
                "symbol": symbol,
                "direction": direction,
                "lots": lots,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pips": pnl_pips,
                "ticket": ticket,
                "reason": reason,
            }
            f.write(json.dumps(record) + "\n")
    
    def log_error(self, message: str, error: Exception | None = None) -> None:
        """Log an error."""
        if error:
            self._logger.error(f"ERROR | {message} | {type(error).__name__}: {error}")
        else:
            self._logger.error(f"ERROR | {message}")


class PerformanceLogger:
    """
    Logger for performance metrics and monitoring.
    """
    
    def __init__(self, log_dir: Path | str | None = None):
        """Initialize performance logger."""
        if log_dir is None:
            log_dir = get_project_root() / "logs" / "monitoring"
        else:
            log_dir = Path(log_dir)
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self._metrics_log = log_dir / "metrics.log"
        self._logger = get_logger("performance")
    
    def log_daily_summary(
        self,
        date: str,
        total_trades: int,
        win_rate: float,
        pnl: float,
        drawdown: float,
        sharpe: float | None = None,
    ) -> None:
        """Log daily performance summary."""
        self._logger.info(
            f"DAILY | {date} | trades={total_trades} | "
            f"win_rate={win_rate:.1%} | PnL={pnl:+.2f} | "
            f"DD={drawdown:.2%} | sharpe={sharpe:.2f if sharpe else 'N/A'}"
        )
        
        # Write to metrics log
        with open(self._metrics_log, "a", encoding="utf-8") as f:
            import json
            
            record = {
                "date": date,
                "type": "daily_summary",
                "total_trades": total_trades,
                "win_rate": win_rate,
                "pnl": pnl,
                "drawdown": drawdown,
                "sharpe": sharpe,
            }
            f.write(json.dumps(record) + "\n")
    
    def log_model_metrics(
        self,
        model_version: str,
        accuracy: float,
        profit_factor: float,
        sharpe_ratio: float,
        max_drawdown: float,
    ) -> None:
        """Log model performance metrics."""
        self._logger.info(
            f"MODEL | v{model_version} | acc={accuracy:.3f} | "
            f"PF={profit_factor:.2f} | sharpe={sharpe_ratio:.2f} | "
            f"maxDD={max_drawdown:.2%}"
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def log_debug(message: str, **kwargs: Any) -> None:
    """Log debug message."""
    get_logger().debug(message, **kwargs)


def log_info(message: str, **kwargs: Any) -> None:
    """Log info message."""
    get_logger().info(message, **kwargs)


def log_warning(message: str, **kwargs: Any) -> None:
    """Log warning message."""
    get_logger().warning(message, **kwargs)


def log_error(message: str, **kwargs: Any) -> None:
    """Log error message."""
    get_logger().error(message, **kwargs)


def log_critical(message: str, **kwargs: Any) -> None:
    """Log critical message."""
    get_logger().critical(message, **kwargs)
