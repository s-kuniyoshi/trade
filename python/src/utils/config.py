"""
Configuration management module.

Loads and validates configuration from YAML files using Pydantic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


# =============================================================================
# Base Configuration
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from python/src/utils to project root
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "config").exists():
            return parent
    return current.parent.parent.parent.parent


def load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = get_project_root() / "config" / filename
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# Trading Configuration Models
# =============================================================================

class SymbolConfig(BaseModel):
    """Configuration for a trading symbol."""
    symbol: str
    priority: int = 1
    pip_value: float
    min_lot: float = 0.01
    max_lot: float = 10.0


class SessionConfig(BaseModel):
    """Configuration for a trading session."""
    start: str
    end: str
    enabled: bool = True


class TradingHoursConfig(BaseModel):
    """Configuration for trading hours."""
    enabled: bool = True
    sessions: dict[str, SessionConfig] = Field(default_factory=dict)
    trading_days: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    news_blackout_minutes: int = 30


class NewsFilterConfig(BaseModel):
    """
    Configuration for news-based trading filter.
    
    Blocks trading during high-impact economic events.
    """
    enabled: bool = True
    """Whether news filter is active."""
    
    blackout_before_minutes: int = 30
    """Minutes before event to start blackout."""
    
    blackout_after_minutes: int = 15
    """Minutes after event to end blackout."""
    
    min_impact: str = "high"
    """Minimum impact level to trigger blackout: 'low', 'medium', 'high'."""
    
    default_action: str = "block_entry"
    """Action during blackout: 'block_entry', 'close_positions', 'tighten_stops'."""
    
    refresh_interval_minutes: int = 60
    """How often to refresh calendar data."""
    
    cache_dir: str = "data/cache/news"
    """Directory for caching news data."""
    
    symbol_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """
    Per-symbol configuration overrides.
    
    Example:
        USDJPY:
            min_impact: medium
            action: close_positions
            blackout_before_minutes: 45
    """


class ZeroMQConfig(BaseModel):
    """ZeroMQ communication configuration."""
    request_port: int = 5555
    response_port: int = 5556
    timeout_ms: int = 5000


class HttpConfig(BaseModel):
    """HTTP communication configuration."""
    host: str = "127.0.0.1"
    port: int = 8080
    timeout_ms: int = 5000


class DiscordConfig(BaseModel):
    """Discord notification configuration."""
    enabled: bool = True
    webhook_url: str = ""
    min_severity: str = "warning"  # info, warning, error, critical
    username: str = "Trading Bot"


class CommunicationConfig(BaseModel):
    """Communication configuration."""
    protocol: str = "zeromq"
    zeromq: ZeroMQConfig = Field(default_factory=ZeroMQConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)


class TradingConfig(BaseModel):
    """Main trading configuration."""
    general: dict[str, Any] = Field(default_factory=dict)
    broker: dict[str, Any] = Field(default_factory=dict)
    symbols: dict[str, Any] = Field(default_factory=dict)
    timeframes: dict[str, Any] = Field(default_factory=dict)
    trading_hours: TradingHoursConfig = Field(default_factory=TradingHoursConfig)
    news_filter: NewsFilterConfig = Field(default_factory=NewsFilterConfig)
    signals: dict[str, Any] = Field(default_factory=dict)
    execution: dict[str, Any] = Field(default_factory=dict)
    communication: CommunicationConfig = Field(default_factory=CommunicationConfig)
    logging: dict[str, Any] = Field(default_factory=dict)
    
    @property
    def enabled_symbols(self) -> list[SymbolConfig]:
        """Get list of enabled symbols."""
        symbols_data = self.symbols.get("enabled", [])
        return [SymbolConfig(**s) for s in symbols_data]
    
    @property
    def primary_timeframe(self) -> str:
        """Get primary timeframe."""
        return self.timeframes.get("primary", "H1")
    
    @property
    def higher_timeframes(self) -> list[str]:
        """Get higher timeframes for MTF analysis."""
        return self.timeframes.get("higher", ["H4", "D1"])


# =============================================================================
# Risk Configuration Models
# =============================================================================

class AccountLimitsConfig(BaseModel):
    """Account-level risk limits."""
    max_daily_loss_pct: float = 2.0
    max_weekly_loss_pct: float = 5.0
    max_drawdown_pct: float = 10.0
    max_exposure_pct: float = 50.0
    max_positions: int = 3
    max_positions_per_symbol: int = 1


class TradeRiskConfig(BaseModel):
    """Trade-level risk parameters."""
    risk_per_trade_pct: float = 1.0
    min_lot_size: float = 0.01
    max_lot_size: float = 1.0
    sizing_method: str = "volatility_target"
    volatility_target: dict[str, Any] = Field(default_factory=dict)


class StopLossConfig(BaseModel):
    """Stop loss configuration."""
    enabled: bool = True
    method: str = "atr"
    atr: dict[str, float] = Field(default_factory=lambda: {"multiplier": 2.0, "period": 14})
    fixed_pips: dict[str, int] = Field(default_factory=dict)
    percentage: dict[str, float] = Field(default_factory=dict)
    min_sl_pips: int = 10
    max_sl_pips: int = 100


class TakeProfitConfig(BaseModel):
    """Take profit configuration."""
    enabled: bool = True
    method: str = "risk_reward"
    risk_reward_ratio: float = 1.5
    atr: dict[str, float] = Field(default_factory=lambda: {"multiplier": 3.0, "period": 14})
    fixed_pips: dict[str, int] = Field(default_factory=dict)
    partial_tp: dict[str, Any] = Field(default_factory=dict)


class TrailingStopConfig(BaseModel):
    """Trailing stop configuration."""
    enabled: bool = True
    activation_atr: float = 1.5
    trail_atr: float = 1.0
    step_pips: int = 5


class EmergencyConfig(BaseModel):
    """Emergency stop configuration."""
    max_consecutive_losses: int = 5
    pause_duration_hours: int = 24
    pause_drawdown_pct: float = 5.0
    stop_drawdown_pct: float = 10.0
    pause_daily_loss_pct: float = 2.0
    max_error_rate: int = 10


class RiskConfig(BaseModel):
    """Main risk configuration."""
    account_limits: AccountLimitsConfig = Field(default_factory=AccountLimitsConfig)
    trade_risk: TradeRiskConfig = Field(default_factory=TradeRiskConfig)
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = Field(default_factory=TakeProfitConfig)
    trailing_stop: TrailingStopConfig = Field(default_factory=TrailingStopConfig)
    filters: dict[str, Any] = Field(default_factory=dict)
    emergency: EmergencyConfig = Field(default_factory=EmergencyConfig)
    recovery: dict[str, Any] = Field(default_factory=dict)
    currency_exposure: dict[str, Any] = Field(default_factory=dict)
    alerts: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Model Configuration
# =============================================================================

class LightGBMConfig(BaseModel):
    """LightGBM model configuration."""
    objective: str = "regression"
    metric: str = "rmse"
    boosting_type: str = "gbdt"
    num_leaves: int = 31
    max_depth: int = -1
    min_data_in_leaf: int = 20
    learning_rate: float = 0.05
    n_estimators: int = 1000
    early_stopping_rounds: int = 50
    lambda_l1: float = 0.1
    lambda_l2: float = 0.1
    min_gain_to_split: float = 0.01
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    verbose: int = -1
    n_jobs: int = -1
    random_state: int = 42
    
    def to_params(self) -> dict[str, Any]:
        """Convert to LightGBM parameters dict."""
        return {
            "objective": self.objective,
            "metric": self.metric,
            "boosting_type": self.boosting_type,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "min_gain_to_split": self.min_gain_to_split,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "verbose": self.verbose,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }


class TargetConfig(BaseModel):
    """Prediction target configuration."""
    type: str = "return"
    horizons: list[int] = Field(default_factory=lambda: [3, 6, 12])
    primary_horizon: int = 6
    direction_threshold: float = 0.0001
    quantiles: list[float] = Field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])


class FeatureGroupConfig(BaseModel):
    """Feature group configuration."""
    enabled: bool = True
    features: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=list)


class FeaturesConfig(BaseModel):
    """Features configuration."""
    groups: dict[str, FeatureGroupConfig] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    
    def get_enabled_features(self) -> list[str]:
        """Get all enabled features."""
        features = []
        for group_name, group in self.groups.items():
            if isinstance(group, dict):
                if group.get("enabled", True):
                    features.extend(group.get("features", []))
            elif group.enabled:
                features.extend(group.features)
        return features


class TrainingConfig(BaseModel):
    """Training configuration."""
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    walk_forward: dict[str, Any] = Field(default_factory=dict)
    cv: dict[str, Any] = Field(default_factory=dict)
    hyperopt: dict[str, Any] = Field(default_factory=dict)
    retrain: dict[str, Any] = Field(default_factory=dict)


class CalibrationConfig(BaseModel):
    """Confidence calibration configuration."""
    enabled: bool = True
    method: str = "isotonic"
    min_confidence: float = 0.65
    adjustment: dict[str, float] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """Main model configuration."""
    model: dict[str, Any] = Field(default_factory=dict)
    target: TargetConfig = Field(default_factory=TargetConfig)
    lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)
    xgboost: dict[str, Any] = Field(default_factory=dict)
    ensemble: dict[str, Any] = Field(default_factory=dict)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    registry: dict[str, Any] = Field(default_factory=dict)
    inference: dict[str, Any] = Field(default_factory=dict)
    
    @property
    def active_model(self) -> str:
        """Get active model type."""
        return self.model.get("active", "lightgbm")
    
    @property
    def model_dir(self) -> Path:
        """Get model directory."""
        return Path(self.model.get("model_dir", "models/production/latest"))


# =============================================================================
# Main Configuration Class
# =============================================================================

class Config:
    """
    Main configuration manager.
    
    Loads and provides access to all configuration sections.
    
    Usage:
        config = Config()
        print(config.trading.primary_timeframe)
        print(config.risk.account_limits.max_drawdown_pct)
        print(config.model.lightgbm.learning_rate)
    """
    
    _instance: Config | None = None
    
    def __new__(cls) -> Config:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize configuration."""
        if self._initialized:
            return
        
        self._trading: TradingConfig | None = None
        self._risk: RiskConfig | None = None
        self._model: ModelConfig | None = None
        self._initialized = True
    
    def _load_trading(self) -> TradingConfig:
        """Load trading configuration."""
        data = load_yaml("trading.yaml")
        return TradingConfig(**data)
    
    def _load_risk(self) -> RiskConfig:
        """Load risk configuration."""
        data = load_yaml("risk.yaml")
        return RiskConfig(**data)
    
    def _load_model(self) -> ModelConfig:
        """Load model configuration."""
        data = load_yaml("model.yaml")
        return ModelConfig(**data)
    
    @property
    def trading(self) -> TradingConfig:
        """Get trading configuration."""
        if self._trading is None:
            self._trading = self._load_trading()
        return self._trading
    
    @property
    def risk(self) -> RiskConfig:
        """Get risk configuration."""
        if self._risk is None:
            self._risk = self._load_risk()
        return self._risk
    
    @property
    def model(self) -> ModelConfig:
        """Get model configuration."""
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return get_project_root()
    
    def reload(self) -> None:
        """Reload all configurations."""
        self._trading = None
        self._risk = None
        self._model = None
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None


# Convenience function
def get_config() -> Config:
    """Get the global configuration instance."""
    return Config()
