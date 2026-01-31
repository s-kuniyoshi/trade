"""
通貨ペア別設定（共通モジュール）

バックテストと本番取引で同一の設定を使用するための共通定義。
このファイルを編集すると、両方に反映される。

更新日: 2026-01-31
"""

# =============================================================================
# 通貨ペア別設定
# =============================================================================
# バックテスト結果 (2026-01-31):
# | Symbol | Return  | Trades | PF   | MaxDD  | 推奨閾値 |
# |--------|---------|--------|------|--------|----------|
# | USDCAD | +134.5% | 232    | 1.28 | -18.6% | 0.60     |
# | GBPUSD | +84.1%  | 245    | 1.20 | -31.7% | 0.50     |
# | AUDJPY | +14.3%  | 303    | 1.02 | -50.8% | 0.50     |
# | USDJPY | +5.1%   | 41     | 1.08 | -15.3% | 0.50     |
#
# ボラティリティ特性に基づくレバレッジ設定:
# - 高ボラ（GBPクロス、EURAUD）: 低レバレッジ（15-50倍）
# - 中ボラ（クロス円、AUDUSD）: 中レバレッジ（25-80倍）
# - 低ボラ（メジャー）: 高レバレッジ（30-100倍）

PAIR_CONFIG = {
    # ========== 推奨ペア（PF > 1.0） ==========
    "USDCAD": {
        "enabled": True,              # 取引対象
        "strategy": "ML_Primary",
        "min_confidence": 0.60,       # 高閾値で精度重視（PF 1.28）
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "base_leverage": 30.0,
        "max_leverage": 100.0,
        "spread_pips": 1.5,
        "pip_value": 0.0001,
        "max_spread_pips": 2.5,
    },
    "GBPUSD": {
        "enabled": True,
        "strategy": "ML_Primary",
        "min_confidence": 0.50,       # 低閾値で取引数確保（PF 1.20）
        "adx_threshold": 12.0,        # 低めでエントリー機会確保
        "sma_atr_threshold": 0.2,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "base_leverage": 25.0,
        "max_leverage": 80.0,
        "spread_pips": 1.5,
        "pip_value": 0.0001,
        "max_spread_pips": 2.5,
    },
    "USDJPY": {
        "enabled": True,
        "strategy": "ML_Primary",
        "min_confidence": 0.50,       # PF 1.08
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": True,            # ロングのみ（ショートにエッジなし）
        "base_leverage": 30.0,
        "max_leverage": 100.0,
        "spread_pips": 1.2,
        "pip_value": 0.01,
        "max_spread_pips": 2.0,
    },
    "AUDJPY": {
        "enabled": True,
        "strategy": "ML_Primary",
        "min_confidence": 0.50,       # PF 1.02（ボーダーライン）
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "base_leverage": 25.0,
        "max_leverage": 80.0,
        "spread_pips": 1.8,
        "pip_value": 0.01,
        "max_spread_pips": 3.0,
    },
    
    # ========== 非推奨ペア（PF < 1.0）==========
    "EURUSD": {
        "enabled": False,             # PF 0.95
        "strategy": "ML_Primary",
        "min_confidence": 0.55,
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "base_leverage": 30.0,
        "max_leverage": 100.0,
        "spread_pips": 1.0,
        "pip_value": 0.0001,
        "max_spread_pips": 2.0,
    },
    "USDCHF": {
        "enabled": False,             # PF 0.75
        "strategy": "ML_Primary",
        "min_confidence": 0.55,
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "base_leverage": 30.0,
        "max_leverage": 100.0,
        "spread_pips": 1.5,
        "pip_value": 0.0001,
        "max_spread_pips": 2.5,
    },
    "AUDUSD": {
        "enabled": False,             # PF 0.64
        "strategy": "ML_Primary",
        "min_confidence": 0.55,
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "base_leverage": 25.0,
        "max_leverage": 80.0,
        "spread_pips": 1.2,
        "pip_value": 0.0001,
        "max_spread_pips": 2.0,
    },
    "EURJPY": {
        "enabled": False,             # PF 0.87
        "strategy": "ML_Primary",
        "min_confidence": 0.55,
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "base_leverage": 25.0,
        "max_leverage": 80.0,
        "spread_pips": 1.5,
        "pip_value": 0.01,
        "max_spread_pips": 2.5,
    },
    "GBPJPY": {
        "enabled": False,             # PF 0.75
        "strategy": "ML_Primary",
        "min_confidence": 0.55,
        "adx_threshold": 18.0,        # 強トレンドのみ
        "sma_atr_threshold": 0.4,
        "tp_atr_mult": 3.0,           # 高ボラのため広め
        "sl_atr_mult": 2.0,
        "long_only": False,
        "base_leverage": 15.0,        # 高ボラのため低レバ
        "max_leverage": 50.0,
        "spread_pips": 2.5,
        "pip_value": 0.01,
        "max_spread_pips": 4.0,
    },
    "CADJPY": {
        "enabled": False,             # PF 0.74
        "strategy": "ML_Primary",
        "min_confidence": 0.55,
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "base_leverage": 25.0,
        "max_leverage": 80.0,
        "spread_pips": 2.0,
        "pip_value": 0.01,
        "max_spread_pips": 3.0,
    },
    "EURGBP": {
        "enabled": False,             # PF 0.73
        "strategy": "ML_Primary",
        "min_confidence": 0.55,
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "base_leverage": 25.0,
        "max_leverage": 80.0,
        "spread_pips": 1.5,
        "pip_value": 0.0001,
        "max_spread_pips": 2.5,
    },
    "EURAUD": {
        "enabled": False,             # PF 0.74
        "strategy": "ML_Primary",
        "min_confidence": 0.55,
        "adx_threshold": 18.0,
        "sma_atr_threshold": 0.4,
        "tp_atr_mult": 3.0,           # 高ボラのため広め
        "sl_atr_mult": 2.0,
        "long_only": False,
        "base_leverage": 15.0,        # 高ボラのため低レバ
        "max_leverage": 50.0,
        "spread_pips": 2.5,
        "pip_value": 0.0001,
        "max_spread_pips": 4.0,
    },
}

# デフォルト設定（未定義ペア用）
DEFAULT_PAIR_CONFIG = {
    "enabled": False,
    "strategy": "ML_Primary",
    "min_confidence": 0.55,
    "adx_threshold": 15.0,
    "sma_atr_threshold": 0.3,
    "tp_atr_mult": 2.5,
    "sl_atr_mult": 1.5,
    "long_only": False,
    "base_leverage": 25.0,
    "max_leverage": 80.0,
    "spread_pips": 2.0,
    "pip_value": 0.0001,
    "max_spread_pips": 3.0,
}


def get_pair_config(symbol: str) -> dict:
    """通貨ペアの設定を取得（未定義の場合はデフォルト）"""
    return PAIR_CONFIG.get(symbol, DEFAULT_PAIR_CONFIG)


def get_enabled_symbols() -> list[str]:
    """有効な通貨ペアのリストを取得"""
    return [symbol for symbol, cfg in PAIR_CONFIG.items() if cfg.get("enabled", False)]


def get_all_symbols() -> list[str]:
    """全通貨ペアのリストを取得"""
    return list(PAIR_CONFIG.keys())


# =============================================================================
# 共通パラメータ
# =============================================================================

COMMON_CONFIG = {
    # 時間足
    "timeframe": "M30",
    
    # Walk-Forward設定
    "train_years": 2,
    "step_months": 6,
    
    # Triple-barrier設定
    "max_hold_bars": 36,
    
    # リスク管理
    "risk_per_trade": 0.03,         # 3%
    "max_drawdown_pct": 0.25,       # 25%
    "consecutive_loss_limit": 5,
    "vol_scale_threshold": 1.5,
    "leverage_confidence_threshold": 0.65,
    
    # スリッページ
    "slippage_pips": 0.2,
}
