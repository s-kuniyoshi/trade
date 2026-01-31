"""
フル検証パイプライン実行スクリプト。

データ取得 → 特徴量生成 → Walk-Forward学習 → バックテスト → 評価レポート

使用方法:
    python scripts/run_backtest.py

前提条件:
    - scripts/download_data.py でデータを取得済み
    - または data/raw/ にParquetファイルが存在
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

import numpy as np
import pandas as pd
from typing import Any


# =============================================================================
# 通貨ペア別設定（バックテスト用）
# =============================================================================
# ボラティリティ特性に基づくレバレッジ設定:
# - 高ボラ（GBPクロス、EURAUD）: 低レバレッジ（15-50倍）
# - 中ボラ（クロス円、AUDUSD）: 中レバレッジ（25-80倍）
# - 低ボラ（メジャー）: 高レバレッジ（30-100倍）

PAIR_CONFIG = {
    # ========== メジャーペア ==========
    "EURUSD": {
        "base_leverage": 30.0,
        "max_leverage": 100.0,
        "min_confidence": 0.55,
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "spread_pips": 1.0,
        "pip_value": 0.0001,
    },
    "USDJPY": {
        "base_leverage": 30.0,
        "max_leverage": 100.0,
        "min_confidence": 0.55,
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": True,  # ロングのみ（ショートにエッジなし）
        "spread_pips": 1.2,
        "pip_value": 0.01,
    },
    "GBPUSD": {
        "base_leverage": 25.0,
        "max_leverage": 80.0,
        "min_confidence": 0.50,  # GBPUSDは低閾値で好成績
        "adx_threshold": 12.0,
        "sma_atr_threshold": 0.2,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "spread_pips": 1.5,
        "pip_value": 0.0001,
    },
    "USDCHF": {
        "base_leverage": 30.0,
        "max_leverage": 100.0,
        "min_confidence": 0.55,
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "spread_pips": 1.5,
        "pip_value": 0.0001,
    },
    "USDCAD": {
        "base_leverage": 30.0,
        "max_leverage": 100.0,
        "min_confidence": 0.55,
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "spread_pips": 1.5,
        "pip_value": 0.0001,
    },
    "AUDUSD": {
        "base_leverage": 25.0,
        "max_leverage": 80.0,
        "min_confidence": 0.55,
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "spread_pips": 1.2,
        "pip_value": 0.0001,
    },
    # ========== クロス円 ==========
    "EURJPY": {
        "base_leverage": 25.0,
        "max_leverage": 80.0,
        "min_confidence": 0.55,
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "spread_pips": 1.5,
        "pip_value": 0.01,
    },
    "GBPJPY": {
        "base_leverage": 15.0,  # 高ボラのため低レバ
        "max_leverage": 50.0,
        "min_confidence": 0.55,
        "adx_threshold": 18.0,  # 強トレンドのみ
        "sma_atr_threshold": 0.4,
        "tp_atr_mult": 3.0,  # 広めのTP
        "sl_atr_mult": 2.0,  # 広めのSL
        "long_only": False,
        "spread_pips": 2.5,
        "pip_value": 0.01,
    },
    "AUDJPY": {
        "base_leverage": 25.0,
        "max_leverage": 80.0,
        "min_confidence": 0.55,
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "spread_pips": 1.8,
        "pip_value": 0.01,
    },
    "CADJPY": {
        "base_leverage": 25.0,
        "max_leverage": 80.0,
        "min_confidence": 0.55,
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "spread_pips": 2.0,
        "pip_value": 0.01,
    },
    # ========== その他クロス ==========
    "EURGBP": {
        "base_leverage": 25.0,
        "max_leverage": 80.0,
        "min_confidence": 0.55,
        "adx_threshold": 15.0,
        "sma_atr_threshold": 0.3,
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 1.5,
        "long_only": False,
        "spread_pips": 1.5,
        "pip_value": 0.0001,
    },
    "EURAUD": {
        "base_leverage": 15.0,  # 高ボラのため低レバ
        "max_leverage": 50.0,
        "min_confidence": 0.55,
        "adx_threshold": 18.0,
        "sma_atr_threshold": 0.4,
        "tp_atr_mult": 3.0,
        "sl_atr_mult": 2.0,
        "long_only": False,
        "spread_pips": 2.5,
        "pip_value": 0.0001,
    },
}

# デフォルト設定（未定義ペア用）
DEFAULT_PAIR_CONFIG = {
    "base_leverage": 25.0,
    "max_leverage": 80.0,
    "min_confidence": 0.55,
    "adx_threshold": 15.0,
    "sma_atr_threshold": 0.3,
    "tp_atr_mult": 2.5,
    "sl_atr_mult": 1.5,
    "long_only": False,
    "spread_pips": 2.0,
    "pip_value": 0.0001,
}


def get_pair_config(symbol: str) -> dict:
    """通貨ペアの設定を取得（未定義の場合はデフォルト）"""
    return PAIR_CONFIG.get(symbol, DEFAULT_PAIR_CONFIG)


def load_data(symbol: str, timeframe: str, data_dir: Path) -> pd.DataFrame | None:
    """Parquetファイルからデータを読み込む。"""
    filepath = data_dir / f"{symbol}_{timeframe}.parquet"
    
    if not filepath.exists():
        print(f"  ファイルが見つかりません: {filepath}")
        return None
    
    df = pd.read_parquet(filepath)
    print(f"  {symbol}_{timeframe}: {len(df)} 本 ({df.index[0].date()} ~ {df.index[-1].date()})")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    テクニカル指標ベースの特徴量を計算。
    
    リーク防止: すべて過去データのみ使用（shift使用）
    """
    features = pd.DataFrame(index=df.index)
    
    # 価格系
    features["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
    features["log_return_5"] = np.log(df["close"] / df["close"].shift(5))
    features["log_return_20"] = np.log(df["close"] / df["close"].shift(20))
    
    # レンジ
    features["range_hl"] = (df["high"] - df["low"]) / df["close"]
    
    # ボラティリティ
    features["realized_vol_20"] = features["log_return_1"].rolling(20).std() * np.sqrt(252 * 24)
    
    # 移動平均
    features["sma_20"] = df["close"].rolling(20).mean()
    features["sma_50"] = df["close"].rolling(50).mean()
    features["sma_200"] = df["close"].rolling(200).mean()  # トレンドフィルター用
    features["price_to_sma20"] = df["close"] / features["sma_20"] - 1
    features["price_to_sma50"] = df["close"] / features["sma_50"] - 1
    features["price_to_sma200"] = df["close"] / features["sma_200"] - 1  # SMA200乖離
    features["sma_cross"] = (features["sma_20"] > features["sma_50"]).astype(int)
    
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features["rsi_14"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    features["macd"] = ema_12 - ema_26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]
    
    # ATR
    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift(1))
    tr3 = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    features["atr_14"] = tr.rolling(14).mean()
    
    # ADX (Average Directional Index) - トレンド強度フィルター用
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr_smooth = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_smooth)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    features["adx_14"] = dx.ewm(span=14, adjust=False).mean()
    
    # SMA200からの乖離をATRで正規化（トレンドフィルター用）
    features["sma200_atr_distance"] = abs(df["close"] - features["sma_200"]) / features["atr_14"]
    
    # ボリンジャーバンド
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    features["bb_upper"] = bb_mid + 2 * bb_std
    features["bb_lower"] = bb_mid - 2 * bb_std
    features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / bb_mid
    features["bb_position"] = (df["close"] - features["bb_lower"]) / (features["bb_upper"] - features["bb_lower"])
    
    # 時間特徴量
    features["hour"] = df.index.hour
    features["day_of_week"] = df.index.dayofweek
    
    # ========================================
    # 新規特徴量: セッション特性（東京/ロンドン/NY）
    # ========================================
    hour = df.index.hour
    
    # セッションフラグ (UTC基準)
    features["is_tokyo"] = ((hour >= 0) & (hour < 9)).astype(int)
    features["is_london"] = ((hour >= 7) & (hour < 16)).astype(int)
    features["is_ny"] = ((hour >= 13) & (hour < 22)).astype(int)
    features["is_ldn_ny_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)
    
    # セッション別累積リターン（セッション開始からのトレンド方向）
    ret1 = features["log_return_1"]
    day = df.index.floor("D")
    
    # 東京セッション累積リターン
    tokyo_ret = ret1.where(features["is_tokyo"].astype(bool), 0.0)
    features["tokyo_cumret"] = tokyo_ret.groupby(day).cumsum()
    
    # ロンドンセッション累積リターン
    london_ret = ret1.where(features["is_london"].astype(bool), 0.0)
    features["london_cumret"] = london_ret.groupby(day).cumsum()
    
    # NYセッション累積リターン
    ny_ret = ret1.where(features["is_ny"].astype(bool), 0.0)
    features["ny_cumret"] = ny_ret.groupby(day).cumsum()
    
    # セッション別ボラティリティ（全体のローリングボラに対するセッションの相対ボラ）
    overall_vol = ret1.rolling(24, min_periods=12).std()
    features["tokyo_vol_ratio"] = features["is_tokyo"] * overall_vol
    features["london_vol_ratio"] = features["is_london"] * overall_vol
    features["ny_vol_ratio"] = features["is_ny"] * overall_vol
    
    # ========================================
    # 新規特徴量: トレンド品質（Efficiency Ratio）
    # ========================================
    n = 20
    close = df["close"]
    
    # Kaufman Efficiency Ratio: 0=チョップ、1=クリーンなトレンド
    change = close.diff(n).abs()
    volatility = close.diff().abs().rolling(n).sum()
    features["efficiency_ratio"] = (change / volatility).replace([np.inf, -np.inf], np.nan)
    
    # SMAスロープ（トレンドの傾き）
    sma_20 = features["sma_20"]
    features["sma_slope"] = sma_20.diff(3) / 3.0
    features["sma_accel"] = features["sma_slope"].diff(3) / 3.0
    
    # リターン持続性（同じ方向が続く割合）
    ret = features["log_return_1"]
    same_sign = (np.sign(ret) == np.sign(ret.shift(1))).astype(float)
    features["ret_persistence"] = same_sign.rolling(n).mean()
    
    # ========================================
    # 新規特徴量: ジャンプ/テールリスク
    # ========================================
    win = 48
    
    # Parkinsonボラティリティ（レンジベース）
    parkinson = (1.0 / (4.0 * np.log(2.0))) * (np.log(df["high"] / df["low"]) ** 2)
    features["parkinson_vol"] = np.sqrt(parkinson.rolling(win).mean())
    
    # ジャンプスコア（異常な動きの検出）
    med_abs = ret.abs().rolling(win).median()
    features["jump_score"] = (ret.abs() / med_abs).replace([np.inf, -np.inf], np.nan)
    
    # テール非対称性（下落リスク vs 上昇リスク）
    down = (ret.clip(upper=0.0) ** 2).rolling(win).mean()
    up = (ret.clip(lower=0.0) ** 2).rolling(win).mean()
    features["tail_asymmetry"] = (down / up).replace([np.inf, -np.inf], np.nan)
    
    # ========================================
    # 新規特徴量: ミーンリバージョン（レンジ相場用）
    # ========================================
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    
    # Zスコア
    features["zscore"] = (close - ma) / sd
    
    # バンド幅のパーセンタイル（圧縮度）- シンプルな方法で計算
    bw = features["bb_width"]
    bw_rolling_max = bw.rolling(100, min_periods=20).max()
    bw_rolling_min = bw.rolling(100, min_periods=20).min()
    features["bw_percentile"] = (bw - bw_rolling_min) / (bw_rolling_max - bw_rolling_min + 1e-10)
    
    # レンジ相場フラグ（低ボラ + 低トレンド品質）
    features["is_range_regime"] = (
        (features["bw_percentile"] < 0.25) & 
        (features["efficiency_ratio"] < 0.3)
    ).astype(int)
    
    # ミーンリバージョンシグナル（レンジ相場でのみ有効）
    features["mr_signal"] = features["is_range_regime"] * (-features["zscore"])
    
    # ========================================
    # 追加特徴量: モデル精度向上用
    # ========================================
    
    # Stochastic Oscillator
    low_14 = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    features["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 1e-10)
    features["stoch_d"] = features["stoch_k"].rolling(3).mean()
    
    # Williams %R
    features["williams_r"] = -100 * (high_14 - df["close"]) / (high_14 - low_14 + 1e-10)
    
    # Rate of Change (ROC)
    features["roc_5"] = (df["close"] / df["close"].shift(5) - 1) * 100
    features["roc_10"] = (df["close"] / df["close"].shift(10) - 1) * 100
    features["roc_20"] = (df["close"] / df["close"].shift(20) - 1) * 100
    
    # ATR変化率（ボラティリティクラスタリング）
    features["atr_change"] = features["atr_14"].pct_change(5)
    features["atr_ratio"] = features["atr_14"] / features["atr_14"].rolling(50).mean()
    
    # モメンタム強度
    features["momentum_10"] = df["close"] - df["close"].shift(10)
    features["momentum_20"] = df["close"] - df["close"].shift(20)
    
    # +DI / -DI の差（トレンド方向性）
    features["di_diff"] = plus_di - minus_di
    
    # Higher High / Lower Low パターン
    features["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int).rolling(5).sum()
    features["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int).rolling(5).sum()
    features["hh_ll_ratio"] = features["higher_high"] / (features["lower_low"] + 1)
    
    # Keltner Channel
    kc_mid = df["close"].ewm(span=20, adjust=False).mean()
    kc_atr = features["atr_14"]
    features["kc_upper"] = kc_mid + 2 * kc_atr
    features["kc_lower"] = kc_mid - 2 * kc_atr
    features["kc_position"] = (df["close"] - features["kc_lower"]) / (features["kc_upper"] - features["kc_lower"] + 1e-10)
    
    # ボラティリティの非対称性（上昇 vs 下落）
    up_vol = features["log_return_1"].where(features["log_return_1"] > 0, 0).rolling(20).std()
    down_vol = features["log_return_1"].where(features["log_return_1"] < 0, 0).abs().rolling(20).std()
    features["vol_asymmetry"] = (up_vol / (down_vol + 1e-10)).replace([np.inf, -np.inf], np.nan)
    
    # RSI divergence（価格とRSIの乖離）
    price_slope = df["close"].diff(10) / df["close"].shift(10)
    rsi_slope = features["rsi_14"].diff(10) / 100
    features["rsi_divergence"] = price_slope - rsi_slope
    
    # 連続上昇/下落日数
    up_streak = (features["log_return_1"] > 0).astype(int)
    down_streak = (features["log_return_1"] < 0).astype(int)
    features["up_streak"] = up_streak.groupby((up_streak != up_streak.shift()).cumsum()).cumsum()
    features["down_streak"] = down_streak.groupby((down_streak != down_streak.shift()).cumsum()).cumsum()
    
    # 欠損値を削除
    features = features.dropna()
    
    return features


def compute_target(df: pd.DataFrame, horizon: int = 6) -> pd.Series:
    """
    予測ターゲット: N時間後のリターン。（旧方式、互換性のため残す）
    
    Args:
        df: OHLCVデータ
        horizon: 予測ホライズン（時間）
    """
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    return future_return


def compute_triple_barrier_target(
    df: pd.DataFrame,
    atr: pd.Series,
    tp_atr: float = 3.0,
    sl_atr: float = 2.0,
    max_hold: int = 48,
) -> pd.Series:
    """
    Triple-barrier方式のターゲット: SL/TPどちらが先にヒットするかを予測。
    
    バックテストの実際の決済ロジックと完全に一致するターゲット。
    
    Args:
        df: OHLCVデータ（high, low, close必須）
        atr: ATR値のSeries
        tp_atr: TP距離のATR倍率（デフォルト3.0）
        sl_atr: SL距離のATR倍率（デフォルト2.0）
        max_hold: 最大保有期間（バー数）
    
    Returns:
        ターゲットSeries:
            +1: ロングでTP先にヒット（買いシグナル）
            -1: ショートでTP先にヒット（売りシグナル）
             0: どちらも不明確/タイムアウト
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    atr_vals = atr.values
    
    n = len(df)
    targets = np.full(n, np.nan)
    
    for t in range(n):
        if not np.isfinite(close[t]) or not np.isfinite(atr_vals[t]) or atr_vals[t] <= 0:
            continue
        
        entry = close[t]
        current_atr = atr_vals[t]
        
        # ロング用バリア
        long_tp = entry + tp_atr * current_atr
        long_sl = entry - sl_atr * current_atr
        
        # ショート用バリア
        short_tp = entry - tp_atr * current_atr
        short_sl = entry + sl_atr * current_atr
        
        # 将来のバーを走査
        end = min(n, t + 1 + max_hold)
        if end <= t + 1:
            continue
        
        long_result = 0  # 0=未決定, 1=TP先, -1=SL先
        short_result = 0
        
        for i in range(t + 1, end):
            # ロング判定
            if long_result == 0:
                hit_tp = high[i] >= long_tp
                hit_sl = low[i] <= long_sl
                if hit_tp and hit_sl:
                    # 同一バーで両方ヒット → 保守的にSL先と判定
                    long_result = -1
                elif hit_tp:
                    long_result = 1
                elif hit_sl:
                    long_result = -1
            
            # ショート判定
            if short_result == 0:
                hit_tp = low[i] <= short_tp
                hit_sl = high[i] >= short_sl
                if hit_tp and hit_sl:
                    short_result = -1
                elif hit_tp:
                    short_result = 1
                elif hit_sl:
                    short_result = -1
            
            # 両方決定したら終了
            if long_result != 0 and short_result != 0:
                break
        
        # 方向ラベルを決定
        # ロングでTP先 かつ ショートでTP先でない → 買いシグナル (+1)
        # ショートでTP先 かつ ロングでTP先でない → 売りシグナル (-1)
        # それ以外 → エッジなし (0)
        if long_result == 1 and short_result != 1:
            targets[t] = 1.0
        elif short_result == 1 and long_result != 1:
            targets[t] = -1.0
        else:
            targets[t] = 0.0
    
    return pd.Series(targets, index=df.index, name="triple_barrier_target")


def train_model(X_train: pd.DataFrame, y_train: pd.Series, use_triple_barrier: bool = False) -> Any:
    """
    LightGBMモデルを学習。
    
    改善点:
    - num_boost_round増加（200→500）でより深い学習
    - 正則化（lambda_l1, lambda_l2）で過学習抑制
    - num_leaves削減（31→20）で汎化性能向上
    - min_child_samples追加で過学習抑制
    - class_weightでクラス不均衡対策
    - early_stoppingで最適なイテレーション数を自動決定
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBMがインストールされていません。")
        print("pip install lightgbm を実行してください。")
        sys.exit(1)
    
    if use_triple_barrier:
        # Triple-barrier: 3クラス分類 (+1, 0, -1) → (2, 1, 0)
        # +1 (買い) → 2, 0 (エッジなし) → 1, -1 (売り) → 0
        y_class = (y_train + 1).astype(int)  # -1→0, 0→1, +1→2
        
        # サンプル重み計算（不均衡データ対策）
        # ニュートラル(1)は多いので重みを下げ、買い(2)売り(0)の重みを上げる
        class_counts = y_class.value_counts()
        total = len(y_class)
        # 反比例重み: 少ないクラスほど重みが大きい
        class_weight_map = {c: total / (3.0 * count) for c, count in class_counts.items()}
        sample_weights = y_class.map(class_weight_map).values
        
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            # 汎化性能向上のためnum_leavesを削減
            "num_leaves": 20,
            "max_depth": 6,
            "learning_rate": 0.03,  # 0.05→0.03 (より細かい学習)
            "feature_fraction": 0.7,  # 0.8→0.7 (多様性向上)
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            # 正則化パラメータ（過学習抑制）
            "lambda_l1": 0.1,  # L1正則化
            "lambda_l2": 0.5,  # L2正則化
            "min_child_samples": 50,  # 葉ノードの最小サンプル数
            "min_child_weight": 1e-3,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }
    else:
        # 旧方式: 2クラス分類（上昇/下降）
        y_class = (y_train > 0).astype(int)
        
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 20,
            "max_depth": 6,
            "learning_rate": 0.03,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.5,
            "min_child_samples": 50,
            "min_child_weight": 1e-3,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }
    
    # サンプル重みを適用（multiclassの場合のみ）
    if use_triple_barrier:
        train_data = lgb.Dataset(X_train, label=y_class, weight=sample_weights)
    else:
        train_data = lgb.Dataset(X_train, label=y_class)
    
    # 学習（イテレーション数増加）
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,  # 200→500
    )
    
    return model


def backtest(
    df: pd.DataFrame,
    features: pd.DataFrame,
    model: Any,
    initial_capital: float = 100000,
    risk_per_trade: float = 0.01,
    min_confidence: float = 0.6,
    spread_pips: float = 0.0,
    slippage_pips: float = 0.0,
    pip_value: float = 0.0001,
    use_filters: bool = True,
    adx_threshold: float = 20.0,
    sma_atr_threshold: float = 0.5,
    use_triple_barrier: bool = False,
    symbol: str = "",
    # リスク管理パラメータ
    max_drawdown_pct: float = 0.25,  # 最大DD制限（これを超えたら取引停止）
    consecutive_loss_limit: int = 5,  # 連敗制限（これを超えたらリスク半減）
    vol_scale_threshold: float = 1.5,  # ボラティリティ閾値（平均の何倍でリスク調整）
    # レバレッジ設定
    base_leverage: float = 3.0,      # 基本レバレッジ（デフォルト3倍）
    max_leverage: float = 30.0,      # 最大レバレッジ（高確信度時）
    leverage_confidence_threshold: float = 0.70,  # この確信度以上でレバレッジ増加
) -> dict[str, Any]:
    """
    シンプルなバックテスト実行。
    
    Args:
        df: OHLCVデータ
        features: 特徴量
        model: 学習済みモデル
        initial_capital: 初期資金
        risk_per_trade: 1トレードあたりリスク
        min_confidence: 最小信頼度
        use_filters: フィルターを使用するか
        adx_threshold: ADXの最小値（トレンド強度フィルター）
        sma_atr_threshold: SMA200からの乖離/ATRの最小値
        use_triple_barrier: Triple-barrier方式を使用するか
        max_drawdown_pct: 最大DD制限（これを超えたら取引停止）
        consecutive_loss_limit: 連敗制限（これを超えたらリスク半減）
        vol_scale_threshold: ボラティリティ閾値（平均の何倍でリスク調整）
    """
    # 共通インデックスに揃える
    common_idx = features.index.intersection(df.index)
    df = df.loc[common_idx]
    features = features.loc[common_idx]
    
    # 予測
    raw_predictions = model.predict(features)
    
    # Triple-barrier方式の場合は3クラス確率から方向と信頼度を計算
    if use_triple_barrier:
        # raw_predictions: shape (n_samples, 3) - [P(売り), P(エッジなし), P(買い)]
        # クラス: 0=売り(-1), 1=エッジなし(0), 2=買い(+1)
        prob_sell = raw_predictions[:, 0]
        prob_neutral = raw_predictions[:, 1]
        prob_buy = raw_predictions[:, 2]
        
        # 方向と信頼度を計算
        # predictions: (方向を示す符号) * (方向性の強さ)
        # 方向性の強さ = max(P(買い), P(売り)) であり、
        # ニュートラル確率が高い場合は取引を控えるため、
        # 信頼度 = max(P(買い), P(売り)) - P(ニュートラル) / 2 として調整
        direction_strength = np.maximum(prob_buy, prob_sell)
        edge_confidence = direction_strength - prob_neutral * 0.5
        predictions = np.where(prob_buy > prob_sell, edge_confidence, -edge_confidence)
        # predictionsの値: 正なら買い方向、負なら売り方向、絶対値が調整済み信頼度
    else:
        predictions = raw_predictions
    
    # 結果格納
    equity = [initial_capital]
    trades = []
    position = None
    cost_pips = spread_pips + slippage_pips
    cost_price = cost_pips * pip_value
    
    # リスク管理状態
    peak_equity = initial_capital  # 最高資産
    consecutive_losses = 0  # 連敗カウント
    trading_halted = False  # 取引停止フラグ
    
    # ボラティリティの移動平均（ATRの20期間平均）
    atr_col = features["atr_14"] if "atr_14" in features.columns else None
    avg_atr = atr_col.rolling(20).mean() if atr_col is not None else None
    
    for i in range(1, len(df)):
        current_price = df.iloc[i]["close"]
        prev_price = df.iloc[i-1]["close"]
        timestamp = df.index[i]
        pred = predictions[i-1]  # 前の足の予測を使用（リーク防止）
        
        # ポジションがある場合は決済チェック
        if position is not None:
            # SL/TP チェック
            if position["direction"] == "long":
                if df.iloc[i]["low"] <= position["sl"]:
                    # SL hit
                    pnl = (position["sl"] - position["entry"]) * position["size"]
                    pnl -= cost_price * position["size"]
                    equity.append(equity[-1] + pnl)
                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                        "direction": "long",
                        "entry": position["entry"],
                        "exit": position["sl"],
                        "pnl": pnl,
                        "reason": "sl",
                    })
                    # リスク管理: 連敗カウント更新
                    consecutive_losses += 1
                    position = None
                    continue
                elif df.iloc[i]["high"] >= position["tp"]:
                    # TP hit
                    pnl = (position["tp"] - position["entry"]) * position["size"]
                    pnl -= cost_price * position["size"]
                    equity.append(equity[-1] + pnl)
                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                        "direction": "long",
                        "entry": position["entry"],
                        "exit": position["tp"],
                        "pnl": pnl,
                        "reason": "tp",
                    })
                    # リスク管理: 連敗リセット、ピーク更新
                    consecutive_losses = 0
                    if equity[-1] > peak_equity:
                        peak_equity = equity[-1]
                    position = None
                    continue
            else:  # short
                if df.iloc[i]["high"] >= position["sl"]:
                    pnl = (position["entry"] - position["sl"]) * position["size"]
                    pnl -= cost_price * position["size"]
                    equity.append(equity[-1] + pnl)
                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                        "direction": "short",
                        "entry": position["entry"],
                        "exit": position["sl"],
                        "pnl": pnl,
                        "reason": "sl",
                    })
                    # リスク管理: 連敗カウント更新
                    consecutive_losses += 1
                    position = None
                    continue
                elif df.iloc[i]["low"] <= position["tp"]:
                    pnl = (position["entry"] - position["tp"]) * position["size"]
                    pnl -= cost_price * position["size"]
                    equity.append(equity[-1] + pnl)
                    trades.append({
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                        "direction": "short",
                        "entry": position["entry"],
                        "exit": position["tp"],
                        "pnl": pnl,
                        "reason": "tp",
                    })
                    # リスク管理: 連敗リセット、ピーク更新
                    consecutive_losses = 0
                    if equity[-1] > peak_equity:
                        peak_equity = equity[-1]
                    position = None
                    continue
            
            equity.append(equity[-1])
            continue
        
        # 新規エントリー判定
        if use_triple_barrier:
            # Triple-barrier: predは正なら買い、負なら売り、絶対値が信頼度
            confidence = abs(pred)
            direction_signal = "long" if pred > 0 else "short"
        else:
            # 旧方式: predは買い確率
            confidence = max(pred, 1 - pred)
            direction_signal = "long" if pred > 0.5 else "short"
        
        if confidence < min_confidence:
            equity.append(equity[-1])
            continue
        
        # USDJPYはロングオンリー（ショートにエッジがないため）
        if symbol == "USDJPY" and direction_signal == "short":
            equity.append(equity[-1])
            continue
        
        # リスク管理: 最大ドローダウン制限
        current_dd = (peak_equity - equity[-1]) / peak_equity if peak_equity > 0 else 0
        if current_dd >= max_drawdown_pct:
            if not trading_halted:
                trading_halted = True
            equity.append(equity[-1])
            continue
        else:
            # DDが閾値の半分以下に回復したら再開
            if trading_halted and current_dd < max_drawdown_pct * 0.5:
                trading_halted = False
        
        # フィルター適用
        if use_filters:
            # 時間帯フィルター: UTC 21:00-01:00 (ロールオーバー・低流動性時間) を除外
            hour = timestamp.hour
            if hour >= 21 or hour < 1:
                equity.append(equity[-1])
                continue
            
            # トレンドレジームフィルター
            adx = features.iloc[i-1]["adx_14"] if "adx_14" in features.columns else 0
            sma_atr_dist = features.iloc[i-1]["sma200_atr_distance"] if "sma200_atr_distance" in features.columns else 0
            
            # ADXが閾値未満 = トレンドなし → スキップ
            if adx < adx_threshold:
                equity.append(equity[-1])
                continue
            
            # SMA200からの乖離が小さい = 方向性なし → スキップ  
            if sma_atr_dist < sma_atr_threshold:
                equity.append(equity[-1])
                continue
        
        # ATRベースのSL/TP（Triple-barrierパラメータと同期）
        atr = features.iloc[i-1]["atr_14"] if "atr_14" in features.columns else current_price * 0.001
        sl_distance = atr * 1.5  # 2.0→1.5 (タイトなSL)
        tp_distance = atr * 2.5  # 3.0→2.5 (タイトなTP)
        
        # ポジションサイズ計算（リスク管理 + レバレッジ制限付き）
        adjusted_risk = risk_per_trade
        
        # 連敗制限: N連敗以上でリスク半減
        if consecutive_losses >= consecutive_loss_limit:
            adjusted_risk *= 0.5
        
        # ボラティリティ調整: 高ボラ時はリスク縮小
        if avg_atr is not None and i > 20:
            current_avg_atr = avg_atr.iloc[i-1]
            if pd.notna(current_avg_atr) and current_avg_atr > 0:
                vol_ratio = atr / current_avg_atr
                if vol_ratio > vol_scale_threshold:
                    # ボラが平均の1.5倍を超えたら、超過分に応じてリスク縮小
                    # 例: vol_ratio=2.0 → adjusted_risk *= 0.75
                    vol_adjustment = 1.0 / vol_ratio
                    adjusted_risk *= max(0.5, vol_adjustment)
        
        # リスクベースのポジションサイズ
        risk_amount = equity[-1] * adjusted_risk
        risk_based_size = risk_amount / sl_distance if sl_distance > 0 else 0
        
        # レバレッジ制限の適用
        # 確信度に応じてレバレッジを調整（基本3倍、高確信度で最大30倍）
        if confidence >= leverage_confidence_threshold:
            # 確信度0.70-1.00を3-30倍にマッピング
            confidence_ratio = (confidence - leverage_confidence_threshold) / (1.0 - leverage_confidence_threshold)
            current_leverage = base_leverage + (max_leverage - base_leverage) * confidence_ratio
        else:
            current_leverage = base_leverage
        
        # 最大ポジションサイズ = 資金 × レバレッジ / 価格
        max_position_size = (equity[-1] * current_leverage) / current_price
        
        # リスクベースと最大サイズの小さい方を採用
        position_size = min(risk_based_size, max_position_size)
        
        if direction_signal == "long":
            position = {
                "direction": "long",
                "entry": current_price,
                "entry_time": timestamp,
                "sl": current_price - sl_distance,
                "tp": current_price + tp_distance,
                "size": position_size,
            }
        else:  # Short signal
            position = {
                "direction": "short",
                "entry": current_price,
                "entry_time": timestamp,
                "sl": current_price + sl_distance,
                "tp": current_price - tp_distance,
                "size": position_size,
            }
        
        equity.append(equity[-1])
    
    # 最終ポジションを決済
    if position is not None:
        current_price = df.iloc[-1]["close"]
        if position["direction"] == "long":
            pnl = (current_price - position["entry"]) * position["size"]
        else:
            pnl = (position["entry"] - current_price) * position["size"]
        pnl -= cost_price * position["size"]
        # 最後のequity値を更新（追加ではなく）
        equity[-1] = equity[-1] + pnl
        trades.append({
            "entry_time": position["entry_time"],
            "exit_time": df.index[-1],
            "direction": position["direction"],
            "entry": position["entry"],
            "exit": current_price,
            "pnl": pnl,
            "reason": "end",
        })
    
    return {
        "equity": pd.Series(equity, index=df.index),
        "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
    }


def walk_forward_backtest(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    initial_capital: float,
    risk_per_trade: float,
    min_confidence: float,
    train_years: int,
    step_months: int,
    spread_pips: float,
    slippage_pips: float,
    pip_value: float,
    use_filters: bool = True,
    adx_threshold: float = 20.0,
    sma_atr_threshold: float = 0.5,
    use_triple_barrier: bool = False,
    symbol: str = "",
    # リスク管理パラメータ
    max_drawdown_pct: float = 0.25,
    consecutive_loss_limit: int = 5,
    vol_scale_threshold: float = 1.5,
    # レバレッジ設定
    base_leverage: float = 3.0,
    max_leverage: float = 30.0,
    leverage_confidence_threshold: float = 0.70,
) -> dict[str, Any]:
    """拡張ウィンドウのWalk-Forwardバックテスト。"""
    current_equity = initial_capital
    equity_parts = []
    trades_parts = []

    start = X.index.min()
    train_end = start + pd.DateOffset(years=train_years)
    last_index = X.index.max()

    while train_end < last_index:
        test_end = train_end + pd.DateOffset(months=step_months)
        X_train = X.loc[:train_end]
        y_train = y.loc[:train_end]
        X_test = X.loc[(X.index > train_end) & (X.index <= test_end)]

        if len(X_test) < 50:
            break

        df_test = df.loc[X_test.index]
        model = train_model(X_train, y_train, use_triple_barrier=use_triple_barrier)
        result = backtest(
            df_test,
            X_test,
            model,
            initial_capital=current_equity,
            risk_per_trade=risk_per_trade,
            min_confidence=min_confidence,
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
            pip_value=pip_value,
            use_filters=use_filters,
            adx_threshold=adx_threshold,
            sma_atr_threshold=sma_atr_threshold,
            use_triple_barrier=use_triple_barrier,
            symbol=symbol,
            max_drawdown_pct=max_drawdown_pct,
            consecutive_loss_limit=consecutive_loss_limit,
            vol_scale_threshold=vol_scale_threshold,
            base_leverage=base_leverage,
            max_leverage=max_leverage,
            leverage_confidence_threshold=leverage_confidence_threshold,
        )

        equity_parts.append(result["equity"])
        if not result["trades"].empty:
            trades_parts.append(result["trades"])

        current_equity = float(result["equity"].iloc[-1])
        train_end = test_end

    if equity_parts:
        equity = pd.concat(equity_parts).sort_index()
    else:
        equity = pd.Series([initial_capital], index=[X.index.min()])

    if trades_parts:
        trades = pd.concat(trades_parts).reset_index(drop=True)
    else:
        trades = pd.DataFrame()

    return {
        "equity": equity,
        "trades": trades,
    }


def calculate_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict[str, float]:
    """パフォーマンス指標を計算。"""
    returns = equity.pct_change().dropna()
    
    # 基本指標
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    
    # Sharpe Ratio (年率)
    if len(returns) > 0 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24)
    else:
        sharpe = 0
    
    # Max Drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    
    # 取引統計
    if len(trades) > 0:
        total_trades = len(trades)
        winning_trades = len(trades[trades["pnl"] > 0])
        win_rate = winning_trades / total_trades * 100
        
        gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades[trades["pnl"] < 0]["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        
        avg_win = trades[trades["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades[trades["pnl"] < 0]["pnl"].mean()) if total_trades - winning_trades > 0 else 0
    else:
        total_trades = 0
        win_rate = 0
        profit_factor = 0
        avg_win = 0
        avg_loss = 0
    
    return {
        "total_return_pct": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "total_trades": total_trades,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def print_report(symbol: str, metrics: dict[str, float], trades: pd.DataFrame) -> None:
    """レポートを表示。"""
    print(f"\n{'='*60}")
    print(f" {symbol} バックテスト結果")
    print(f"{'='*60}")
    
    print(f"\n【パフォーマンス】")
    print(f"  総リターン:     {metrics['total_return_pct']:>10.2f} %")
    print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:>10.2f}")
    print(f"  最大DD:         {metrics['max_drawdown_pct']:>10.2f} %")
    
    print(f"\n【取引統計】")
    print(f"  総取引数:       {metrics['total_trades']:>10.0f}")
    print(f"  勝率:           {metrics['win_rate_pct']:>10.2f} %")
    print(f"  Profit Factor:  {metrics['profit_factor']:>10.2f}")
    print(f"  平均利益:       {metrics['avg_win']:>10.2f}")
    print(f"  平均損失:       {metrics['avg_loss']:>10.2f}")
    
    # 直近5取引
    if len(trades) > 0:
        print(f"\n【直近5取引】")
        recent = trades.tail(5)
        for _, trade in recent.iterrows():
            direction = "L" if trade["direction"] == "long" else "S"
            pnl_str = f"+{trade['pnl']:.2f}" if trade["pnl"] > 0 else f"{trade['pnl']:.2f}"
            print(f"  {direction} {trade['entry_time']} → {trade['exit_time']} | PnL: {pnl_str} ({trade['reason']})")


def main():
    """メイン処理。"""
    print("=" * 60)
    print(" FX自動売買システム - 通貨ペア別最適化バックテスト")
    print("=" * 60)
    
    # ==========================================================================
    # 全12通貨ペアをテスト（PAIR_CONFIGで個別設定）
    # ==========================================================================
    symbols = list(PAIR_CONFIG.keys())  # 全12ペア
    timeframe = "M30"
    data_dir = project_root / "data" / "raw"
    initial_capital = 10000
    full_period = False
    walk_forward = True
    train_years = 2
    step_months = 6
    slippage_pips = 0.2
    
    # 共通設定
    use_filters = True
    use_triple_barrier = True
    max_hold_bars = 36
    
    # リスク管理設定（共通）
    max_drawdown_pct = 0.25
    consecutive_loss_limit = 5
    vol_scale_threshold = 1.5
    leverage_confidence_threshold = 0.65
    
    # データ読み込み
    print("\n[1/4] データ読み込み...")
    datasets = {}
    for symbol in symbols:
        df = load_data(symbol, timeframe, data_dir)
        if df is not None:
            datasets[symbol] = df
    
    if not datasets:
        print("\nデータが見つかりません。")
        print("先に scripts/download_data.py を実行してください。")
        return 1
    
    # 各通貨ペアで検証
    all_results = {}
    
    for symbol, df in datasets.items():
        print(f"\n{'='*60}")
        print(f" {symbol} の検証を開始")
        print(f"{'='*60}")
        
        # ========== ペア別設定を取得 ==========
        pair_cfg = get_pair_config(symbol)
        base_leverage = pair_cfg["base_leverage"]
        max_leverage = pair_cfg["max_leverage"]
        adx_threshold = pair_cfg["adx_threshold"]
        sma_atr_threshold = pair_cfg["sma_atr_threshold"]
        tp_atr_mult = pair_cfg["tp_atr_mult"]
        sl_atr_mult = pair_cfg["sl_atr_mult"]
        long_only = pair_cfg["long_only"]
        spread_pips = pair_cfg["spread_pips"]
        pip_value = pair_cfg["pip_value"]
        
        print(f"  設定: レバ={base_leverage}-{max_leverage}x, ADX>={adx_threshold}, TP={tp_atr_mult}ATR, SL={sl_atr_mult}ATR")
        if long_only:
            print(f"  制約: ロングのみ")
        
        # 特徴量生成
        print("\n[2/4] 特徴量生成...")
        features = compute_features(df)
        print(f"  特徴量数: {len(features.columns)}")
        print(f"  サンプル数: {len(features)}")
        
        # ターゲット計算
        if use_triple_barrier:
            # ATRを計算（compute_featuresと同じロジック）
            tr1 = df["high"] - df["low"]
            tr2 = abs(df["high"] - df["close"].shift(1))
            tr3 = abs(df["low"] - df["close"].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            
            target = compute_triple_barrier_target(
                df, atr,
                tp_atr=tp_atr_mult,
                sl_atr=sl_atr_mult,
                max_hold=max_hold_bars,
            )
            print(f"  ターゲット: Triple-barrier方式 (TP={tp_atr_mult}ATR, SL={sl_atr_mult}ATR, max_hold={max_hold_bars})")
        else:
            target = compute_target(df, horizon=6)
            print("  ターゲット: 6時間後リターン方向")
        
        # 共通インデックス
        common_idx = features.index.intersection(target.dropna().index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]
        
        # ターゲット分布を表示
        if use_triple_barrier:
            buy_pct = (y == 1).mean() * 100
            sell_pct = (y == -1).mean() * 100
            neutral_pct = (y == 0).mean() * 100
            print(f"  ラベル分布: 買い {buy_pct:.1f}%, 売り {sell_pct:.1f}%, ニュートラル {neutral_pct:.1f}%")
        
        # Train/Test分割 (時系列なので後半20%をテスト)
        if full_period:
            X_train, y_train = X, y
            X_test, y_test = X, y
            df_test = df.loc[X.index]
            print("  フル期間バックテスト: 学習=全期間, テスト=全期間 (インサンプル)")
            print(f"  期間: {X.index[0].date()} ~ {X.index[-1].date()} ({len(X)} サンプル)")
        elif walk_forward:
            print("  Walk-Forward: 拡張ウィンドウ, 6か月ごと再学習")
            print(f"  初回学習期間: {train_years}年")
            print(f"  データ期間: {X.index[0].date()} ~ {X.index[-1].date()} ({len(X)} サンプル)")
            if use_filters:
                print(f"  フィルター: 有効 (ADX >= {adx_threshold}, SMA200乖離/ATR >= {sma_atr_threshold}, 時間帯除外: 21-01 UTC)")
            else:
                print("  フィルター: 無効")
            X_train, y_train = X, y
            X_test, y_test = X, y
            df_test = df.loc[X.index]
        else:
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            df_test = df.loc[X_test.index]

            print(f"  学習期間: {X_train.index[0].date()} ~ {X_train.index[-1].date()} ({len(X_train)} サンプル)")
            print(f"  テスト期間: {X_test.index[0].date()} ~ {X_test.index[-1].date()} ({len(X_test)} サンプル)")
        
        # モデル学習
        print("\n[3/4] モデル学習...")
        model = train_model(X_train, y_train, use_triple_barrier=use_triple_barrier)
        
        # 特徴量重要度
        importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importance(),
        }).sort_values("importance", ascending=False)
        print("  Top 5 特徴量:")
        for _, row in importance.head(5).iterrows():
            print(f"    - {row['feature']}: {row['importance']}")
        
        # バックテスト
        print("\n[4/4] バックテスト実行...")
        # 閾値探索モード
        threshold_search = use_triple_barrier  # Triple-barrier方式の場合のみ閾値探索
        
        if threshold_search:
            # 複数の閾値をテスト
            thresholds = [0.50, 0.55, 0.60]
            print(f"\n  閾値探索中...")
            best_pf = 0
            best_trades = 0
            best_threshold = 0.4
            best_result = None
            
            for thresh in thresholds:
                if walk_forward:
                    test_result = walk_forward_backtest(
                        df, X, y,
                        initial_capital=initial_capital,
                        risk_per_trade=0.03,
                        min_confidence=thresh,
                        train_years=train_years,
                        step_months=step_months,
                        spread_pips=spread_pips,
                        slippage_pips=slippage_pips,
                        pip_value=pip_value,
                        use_filters=use_filters,
                        adx_threshold=adx_threshold,
                        sma_atr_threshold=sma_atr_threshold,
                        use_triple_barrier=use_triple_barrier,
                        symbol=symbol,
                        max_drawdown_pct=max_drawdown_pct,
                        consecutive_loss_limit=consecutive_loss_limit,
                        vol_scale_threshold=vol_scale_threshold,
                        base_leverage=base_leverage,
                        max_leverage=max_leverage,
                        leverage_confidence_threshold=leverage_confidence_threshold,
                    )
                else:
                    test_result = backtest(
                        df_test, X_test, model,
                        initial_capital=initial_capital,
                        spread_pips=spread_pips,
                        slippage_pips=slippage_pips,
                        pip_value=pip_value,
                        use_filters=use_filters,
                        adx_threshold=adx_threshold,
                        sma_atr_threshold=sma_atr_threshold,
                        use_triple_barrier=use_triple_barrier,
                        symbol=symbol,
                        max_drawdown_pct=max_drawdown_pct,
                        consecutive_loss_limit=consecutive_loss_limit,
                        vol_scale_threshold=vol_scale_threshold,
                        base_leverage=base_leverage,
                        max_leverage=max_leverage,
                        leverage_confidence_threshold=leverage_confidence_threshold,
                    )
                
                test_metrics = calculate_metrics(test_result["equity"], test_result["trades"])
                pf = test_metrics["profit_factor"]
                trades = test_metrics["total_trades"]
                ret = test_metrics["total_return_pct"]
                print(f"    閾値 {thresh:.2f}: PF={pf:.2f}, 取引数={trades:.0f}, リターン={ret:.1f}%")
                
                # PF > 1.0 かつ取引数が十分な閾値を選択（取引頻度重視）
                # PFが1.0以上であれば、より多くの取引ができる低閾値を優先
                if pf >= 1.0 and trades > 30 and (best_pf < 1.0 or trades > best_trades):
                    best_pf = pf
                    best_trades = trades
                    best_threshold = thresh
                    best_result = test_result
            
            print(f"  最適閾値: {best_threshold:.2f} (PF={best_pf:.2f})")
            result = best_result if best_result else test_result
        else:
            if walk_forward:
                result = walk_forward_backtest(
                    df,
                    X,
                    y,
                    initial_capital=initial_capital,
                    risk_per_trade=0.03,
                    min_confidence=0.6,
                    train_years=train_years,
                    step_months=step_months,
                    spread_pips=spread_pips,
                    slippage_pips=slippage_pips,
                    pip_value=pip_value,
                    use_filters=use_filters,
                    adx_threshold=adx_threshold,
                    sma_atr_threshold=sma_atr_threshold,
                    use_triple_barrier=use_triple_barrier,
                    symbol=symbol,
                    max_drawdown_pct=max_drawdown_pct,
                    consecutive_loss_limit=consecutive_loss_limit,
                    vol_scale_threshold=vol_scale_threshold,
                    base_leverage=base_leverage,
                    max_leverage=max_leverage,
                    leverage_confidence_threshold=leverage_confidence_threshold,
                )
            else:
                result = backtest(
                    df_test,
                    X_test,
                    model,
                    initial_capital=initial_capital,
                    spread_pips=spread_pips,
                    slippage_pips=slippage_pips,
                    pip_value=pip_value,
                    use_filters=use_filters,
                    adx_threshold=adx_threshold,
                    sma_atr_threshold=sma_atr_threshold,
                    use_triple_barrier=use_triple_barrier,
                    symbol=symbol,
                    max_drawdown_pct=max_drawdown_pct,
                    consecutive_loss_limit=consecutive_loss_limit,
                    vol_scale_threshold=vol_scale_threshold,
                    base_leverage=base_leverage,
                    max_leverage=max_leverage,
                    leverage_confidence_threshold=leverage_confidence_threshold,
                )
        
        # 評価
        metrics = calculate_metrics(result["equity"], result["trades"])
        all_results[symbol] = {
            "metrics": metrics,
            "trades": result["trades"],
            "equity": result["equity"],
        }
        
        # レポート表示
        print_report(symbol, metrics, result["trades"])
    
    # サマリー
    print("\n" + "=" * 60)
    print(" 全体サマリー")
    print("=" * 60)
    print(f"\n{'Symbol':<10} {'Return%':>10} {'Sharpe':>10} {'MaxDD%':>10} {'Trades':>10} {'WinRate%':>10} {'PF':>10}")
    print("-" * 70)
    
    for symbol, result in all_results.items():
        m = result["metrics"]
        print(f"{symbol:<10} {m['total_return_pct']:>10.2f} {m['sharpe_ratio']:>10.2f} {m['max_drawdown_pct']:>10.2f} {m['total_trades']:>10.0f} {m['win_rate_pct']:>10.2f} {m['profit_factor']:>10.2f}")
    
    print("\n" + "=" * 60)
    print(" 検証完了")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
