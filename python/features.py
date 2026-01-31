"""
共通特徴量計算モジュール

バックテストと本番取引で同一の特徴量を使用するための共通定義。
このファイルを編集すると、両方に反映される。

更新日: 2026-01-31
"""

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    テクニカル指標ベースの特徴量を計算。
    
    リーク防止: すべて過去データのみ使用（shift使用）
    
    Args:
        df: OHLCVデータ（open, high, low, close, volume）
        
    Returns:
        特徴量DataFrame（67カラム）
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
    # セッション特性（東京/ロンドン/NY）
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
    
    # セッション別ボラティリティ
    overall_vol = ret1.rolling(24, min_periods=12).std()
    features["tokyo_vol_ratio"] = features["is_tokyo"] * overall_vol
    features["london_vol_ratio"] = features["is_london"] * overall_vol
    features["ny_vol_ratio"] = features["is_ny"] * overall_vol
    
    # ========================================
    # トレンド品質（Efficiency Ratio）
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
    # ジャンプ/テールリスク
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
    # ミーンリバージョン（レンジ相場用）
    # ========================================
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    
    # Zスコア
    features["zscore"] = (close - ma) / sd
    
    # バンド幅のパーセンタイル（圧縮度）
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
    tp_atr: float = 2.5,
    sl_atr: float = 1.5,
    max_hold: int = 36,
) -> pd.Series:
    """
    Triple-barrier方式のターゲット: SL/TPどちらが先にヒットするかを予測。
    
    バックテストの実際の決済ロジックと完全に一致するターゲット。
    
    Args:
        df: OHLCVデータ（high, low, close必須）
        atr: ATR値のSeries
        tp_atr: TP距離のATR倍率（デフォルト2.5）
        sl_atr: SL距離のATR倍率（デフォルト1.5）
        max_hold: 最大保有期間（バー数、デフォルト36）
    
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
        if long_result == 1 and short_result != 1:
            targets[t] = 1.0
        elif short_result == 1 and long_result != 1:
            targets[t] = -1.0
        else:
            targets[t] = 0.0
    
    return pd.Series(targets, index=df.index, name="triple_barrier_target")


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR (Average True Range) を計算。
    
    Args:
        df: OHLCVデータ
        period: 期間（デフォルト14）
    
    Returns:
        ATRのSeries
    """
    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift(1))
    tr3 = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()
