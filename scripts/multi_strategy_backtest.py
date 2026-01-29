"""
複数戦略 × 複数通貨ペアの包括的バックテスト。

各戦略を全通貨ペアでテストし、最も勝率・PFが高い組み合わせを特定する。

使用方法:
    python scripts/multi_strategy_backtest.py

前提条件:
    - scripts/download_data.py でデータを取得済み
"""

import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Callable
import warnings
import json

warnings.filterwarnings("ignore")

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

import numpy as np
import pandas as pd


# =============================================================================
# 戦略定義
# =============================================================================

@dataclass
class StrategyConfig:
    """戦略設定"""
    name: str
    description: str
    signal_func: Callable
    min_confidence: float = 0.5
    tp_atr_mult: float = 3.0
    sl_atr_mult: float = 2.0


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """共通特徴量を計算"""
    features = pd.DataFrame(index=df.index)
    
    # 価格系
    features["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
    features["log_return_5"] = np.log(df["close"] / df["close"].shift(5))
    features["log_return_20"] = np.log(df["close"] / df["close"].shift(20))
    
    # ボラティリティ
    features["realized_vol_20"] = features["log_return_1"].rolling(20).std() * np.sqrt(252 * 24)
    
    # 移動平均
    features["sma_10"] = df["close"].rolling(10).mean()
    features["sma_20"] = df["close"].rolling(20).mean()
    features["sma_50"] = df["close"].rolling(50).mean()
    features["sma_200"] = df["close"].rolling(200).mean()
    features["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    features["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features["rsi_14"] = 100 - (100 / (1 + rs))
    
    # MACD
    features["macd"] = features["ema_12"] - features["ema_26"]
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]
    
    # ATR
    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift(1))
    tr3 = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    features["atr_14"] = tr.rolling(14).mean()
    
    # ADX
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr_smooth = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / (atr_smooth + 1e-10))
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / (atr_smooth + 1e-10))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    features["adx_14"] = dx.ewm(span=14, adjust=False).mean()
    features["plus_di"] = plus_di
    features["minus_di"] = minus_di
    
    # ボリンジャーバンド
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    features["bb_upper"] = bb_mid + 2 * bb_std
    features["bb_lower"] = bb_mid - 2 * bb_std
    features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / (bb_mid + 1e-10)
    features["bb_position"] = (df["close"] - features["bb_lower"]) / (features["bb_upper"] - features["bb_lower"] + 1e-10)
    
    # ストキャスティクス
    low_14 = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    features["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 1e-10)
    features["stoch_d"] = features["stoch_k"].rolling(3).mean()
    
    # CCI
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma_tp = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    features["cci_20"] = (tp - ma_tp) / (0.015 * md + 1e-10)
    
    # モメンタム
    features["momentum_10"] = df["close"] / df["close"].shift(10) - 1
    features["momentum_20"] = df["close"] / df["close"].shift(20) - 1
    
    # ROC
    features["roc_10"] = (df["close"] - df["close"].shift(10)) / (df["close"].shift(10) + 1e-10) * 100
    
    # Williams %R
    features["williams_r"] = -100 * (high_14 - df["close"]) / (high_14 - low_14 + 1e-10)
    
    # 時間特徴量
    features["hour"] = df.index.hour
    features["day_of_week"] = df.index.dayofweek
    
    # 価格位置
    features["price_to_sma20"] = df["close"] / features["sma_20"] - 1
    features["price_to_sma50"] = df["close"] / features["sma_50"] - 1
    features["price_to_sma200"] = df["close"] / features["sma_200"] - 1
    
    return features.dropna()


# =============================================================================
# 戦略1: トレンドフォロー (SMA クロス + ADX)
# =============================================================================

def strategy_trend_following(df: pd.DataFrame, features: pd.DataFrame, i: int) -> dict | None:
    """
    トレンドフォロー戦略
    - SMA20 > SMA50 かつ ADX > 25 でロング
    - SMA20 < SMA50 かつ ADX > 25 でショート
    """
    if i < 1:
        return None
    
    f = features.iloc[i-1]
    
    # ADXフィルター
    if f["adx_14"] < 25:
        return None
    
    sma20 = f["sma_20"]
    sma50 = f["sma_50"]
    close = df.iloc[i-1]["close"]
    atr = f["atr_14"]
    
    # トレンド方向
    if sma20 > sma50 and close > sma20:
        direction = "long"
        confidence = min(0.9, 0.5 + f["adx_14"] / 100)
    elif sma20 < sma50 and close < sma20:
        direction = "short"
        confidence = min(0.9, 0.5 + f["adx_14"] / 100)
    else:
        return None
    
    return {
        "direction": direction,
        "confidence": confidence,
        "atr": atr,
    }


# =============================================================================
# 戦略2: ミーンリバージョン (ボリンジャーバンド + RSI)
# =============================================================================

def strategy_mean_reversion(df: pd.DataFrame, features: pd.DataFrame, i: int) -> dict | None:
    """
    ミーンリバージョン戦略
    - BB下限 + RSI < 30 でロング
    - BB上限 + RSI > 70 でショート
    """
    if i < 1:
        return None
    
    f = features.iloc[i-1]
    close = df.iloc[i-1]["close"]
    atr = f["atr_14"]
    
    bb_pos = f["bb_position"]
    rsi = f["rsi_14"]
    adx = f["adx_14"]
    
    # 強いトレンド中はスキップ（レンジ相場のみ）
    if adx > 30:
        return None
    
    if bb_pos < 0.1 and rsi < 30:
        direction = "long"
        confidence = min(0.85, 0.5 + (30 - rsi) / 60 + (0.1 - bb_pos) * 2)
    elif bb_pos > 0.9 and rsi > 70:
        direction = "short"
        confidence = min(0.85, 0.5 + (rsi - 70) / 60 + (bb_pos - 0.9) * 2)
    else:
        return None
    
    return {
        "direction": direction,
        "confidence": confidence,
        "atr": atr,
    }


# =============================================================================
# 戦略3: ブレイクアウト (ボリンジャーバンド幅 + ボリューム)
# =============================================================================

def strategy_breakout(df: pd.DataFrame, features: pd.DataFrame, i: int) -> dict | None:
    """
    ブレイクアウト戦略
    - BB幅が縮小後、価格がBB外に出たらブレイクアウト
    """
    if i < 5:
        return None
    
    f = features.iloc[i-1]
    f_prev = features.iloc[i-5:i-1]
    close = df.iloc[i-1]["close"]
    atr = f["atr_14"]
    
    # BB幅の縮小を確認
    bb_width = f["bb_width"]
    bb_width_avg = f_prev["bb_width"].mean()
    
    # 幅が縮小していた場合のみ
    if bb_width > bb_width_avg * 0.8:
        return None
    
    bb_upper = f["bb_upper"]
    bb_lower = f["bb_lower"]
    
    # ブレイクアウト判定
    if close > bb_upper:
        direction = "long"
        confidence = min(0.8, 0.5 + (close - bb_upper) / atr * 0.1)
    elif close < bb_lower:
        direction = "short"
        confidence = min(0.8, 0.5 + (bb_lower - close) / atr * 0.1)
    else:
        return None
    
    return {
        "direction": direction,
        "confidence": confidence,
        "atr": atr,
    }


# =============================================================================
# 戦略4: MACD ダイバージェンス
# =============================================================================

def strategy_macd(df: pd.DataFrame, features: pd.DataFrame, i: int) -> dict | None:
    """
    MACD戦略
    - MACDがシグナルを上抜け + ヒストグラムが正転 でロング
    - MACDがシグナルを下抜け + ヒストグラムが負転 でショート
    """
    if i < 2:
        return None
    
    f = features.iloc[i-1]
    f_prev = features.iloc[i-2]
    atr = f["atr_14"]
    
    macd_hist = f["macd_hist"]
    macd_hist_prev = f_prev["macd_hist"]
    
    # ゼロラインクロス
    if macd_hist > 0 and macd_hist_prev <= 0:
        direction = "long"
        confidence = min(0.75, 0.5 + abs(macd_hist) / atr * 10)
    elif macd_hist < 0 and macd_hist_prev >= 0:
        direction = "short"
        confidence = min(0.75, 0.5 + abs(macd_hist) / atr * 10)
    else:
        return None
    
    return {
        "direction": direction,
        "confidence": confidence,
        "atr": atr,
    }


# =============================================================================
# 戦略5: RSI + ストキャスティクス
# =============================================================================

def strategy_rsi_stoch(df: pd.DataFrame, features: pd.DataFrame, i: int) -> dict | None:
    """
    RSI + ストキャスティクス戦略
    - RSI < 35 かつ Stoch K < 20 でロング
    - RSI > 65 かつ Stoch K > 80 でショート
    """
    if i < 1:
        return None
    
    f = features.iloc[i-1]
    atr = f["atr_14"]
    
    rsi = f["rsi_14"]
    stoch_k = f["stoch_k"]
    stoch_d = f["stoch_d"]
    adx = f["adx_14"]
    
    # レンジ相場でのみ有効
    if adx > 35:
        return None
    
    if rsi < 35 and stoch_k < 20 and stoch_k > stoch_d:
        direction = "long"
        confidence = min(0.8, 0.5 + (35 - rsi) / 70 + (20 - stoch_k) / 40)
    elif rsi > 65 and stoch_k > 80 and stoch_k < stoch_d:
        direction = "short"
        confidence = min(0.8, 0.5 + (rsi - 65) / 70 + (stoch_k - 80) / 40)
    else:
        return None
    
    return {
        "direction": direction,
        "confidence": confidence,
        "atr": atr,
    }


# =============================================================================
# 戦略6: モメンタム + ADX
# =============================================================================

def strategy_momentum(df: pd.DataFrame, features: pd.DataFrame, i: int) -> dict | None:
    """
    モメンタム戦略
    - 強いトレンド中に順張り
    """
    if i < 1:
        return None
    
    f = features.iloc[i-1]
    atr = f["atr_14"]
    
    momentum = f["momentum_20"]
    adx = f["adx_14"]
    plus_di = f["plus_di"]
    minus_di = f["minus_di"]
    
    # 強いトレンドが必要
    if adx < 30:
        return None
    
    if momentum > 0.02 and plus_di > minus_di:
        direction = "long"
        confidence = min(0.85, 0.5 + momentum * 10 + (adx - 30) / 100)
    elif momentum < -0.02 and minus_di > plus_di:
        direction = "short"
        confidence = min(0.85, 0.5 + abs(momentum) * 10 + (adx - 30) / 100)
    else:
        return None
    
    return {
        "direction": direction,
        "confidence": confidence,
        "atr": atr,
    }


# =============================================================================
# 戦略7: Triple Screen (エルダー)
# =============================================================================

def strategy_triple_screen(df: pd.DataFrame, features: pd.DataFrame, i: int) -> dict | None:
    """
    Triple Screen戦略 (簡易版)
    - 長期: SMA200でトレンド判定
    - 中期: MACD方向確認
    - 短期: RSI/Stochで押し目買い/戻り売り
    """
    if i < 1:
        return None
    
    f = features.iloc[i-1]
    close = df.iloc[i-1]["close"]
    atr = f["atr_14"]
    
    # Screen 1: 長期トレンド
    sma200 = f["sma_200"]
    long_trend = close > sma200
    
    # Screen 2: MACD方向
    macd_hist = f["macd_hist"]
    macd_bullish = macd_hist > 0
    
    # Screen 3: オシレーター
    rsi = f["rsi_14"]
    stoch_k = f["stoch_k"]
    
    if long_trend and macd_bullish and rsi < 50 and stoch_k < 40:
        direction = "long"
        confidence = min(0.8, 0.55 + (50 - rsi) / 100)
    elif not long_trend and not macd_bullish and rsi > 50 and stoch_k > 60:
        direction = "short"
        confidence = min(0.8, 0.55 + (rsi - 50) / 100)
    else:
        return None
    
    return {
        "direction": direction,
        "confidence": confidence,
        "atr": atr,
    }


# =============================================================================
# 戦略8: CCI + EMA
# =============================================================================

def strategy_cci(df: pd.DataFrame, features: pd.DataFrame, i: int) -> dict | None:
    """
    CCI戦略
    - CCIが-100から上抜けでロング
    - CCIが+100から下抜けでショート
    """
    if i < 2:
        return None
    
    f = features.iloc[i-1]
    f_prev = features.iloc[i-2]
    close = df.iloc[i-1]["close"]
    atr = f["atr_14"]
    
    cci = f["cci_20"]
    cci_prev = f_prev["cci_20"]
    ema_12 = f["ema_12"]
    ema_26 = f["ema_26"]
    
    # EMAトレンド確認
    bullish_trend = ema_12 > ema_26
    
    if cci > -100 and cci_prev <= -100 and bullish_trend:
        direction = "long"
        confidence = min(0.75, 0.5 + (cci + 100) / 400)
    elif cci < 100 and cci_prev >= 100 and not bullish_trend:
        direction = "short"
        confidence = min(0.75, 0.5 + (100 - cci) / 400)
    else:
        return None
    
    return {
        "direction": direction,
        "confidence": confidence,
        "atr": atr,
    }


# =============================================================================
# バックテストエンジン
# =============================================================================

def backtest_strategy(
    df: pd.DataFrame,
    features: pd.DataFrame,
    strategy_func: Callable,
    initial_capital: float = 10000,
    risk_per_trade: float = 0.01,
    min_confidence: float = 0.5,
    tp_atr_mult: float = 3.0,
    sl_atr_mult: float = 2.0,
    pip_value: float = 0.0001,
    spread_pips: float = 1.0,
) -> dict[str, Any]:
    """シンプルなバックテスト実行"""
    
    common_idx = features.index.intersection(df.index)
    df = df.loc[common_idx]
    features = features.loc[common_idx]
    
    equity = [initial_capital]
    trades = []
    position = None
    cost_price = spread_pips * pip_value
    
    for i in range(1, len(df)):
        current_price = df.iloc[i]["close"]
        timestamp = df.index[i]
        
        # ポジションがある場合は決済チェック
        if position is not None:
            if position["direction"] == "long":
                if df.iloc[i]["low"] <= position["sl"]:
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
                    position = None
                    continue
                elif df.iloc[i]["high"] >= position["tp"]:
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
                    position = None
                    continue
            else:
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
                    position = None
                    continue
            
            equity.append(equity[-1])
            continue
        
        # シグナル取得
        signal = strategy_func(df, features, i)
        
        if signal is None or signal.get("confidence", 0) < min_confidence:
            equity.append(equity[-1])
            continue
        
        # 時間帯フィルター
        hour = timestamp.hour
        if hour >= 21 or hour < 1:
            equity.append(equity[-1])
            continue
        
        direction = signal["direction"]
        atr = signal.get("atr", current_price * 0.001)
        
        sl_distance = atr * sl_atr_mult
        tp_distance = atr * tp_atr_mult
        
        risk_amount = equity[-1] * risk_per_trade
        position_size = risk_amount / sl_distance if sl_distance > 0 else 0
        
        if direction == "long":
            position = {
                "direction": "long",
                "entry": current_price,
                "entry_time": timestamp,
                "sl": current_price - sl_distance,
                "tp": current_price + tp_distance,
                "size": position_size,
            }
        else:
            position = {
                "direction": "short",
                "entry": current_price,
                "entry_time": timestamp,
                "sl": current_price + sl_distance,
                "tp": current_price - tp_distance,
                "size": position_size,
            }
        
        equity.append(equity[-1])
    
    # 最終ポジション決済
    if position is not None:
        current_price = df.iloc[-1]["close"]
        if position["direction"] == "long":
            pnl = (current_price - position["entry"]) * position["size"]
        else:
            pnl = (position["entry"] - current_price) * position["size"]
        pnl -= cost_price * position["size"]
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


def calculate_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict[str, float]:
    """パフォーマンス指標を計算"""
    returns = equity.pct_change().dropna()
    
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    
    if len(returns) > 0 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24)
    else:
        sharpe = 0
    
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100
    
    if len(trades) > 0:
        total_trades = len(trades)
        winning_trades = len(trades[trades["pnl"] > 0])
        win_rate = winning_trades / total_trades * 100
        
        gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades[trades["pnl"] < 0]["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    else:
        total_trades = 0
        win_rate = 0
        profit_factor = 0
    
    return {
        "total_return_pct": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "total_trades": total_trades,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
    }


# =============================================================================
# メイン処理
# =============================================================================

def main():
    print("=" * 80)
    print(" 複数戦略 × 複数通貨ペア バックテスト")
    print("=" * 80)
    
    # 通貨ペア設定
    all_symbols = [
        "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD",
        "EURJPY", "GBPJPY", "AUDJPY", "CADJPY",
        "EURGBP", "EURAUD",
    ]
    
    pip_values = {
        "EURUSD": 0.0001, "GBPUSD": 0.0001, "AUDUSD": 0.0001,
        "USDCHF": 0.0001, "USDCAD": 0.0001, "EURGBP": 0.0001,
        "EURAUD": 0.0001,
        "USDJPY": 0.01, "EURJPY": 0.01, "GBPJPY": 0.01,
        "AUDJPY": 0.01, "CADJPY": 0.01,
    }
    
    spread_pips = {
        "EURUSD": 1.0, "GBPUSD": 1.5, "AUDUSD": 1.2,
        "USDCHF": 1.5, "USDCAD": 1.5, "EURGBP": 1.5,
        "EURAUD": 2.0,
        "USDJPY": 1.2, "EURJPY": 1.5, "GBPJPY": 2.0,
        "AUDJPY": 1.5, "CADJPY": 1.8,
    }
    
    # 戦略定義
    strategies = [
        StrategyConfig("TrendFollowing", "SMA Cross + ADX", strategy_trend_following, 0.5, 3.0, 2.0),
        StrategyConfig("MeanReversion", "BB + RSI", strategy_mean_reversion, 0.5, 2.0, 1.5),
        StrategyConfig("Breakout", "BB Squeeze", strategy_breakout, 0.5, 4.0, 2.0),
        StrategyConfig("MACD", "MACD Histogram", strategy_macd, 0.5, 3.0, 2.0),
        StrategyConfig("RSI_Stoch", "RSI + Stochastic", strategy_rsi_stoch, 0.5, 2.5, 1.5),
        StrategyConfig("Momentum", "Momentum + ADX", strategy_momentum, 0.5, 3.5, 2.0),
        StrategyConfig("TripleScreen", "Elder Triple Screen", strategy_triple_screen, 0.5, 3.0, 2.0),
        StrategyConfig("CCI", "CCI Reversal", strategy_cci, 0.5, 2.5, 1.5),
    ]
    
    # データディレクトリ
    data_dir = project_root / "data" / "raw"
    timeframe = "H1"
    initial_capital = 10000
    
    # 結果格納
    all_results = []
    
    print(f"\n対象通貨ペア: {len(all_symbols)}")
    print(f"対象戦略: {len(strategies)}")
    print(f"組み合わせ数: {len(all_symbols) * len(strategies)}")
    print()
    
    # データ読み込み
    print("[1/2] データ読み込み...")
    datasets = {}
    for symbol in all_symbols:
        filepath = data_dir / f"{symbol}_{timeframe}.parquet"
        if filepath.exists():
            df = pd.read_parquet(filepath)
            datasets[symbol] = df
            print(f"  {symbol}: {len(df)} bars ({df.index[0].date()} ~ {df.index[-1].date()})")
        else:
            print(f"  {symbol}: データなし")
    
    if not datasets:
        print("\nデータが見つかりません。先に download_data.py を実行してください。")
        return 1
    
    # バックテスト実行
    print(f"\n[2/2] バックテスト実行中...")
    print()
    
    for symbol, df in datasets.items():
        print(f"\n{symbol}:")
        features = compute_base_features(df)
        
        for strategy in strategies:
            try:
                result = backtest_strategy(
                    df=df,
                    features=features,
                    strategy_func=strategy.signal_func,
                    initial_capital=initial_capital,
                    min_confidence=strategy.min_confidence,
                    tp_atr_mult=strategy.tp_atr_mult,
                    sl_atr_mult=strategy.sl_atr_mult,
                    pip_value=pip_values.get(symbol, 0.0001),
                    spread_pips=spread_pips.get(symbol, 1.5),
                )
                
                metrics = calculate_metrics(result["equity"], result["trades"])
                
                all_results.append({
                    "symbol": symbol,
                    "strategy": strategy.name,
                    "total_return_pct": metrics["total_return_pct"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown_pct": metrics["max_drawdown_pct"],
                    "total_trades": metrics["total_trades"],
                    "win_rate_pct": metrics["win_rate_pct"],
                    "profit_factor": metrics["profit_factor"],
                })
                
                print(f"  {strategy.name:<15} | Ret: {metrics['total_return_pct']:>7.1f}% | "
                      f"WR: {metrics['win_rate_pct']:>5.1f}% | PF: {metrics['profit_factor']:>5.2f} | "
                      f"Trades: {metrics['total_trades']:>4.0f} | MDD: {metrics['max_drawdown_pct']:>6.1f}%")
                
            except Exception as e:
                print(f"  {strategy.name:<15} | エラー: {e}")
    
    # 結果をDataFrameに
    results_df = pd.DataFrame(all_results)
    
    # 結果保存
    output_dir = project_root / "data" / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "multi_strategy_results.csv", index=False)
    
    # サマリー表示
    print("\n" + "=" * 80)
    print(" 結果サマリー")
    print("=" * 80)
    
    # 勝率トップ10
    print("\n【勝率 Top 10】")
    print("-" * 70)
    top_win_rate = results_df[results_df["total_trades"] >= 50].nlargest(10, "win_rate_pct")
    for _, row in top_win_rate.iterrows():
        print(f"  {row['symbol']:<8} × {row['strategy']:<15} | WR: {row['win_rate_pct']:>5.1f}% | "
              f"PF: {row['profit_factor']:>5.2f} | Trades: {row['total_trades']:>4.0f}")
    
    # Profit Factor トップ10
    print("\n【Profit Factor Top 10】")
    print("-" * 70)
    top_pf = results_df[(results_df["total_trades"] >= 50) & (results_df["profit_factor"] < 100)].nlargest(10, "profit_factor")
    for _, row in top_pf.iterrows():
        print(f"  {row['symbol']:<8} × {row['strategy']:<15} | PF: {row['profit_factor']:>5.2f} | "
              f"WR: {row['win_rate_pct']:>5.1f}% | Trades: {row['total_trades']:>4.0f}")
    
    # トータルリターン トップ10
    print("\n【トータルリターン Top 10】")
    print("-" * 70)
    top_return = results_df[results_df["total_trades"] >= 50].nlargest(10, "total_return_pct")
    for _, row in top_return.iterrows():
        print(f"  {row['symbol']:<8} × {row['strategy']:<15} | Ret: {row['total_return_pct']:>7.1f}% | "
              f"Sharpe: {row['sharpe_ratio']:>5.2f} | MDD: {row['max_drawdown_pct']:>6.1f}%")
    
    # 戦略別平均
    print("\n【戦略別平均パフォーマンス】")
    print("-" * 70)
    strategy_avg = results_df.groupby("strategy").agg({
        "win_rate_pct": "mean",
        "profit_factor": "mean",
        "total_return_pct": "mean",
        "total_trades": "mean",
    }).round(2)
    for strategy, row in strategy_avg.iterrows():
        print(f"  {strategy:<15} | WR: {row['win_rate_pct']:>5.1f}% | PF: {row['profit_factor']:>5.2f} | "
              f"Ret: {row['total_return_pct']:>7.1f}% | Avg Trades: {row['total_trades']:>5.0f}")
    
    # 通貨ペア別平均
    print("\n【通貨ペア別平均パフォーマンス】")
    print("-" * 70)
    symbol_avg = results_df.groupby("symbol").agg({
        "win_rate_pct": "mean",
        "profit_factor": "mean",
        "total_return_pct": "mean",
        "total_trades": "mean",
    }).round(2)
    for symbol, row in symbol_avg.iterrows():
        print(f"  {symbol:<8} | WR: {row['win_rate_pct']:>5.1f}% | PF: {row['profit_factor']:>5.2f} | "
              f"Ret: {row['total_return_pct']:>7.1f}% | Avg Trades: {row['total_trades']:>5.0f}")
    
    # 最良の組み合わせ推奨
    print("\n" + "=" * 80)
    print(" 推奨組み合わせ (PF > 1.3 かつ 取引数 >= 100)")
    print("=" * 80)
    
    recommended = results_df[
        (results_df["profit_factor"] > 1.3) & 
        (results_df["total_trades"] >= 100) &
        (results_df["profit_factor"] < 100)
    ].sort_values(["profit_factor", "win_rate_pct"], ascending=[False, False])
    
    if len(recommended) > 0:
        for _, row in recommended.head(15).iterrows():
            print(f"  {row['symbol']:<8} × {row['strategy']:<15} | "
                  f"PF: {row['profit_factor']:>5.2f} | WR: {row['win_rate_pct']:>5.1f}% | "
                  f"Ret: {row['total_return_pct']:>7.1f}% | Trades: {row['total_trades']:>4.0f} | "
                  f"MDD: {row['max_drawdown_pct']:>6.1f}%")
    else:
        print("  条件を満たす組み合わせがありません")
    
    print("\n" + "=" * 80)
    print(f" 結果保存先: {output_dir / 'multi_strategy_results.csv'}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
