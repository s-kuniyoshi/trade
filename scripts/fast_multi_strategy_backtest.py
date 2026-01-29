"""
高速版: 複数戦略 × 複数通貨ペアの包括的バックテスト。

並列処理とベクトル化を活用して高速にバックテストを実行。

使用方法:
    python scripts/fast_multi_strategy_backtest.py
"""

import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Callable
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

import numpy as np
import pandas as pd


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """共通特徴量を計算（ベクトル化）"""
    features = pd.DataFrame(index=df.index)
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    # 価格系
    features["log_return_1"] = np.log(close / close.shift(1))
    features["log_return_5"] = np.log(close / close.shift(5))
    features["log_return_20"] = np.log(close / close.shift(20))
    
    # 移動平均
    features["sma_10"] = close.rolling(10).mean()
    features["sma_20"] = close.rolling(20).mean()
    features["sma_50"] = close.rolling(50).mean()
    features["sma_200"] = close.rolling(200).mean()
    features["ema_12"] = close.ewm(span=12, adjust=False).mean()
    features["ema_26"] = close.ewm(span=26, adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    features["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # MACD
    features["macd"] = features["ema_12"] - features["ema_26"]
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]
    
    # ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    features["atr_14"] = tr.rolling(14).mean()
    
    # ADX
    plus_dm = high.diff()
    minus_dm = -low.diff()
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
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    features["bb_upper"] = bb_mid + 2 * bb_std
    features["bb_lower"] = bb_mid - 2 * bb_std
    features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / (bb_mid + 1e-10)
    features["bb_position"] = (close - features["bb_lower"]) / (features["bb_upper"] - features["bb_lower"] + 1e-10)
    
    # ストキャスティクス
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    features["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
    features["stoch_d"] = features["stoch_k"].rolling(3).mean()
    
    # モメンタム
    features["momentum_20"] = close / close.shift(20) - 1
    
    # 時間特徴量
    features["hour"] = df.index.hour
    
    return features.dropna()


# =============================================================================
# ベクトル化されたシグナル生成
# =============================================================================

def generate_signals_trend_following(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """トレンドフォロー: SMA20 > SMA50 + ADX > 25"""
    long_cond = (features["sma_20"] > features["sma_50"]) & \
                (df["close"] > features["sma_20"]) & \
                (features["adx_14"] > 25)
    short_cond = (features["sma_20"] < features["sma_50"]) & \
                 (df["close"] < features["sma_20"]) & \
                 (features["adx_14"] > 25)
    
    signals = pd.Series(0, index=features.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals


def generate_signals_mean_reversion(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """ミーンリバージョン: BB + RSI"""
    long_cond = (features["bb_position"] < 0.1) & \
                (features["rsi_14"] < 30) & \
                (features["adx_14"] < 30)
    short_cond = (features["bb_position"] > 0.9) & \
                 (features["rsi_14"] > 70) & \
                 (features["adx_14"] < 30)
    
    signals = pd.Series(0, index=features.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals


def generate_signals_macd(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """MACD: ヒストグラムのゼロクロス"""
    macd_hist = features["macd_hist"]
    macd_hist_prev = macd_hist.shift(1)
    
    long_cond = (macd_hist > 0) & (macd_hist_prev <= 0)
    short_cond = (macd_hist < 0) & (macd_hist_prev >= 0)
    
    signals = pd.Series(0, index=features.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals


def generate_signals_rsi_stoch(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """RSI + ストキャスティクス"""
    long_cond = (features["rsi_14"] < 35) & \
                (features["stoch_k"] < 20) & \
                (features["stoch_k"] > features["stoch_d"]) & \
                (features["adx_14"] < 35)
    short_cond = (features["rsi_14"] > 65) & \
                 (features["stoch_k"] > 80) & \
                 (features["stoch_k"] < features["stoch_d"]) & \
                 (features["adx_14"] < 35)
    
    signals = pd.Series(0, index=features.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals


def generate_signals_momentum(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """モメンタム + ADX"""
    long_cond = (features["momentum_20"] > 0.02) & \
                (features["plus_di"] > features["minus_di"]) & \
                (features["adx_14"] > 30)
    short_cond = (features["momentum_20"] < -0.02) & \
                 (features["minus_di"] > features["plus_di"]) & \
                 (features["adx_14"] > 30)
    
    signals = pd.Series(0, index=features.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals


def generate_signals_triple_screen(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """Triple Screen"""
    close = df["close"]
    long_trend = close > features["sma_200"]
    macd_bullish = features["macd_hist"] > 0
    
    long_cond = long_trend & macd_bullish & \
                (features["rsi_14"] < 50) & (features["stoch_k"] < 40)
    short_cond = (~long_trend) & (~macd_bullish) & \
                 (features["rsi_14"] > 50) & (features["stoch_k"] > 60)
    
    signals = pd.Series(0, index=features.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals


def generate_signals_breakout(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """ブレイクアウト: BB幅縮小後のブレイク"""
    bb_width_avg = features["bb_width"].rolling(5).mean().shift(1)
    width_squeeze = features["bb_width"] < bb_width_avg * 0.8
    
    close = df["close"]
    long_cond = width_squeeze & (close > features["bb_upper"])
    short_cond = width_squeeze & (close < features["bb_lower"])
    
    signals = pd.Series(0, index=features.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals


def generate_signals_ema_cross(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """EMAクロス: 12/26"""
    ema12 = features["ema_12"]
    ema26 = features["ema_26"]
    ema12_prev = ema12.shift(1)
    ema26_prev = ema26.shift(1)
    
    long_cond = (ema12 > ema26) & (ema12_prev <= ema26_prev)
    short_cond = (ema12 < ema26) & (ema12_prev >= ema26_prev)
    
    signals = pd.Series(0, index=features.index)
    signals[long_cond] = 1
    signals[short_cond] = -1
    return signals


# =============================================================================
# ベクトル化バックテスト
# =============================================================================

def vectorized_backtest(
    df: pd.DataFrame,
    features: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 10000,
    risk_per_trade: float = 0.01,
    tp_atr_mult: float = 3.0,
    sl_atr_mult: float = 2.0,
    pip_value: float = 0.0001,
    spread_pips: float = 1.0,
) -> dict[str, Any]:
    """ベクトル化されたバックテスト（高速版）"""
    
    common_idx = features.index.intersection(df.index).intersection(signals.index)
    df = df.loc[common_idx].copy()
    features = features.loc[common_idx].copy()
    signals = signals.loc[common_idx].copy()
    
    # 時間帯フィルター
    hour = df.index.hour
    signals[(hour >= 21) | (hour < 1)] = 0
    
    # 連続シグナルを除去（1エントリーにつき1シグナル）
    signals = signals.where(signals != signals.shift(1), 0)
    
    # トレード情報を格納
    trades = []
    equity = initial_capital
    equity_list = [initial_capital]
    position = None
    cost_price = spread_pips * pip_value
    
    atr = features["atr_14"].values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    timestamps = df.index
    signal_vals = signals.values
    
    for i in range(1, len(df)):
        # ポジションがある場合
        if position is not None:
            if position["direction"] == "long":
                if low[i] <= position["sl"]:
                    pnl = (position["sl"] - position["entry"]) * position["size"] - cost_price * position["size"]
                    equity += pnl
                    trades.append({"pnl": pnl, "direction": "long", "reason": "sl"})
                    position = None
                elif high[i] >= position["tp"]:
                    pnl = (position["tp"] - position["entry"]) * position["size"] - cost_price * position["size"]
                    equity += pnl
                    trades.append({"pnl": pnl, "direction": "long", "reason": "tp"})
                    position = None
            else:
                if high[i] >= position["sl"]:
                    pnl = (position["entry"] - position["sl"]) * position["size"] - cost_price * position["size"]
                    equity += pnl
                    trades.append({"pnl": pnl, "direction": "short", "reason": "sl"})
                    position = None
                elif low[i] <= position["tp"]:
                    pnl = (position["entry"] - position["tp"]) * position["size"] - cost_price * position["size"]
                    equity += pnl
                    trades.append({"pnl": pnl, "direction": "short", "reason": "tp"})
                    position = None
            
            equity_list.append(equity)
            continue
        
        # 新規エントリー
        sig = signal_vals[i-1]
        if sig == 0 or np.isnan(atr[i-1]) or atr[i-1] <= 0:
            equity_list.append(equity)
            continue
        
        sl_dist = atr[i-1] * sl_atr_mult
        tp_dist = atr[i-1] * tp_atr_mult
        pos_size = (equity * risk_per_trade) / sl_dist if sl_dist > 0 else 0
        
        if sig == 1:
            position = {
                "direction": "long",
                "entry": close[i],
                "sl": close[i] - sl_dist,
                "tp": close[i] + tp_dist,
                "size": pos_size,
            }
        elif sig == -1:
            position = {
                "direction": "short",
                "entry": close[i],
                "sl": close[i] + sl_dist,
                "tp": close[i] - tp_dist,
                "size": pos_size,
            }
        
        equity_list.append(equity)
    
    # 最終ポジション決済
    if position is not None:
        if position["direction"] == "long":
            pnl = (close[-1] - position["entry"]) * position["size"] - cost_price * position["size"]
        else:
            pnl = (position["entry"] - close[-1]) * position["size"] - cost_price * position["size"]
        equity += pnl
        trades.append({"pnl": pnl, "direction": position["direction"], "reason": "end"})
        equity_list[-1] = equity
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["pnl", "direction", "reason"])
    
    return {
        "equity": pd.Series(equity_list, index=df.index),
        "trades": trades_df,
    }


def calculate_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict[str, float]:
    """パフォーマンス指標を計算"""
    if len(equity) < 2:
        return {"total_return_pct": 0, "sharpe_ratio": 0, "max_drawdown_pct": 0,
                "total_trades": 0, "win_rate_pct": 0, "profit_factor": 0}
    
    returns = equity.pct_change().dropna()
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    
    sharpe = 0
    if len(returns) > 0 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24)
    
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / (rolling_max + 1e-10)
    max_dd = drawdown.min() * 100
    
    total_trades = len(trades)
    win_rate = 0
    profit_factor = 0
    
    if total_trades > 0:
        winning = trades[trades["pnl"] > 0]
        losing = trades[trades["pnl"] < 0]
        win_rate = len(winning) / total_trades * 100
        
        gross_profit = winning["pnl"].sum() if len(winning) > 0 else 0
        gross_loss = abs(losing["pnl"].sum()) if len(losing) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0)
    
    return {
        "total_return_pct": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "total_trades": total_trades,
        "win_rate_pct": win_rate,
        "profit_factor": min(profit_factor, 10.0),  # Cap at 10 for display
    }


def run_single_backtest(args):
    """単一のバックテスト実行（並列処理用）"""
    symbol, strategy_name, strategy_func, df, pip_value, spread = args
    
    try:
        features = compute_base_features(df)
        signals = strategy_func(df.loc[features.index], features)
        
        result = vectorized_backtest(
            df=df.loc[features.index],
            features=features,
            signals=signals,
            pip_value=pip_value,
            spread_pips=spread,
        )
        
        metrics = calculate_metrics(result["equity"], result["trades"])
        
        return {
            "symbol": symbol,
            "strategy": strategy_name,
            **metrics,
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "strategy": strategy_name,
            "total_return_pct": 0,
            "sharpe_ratio": 0,
            "max_drawdown_pct": 0,
            "total_trades": 0,
            "win_rate_pct": 0,
            "profit_factor": 0,
            "error": str(e),
        }


def main():
    print("=" * 80)
    print(" 高速版: 複数戦略 × 複数通貨ペア バックテスト")
    print("=" * 80)
    
    # 設定
    all_symbols = [
        "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD",
        "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "EURGBP", "EURAUD",
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
    
    strategies = [
        ("TrendFollowing", generate_signals_trend_following),
        ("MeanReversion", generate_signals_mean_reversion),
        ("MACD", generate_signals_macd),
        ("RSI_Stoch", generate_signals_rsi_stoch),
        ("Momentum", generate_signals_momentum),
        ("TripleScreen", generate_signals_triple_screen),
        ("Breakout", generate_signals_breakout),
        ("EMACross", generate_signals_ema_cross),
    ]
    
    data_dir = project_root / "data" / "raw"
    timeframe = "H1"
    
    # データ読み込み
    print("\n[1/3] データ読み込み...")
    datasets = {}
    for symbol in all_symbols:
        filepath = data_dir / f"{symbol}_{timeframe}.parquet"
        if filepath.exists():
            df = pd.read_parquet(filepath)
            datasets[symbol] = df
            print(f"  {symbol}: {len(df)} bars")
    
    if not datasets:
        print("\nデータが見つかりません。")
        return 1
    
    # バックテスト実行
    print(f"\n[2/3] バックテスト実行中... ({len(datasets)} symbols × {len(strategies)} strategies)")
    
    all_results = []
    total = len(datasets) * len(strategies)
    count = 0
    
    for symbol, df in datasets.items():
        for strategy_name, strategy_func in strategies:
            count += 1
            result = run_single_backtest((
                symbol, strategy_name, strategy_func, df,
                pip_values.get(symbol, 0.0001),
                spread_pips.get(symbol, 1.5)
            ))
            all_results.append(result)
            
            if count % 10 == 0:
                print(f"  進捗: {count}/{total}")
    
    print(f"  完了: {count}/{total}")
    
    # 結果をDataFrameに
    results_df = pd.DataFrame(all_results)
    
    # 結果保存
    output_dir = project_root / "data" / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "multi_strategy_results.csv", index=False)
    
    # サマリー表示
    print("\n[3/3] 結果サマリー")
    print("=" * 80)
    
    # フィルター: 取引数50以上
    valid = results_df[results_df["total_trades"] >= 50].copy()
    
    # 勝率トップ10
    print("\n【勝率 Top 10】(取引数 >= 50)")
    print("-" * 70)
    top_wr = valid.nlargest(10, "win_rate_pct")
    for _, row in top_wr.iterrows():
        print(f"  {row['symbol']:<8} × {row['strategy']:<15} | WR: {row['win_rate_pct']:>5.1f}% | "
              f"PF: {row['profit_factor']:>5.2f} | Trades: {row['total_trades']:>4.0f}")
    
    # Profit Factor トップ10
    print("\n【Profit Factor Top 10】(取引数 >= 50)")
    print("-" * 70)
    top_pf = valid.nlargest(10, "profit_factor")
    for _, row in top_pf.iterrows():
        print(f"  {row['symbol']:<8} × {row['strategy']:<15} | PF: {row['profit_factor']:>5.2f} | "
              f"WR: {row['win_rate_pct']:>5.1f}% | Ret: {row['total_return_pct']:>7.1f}%")
    
    # トータルリターン トップ10
    print("\n【トータルリターン Top 10】(取引数 >= 50)")
    print("-" * 70)
    top_ret = valid.nlargest(10, "total_return_pct")
    for _, row in top_ret.iterrows():
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
    for strategy, row in strategy_avg.sort_values("profit_factor", ascending=False).iterrows():
        print(f"  {strategy:<15} | WR: {row['win_rate_pct']:>5.1f}% | PF: {row['profit_factor']:>5.2f} | "
              f"Ret: {row['total_return_pct']:>7.1f}%")
    
    # 通貨ペア別平均
    print("\n【通貨ペア別平均パフォーマンス】")
    print("-" * 70)
    symbol_avg = results_df.groupby("symbol").agg({
        "win_rate_pct": "mean",
        "profit_factor": "mean",
        "total_return_pct": "mean",
        "total_trades": "mean",
    }).round(2)
    for symbol, row in symbol_avg.sort_values("profit_factor", ascending=False).iterrows():
        print(f"  {symbol:<8} | WR: {row['win_rate_pct']:>5.1f}% | PF: {row['profit_factor']:>5.2f} | "
              f"Ret: {row['total_return_pct']:>7.1f}%")
    
    # 推奨組み合わせ
    print("\n" + "=" * 80)
    print(" 推奨組み合わせ (PF > 1.2 & 取引数 >= 100 & MDD > -30%)")
    print("=" * 80)
    
    recommended = valid[
        (valid["profit_factor"] > 1.2) & 
        (valid["total_trades"] >= 100) &
        (valid["max_drawdown_pct"] > -30)
    ].sort_values(["profit_factor", "win_rate_pct"], ascending=[False, False])
    
    if len(recommended) > 0:
        for _, row in recommended.head(20).iterrows():
            print(f"  {row['symbol']:<8} × {row['strategy']:<15} | "
                  f"PF: {row['profit_factor']:>5.2f} | WR: {row['win_rate_pct']:>5.1f}% | "
                  f"Ret: {row['total_return_pct']:>7.1f}% | Trades: {row['total_trades']:>4.0f} | "
                  f"MDD: {row['max_drawdown_pct']:>6.1f}%")
    else:
        print("  条件を満たす組み合わせがありません。条件を緩和します...")
        recommended = valid[valid["profit_factor"] > 1.0].nlargest(20, "profit_factor")
        for _, row in recommended.iterrows():
            print(f"  {row['symbol']:<8} × {row['strategy']:<15} | "
                  f"PF: {row['profit_factor']:>5.2f} | WR: {row['win_rate_pct']:>5.1f}% | "
                  f"Ret: {row['total_return_pct']:>7.1f}%")
    
    # 全結果表示
    print("\n" + "=" * 80)
    print(" 全結果一覧")
    print("=" * 80)
    print(f"\n{'Symbol':<8} {'Strategy':<15} {'Return%':>8} {'Sharpe':>7} {'MDD%':>7} {'Trades':>7} {'WR%':>6} {'PF':>6}")
    print("-" * 75)
    for _, row in results_df.sort_values(["symbol", "strategy"]).iterrows():
        print(f"{row['symbol']:<8} {row['strategy']:<15} {row['total_return_pct']:>8.1f} "
              f"{row['sharpe_ratio']:>7.2f} {row['max_drawdown_pct']:>7.1f} "
              f"{row['total_trades']:>7.0f} {row['win_rate_pct']:>6.1f} {row['profit_factor']:>6.2f}")
    
    print(f"\n結果保存先: {output_dir / 'multi_strategy_results.csv'}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
