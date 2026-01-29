"""
LightGBM MLモデルによる全通貨ペアバックテスト。

Walk-Forward検証で全通貨ペアをテスト。

使用方法:
    python scripts/ml_backtest_all_pairs.py
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Any
import warnings

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量を計算（既存のrun_backtest.pyから流用）"""
    features = pd.DataFrame(index=df.index)
    
    # 価格系
    features["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
    features["log_return_5"] = np.log(df["close"] / df["close"].shift(5))
    features["log_return_20"] = np.log(df["close"] / df["close"].shift(20))
    features["range_hl"] = (df["high"] - df["low"]) / df["close"]
    features["realized_vol_20"] = features["log_return_1"].rolling(20).std() * np.sqrt(252 * 24)
    
    # 移動平均
    features["sma_20"] = df["close"].rolling(20).mean()
    features["sma_50"] = df["close"].rolling(50).mean()
    features["sma_200"] = df["close"].rolling(200).mean()
    features["price_to_sma20"] = df["close"] / features["sma_20"] - 1
    features["price_to_sma50"] = df["close"] / features["sma_50"] - 1
    features["price_to_sma200"] = df["close"] / features["sma_200"] - 1
    features["sma_cross"] = (features["sma_20"] > features["sma_50"]).astype(int)
    
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
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
    features["sma200_atr_distance"] = abs(df["close"] - features["sma_200"]) / (features["atr_14"] + 1e-10)
    
    # ボリンジャーバンド
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    features["bb_upper"] = bb_mid + 2 * bb_std
    features["bb_lower"] = bb_mid - 2 * bb_std
    features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / (bb_mid + 1e-10)
    features["bb_position"] = (df["close"] - features["bb_lower"]) / (features["bb_upper"] - features["bb_lower"] + 1e-10)
    
    # 時間特徴量
    features["hour"] = df.index.hour
    features["day_of_week"] = df.index.dayofweek
    
    # 効率比
    n = 20
    close = df["close"]
    change = close.diff(n).abs()
    volatility = close.diff().abs().rolling(n).sum()
    features["efficiency_ratio"] = (change / (volatility + 1e-10)).replace([np.inf, -np.inf], np.nan)
    
    return features.dropna()


def compute_triple_barrier_target(
    df: pd.DataFrame,
    atr: pd.Series,
    tp_atr: float = 3.0,
    sl_atr: float = 2.0,
    max_hold: int = 48,
) -> pd.Series:
    """Triple-barrier方式のターゲット"""
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
        
        long_tp = entry + tp_atr * current_atr
        long_sl = entry - sl_atr * current_atr
        short_tp = entry - tp_atr * current_atr
        short_sl = entry + sl_atr * current_atr
        
        end = min(n, t + 1 + max_hold)
        if end <= t + 1:
            continue
        
        long_result = 0
        short_result = 0
        
        for i in range(t + 1, end):
            if long_result == 0:
                hit_tp = high[i] >= long_tp
                hit_sl = low[i] <= long_sl
                if hit_tp and hit_sl:
                    long_result = -1
                elif hit_tp:
                    long_result = 1
                elif hit_sl:
                    long_result = -1
            
            if short_result == 0:
                hit_tp = low[i] <= short_tp
                hit_sl = high[i] >= short_sl
                if hit_tp and hit_sl:
                    short_result = -1
                elif hit_tp:
                    short_result = 1
                elif hit_sl:
                    short_result = -1
            
            if long_result != 0 and short_result != 0:
                break
        
        if long_result == 1 and short_result != 1:
            targets[t] = 1.0
        elif short_result == 1 and long_result != 1:
            targets[t] = -1.0
        else:
            targets[t] = 0.0
    
    return pd.Series(targets, index=df.index, name="target")


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """LightGBMモデルを学習"""
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBMがインストールされていません。")
        sys.exit(1)
    
    # 3クラス分類
    y_class = (y_train + 1).astype(int)  # -1→0, 0→1, +1→2
    
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
    }
    
    train_data = lgb.Dataset(X_train, label=y_class)
    model = lgb.train(params, train_data, num_boost_round=200)
    
    return model


def backtest(
    df: pd.DataFrame,
    features: pd.DataFrame,
    model: Any,
    initial_capital: float = 10000,
    risk_per_trade: float = 0.01,
    min_confidence: float = 0.4,
    spread_pips: float = 1.0,
    slippage_pips: float = 0.2,
    pip_value: float = 0.0001,
    use_filters: bool = True,
    long_only: bool = False,
) -> dict[str, Any]:
    """バックテスト実行"""
    common_idx = features.index.intersection(df.index)
    df = df.loc[common_idx]
    features = features.loc[common_idx]
    
    raw_predictions = model.predict(features)
    
    # 3クラス確率から方向と信頼度を計算
    prob_sell = raw_predictions[:, 0]
    prob_neutral = raw_predictions[:, 1]
    prob_buy = raw_predictions[:, 2]
    
    direction_strength = np.maximum(prob_buy, prob_sell)
    edge_confidence = direction_strength - prob_neutral * 0.5
    predictions = np.where(prob_buy > prob_sell, edge_confidence, -edge_confidence)
    
    equity = [initial_capital]
    trades = []
    position = None
    cost_pips = spread_pips + slippage_pips
    cost_price = cost_pips * pip_value
    
    atr_col = features["atr_14"] if "atr_14" in features.columns else None
    
    for i in range(1, len(df)):
        current_price = df.iloc[i]["close"]
        timestamp = df.index[i]
        pred = predictions[i-1]
        
        if position is not None:
            if position["direction"] == "long":
                if df.iloc[i]["low"] <= position["sl"]:
                    pnl = (position["sl"] - position["entry"]) * position["size"]
                    pnl -= cost_price * position["size"]
                    equity.append(equity[-1] + pnl)
                    trades.append({"pnl": pnl, "direction": "long", "reason": "sl"})
                    position = None
                    continue
                elif df.iloc[i]["high"] >= position["tp"]:
                    pnl = (position["tp"] - position["entry"]) * position["size"]
                    pnl -= cost_price * position["size"]
                    equity.append(equity[-1] + pnl)
                    trades.append({"pnl": pnl, "direction": "long", "reason": "tp"})
                    position = None
                    continue
            else:
                if df.iloc[i]["high"] >= position["sl"]:
                    pnl = (position["entry"] - position["sl"]) * position["size"]
                    pnl -= cost_price * position["size"]
                    equity.append(equity[-1] + pnl)
                    trades.append({"pnl": pnl, "direction": "short", "reason": "sl"})
                    position = None
                    continue
                elif df.iloc[i]["low"] <= position["tp"]:
                    pnl = (position["entry"] - position["tp"]) * position["size"]
                    pnl -= cost_price * position["size"]
                    equity.append(equity[-1] + pnl)
                    trades.append({"pnl": pnl, "direction": "short", "reason": "tp"})
                    position = None
                    continue
            
            equity.append(equity[-1])
            continue
        
        # エントリー判定
        confidence = abs(pred)
        direction_signal = "long" if pred > 0 else "short"
        
        if confidence < min_confidence:
            equity.append(equity[-1])
            continue
        
        # Long-only制限
        if long_only and direction_signal == "short":
            equity.append(equity[-1])
            continue
        
        # 時間帯フィルター
        if use_filters:
            hour = timestamp.hour
            if hour >= 21 or hour < 1:
                equity.append(equity[-1])
                continue
            
            # ADXフィルター
            adx = features.iloc[i-1]["adx_14"] if "adx_14" in features.columns else 0
            if adx < 20:
                equity.append(equity[-1])
                continue
        
        # ATRベースのSL/TP
        atr = features.iloc[i-1]["atr_14"] if atr_col is not None else current_price * 0.001
        sl_distance = atr * 2
        tp_distance = atr * 3
        
        risk_amount = equity[-1] * risk_per_trade
        position_size = risk_amount / sl_distance if sl_distance > 0 else 0
        
        if direction_signal == "long":
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
        trades.append({"pnl": pnl, "direction": position["direction"], "reason": "end"})
    
    return {
        "equity": pd.Series(equity, index=df.index),
        "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
    }


def walk_forward_backtest(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    initial_capital: float = 10000,
    train_years: int = 2,
    step_months: int = 6,
    min_confidence: float = 0.4,
    spread_pips: float = 1.0,
    pip_value: float = 0.0001,
    long_only: bool = False,
) -> dict[str, Any]:
    """Walk-Forwardバックテスト"""
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
        model = train_model(X_train, y_train)
        result = backtest(
            df_test, X_test, model,
            initial_capital=current_equity,
            min_confidence=min_confidence,
            spread_pips=spread_pips,
            pip_value=pip_value,
            long_only=long_only,
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
    
    return {"equity": equity, "trades": trades}


def calculate_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict[str, float]:
    """パフォーマンス指標"""
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
        "profit_factor": min(profit_factor, 10.0),
    }


def main():
    print("=" * 80)
    print(" LightGBM ML戦略: 全通貨ペア Walk-Forward バックテスト")
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
    
    # Long-onlyシンボル（ショートにエッジがないペア）
    long_only_symbols = {"USDJPY"}
    
    data_dir = project_root / "data" / "raw"
    timeframe = "H1"
    initial_capital = 10000
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]
    
    # データ読み込み
    print("\n[1/3] データ読み込み...")
    datasets = {}
    for symbol in all_symbols:
        filepath = data_dir / f"{symbol}_{timeframe}.parquet"
        if filepath.exists():
            df = pd.read_parquet(filepath)
            datasets[symbol] = df
            print(f"  {symbol}: {len(df)} bars ({df.index[0].date()} ~ {df.index[-1].date()})")
    
    if not datasets:
        print("\nデータが見つかりません。")
        return 1
    
    # バックテスト実行
    print(f"\n[2/3] Walk-Forward バックテスト実行中...")
    print(f"  通貨ペア数: {len(datasets)}")
    print(f"  閾値候補: {thresholds}")
    
    all_results = []
    
    for symbol, df in datasets.items():
        print(f"\n{symbol}:")
        
        # 特徴量生成
        features = compute_features(df)
        
        # ターゲット計算
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift(1))
        tr3 = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        target = compute_triple_barrier_target(df, atr)
        
        # 共通インデックス
        common_idx = features.index.intersection(target.dropna().index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]
        
        pip_val = pip_values.get(symbol, 0.0001)
        spread = spread_pips.get(symbol, 1.5)
        long_only = symbol in long_only_symbols
        
        # 各閾値でテスト
        best_result = None
        best_pf = 0
        best_threshold = 0.4
        
        for thresh in thresholds:
            try:
                result = walk_forward_backtest(
                    df, X, y,
                    initial_capital=initial_capital,
                    min_confidence=thresh,
                    spread_pips=spread,
                    pip_value=pip_val,
                    long_only=long_only,
                )
                
                metrics = calculate_metrics(result["equity"], result["trades"])
                
                # PFが最大かつ取引数が十分な閾値を選択
                if metrics["profit_factor"] > best_pf and metrics["total_trades"] >= 50:
                    best_pf = metrics["profit_factor"]
                    best_threshold = thresh
                    best_result = metrics
                
                print(f"  閾値 {thresh:.2f}: PF={metrics['profit_factor']:.2f}, "
                      f"WR={metrics['win_rate_pct']:.1f}%, Trades={metrics['total_trades']:.0f}")
            except Exception as e:
                print(f"  閾値 {thresh:.2f}: エラー - {e}")
        
        if best_result:
            all_results.append({
                "symbol": symbol,
                "strategy": "ML_LightGBM",
                "threshold": best_threshold,
                "long_only": long_only,
                **best_result,
            })
            print(f"  >>> 最適閾値: {best_threshold:.2f}")
    
    # 結果をDataFrameに
    results_df = pd.DataFrame(all_results)
    
    # 結果保存
    output_dir = project_root / "data" / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "ml_strategy_results.csv", index=False)
    
    # サマリー表示
    print("\n[3/3] 結果サマリー")
    print("=" * 80)
    
    print("\n【ML戦略 全通貨ペア結果】")
    print("-" * 75)
    print(f"{'Symbol':<8} {'Thresh':>6} {'Return%':>9} {'Sharpe':>7} {'MDD%':>7} {'Trades':>7} {'WR%':>6} {'PF':>6}")
    print("-" * 75)
    
    for _, row in results_df.sort_values("profit_factor", ascending=False).iterrows():
        lo = " (L)" if row.get("long_only", False) else ""
        print(f"{row['symbol']:<8}{lo} {row['threshold']:>6.2f} {row['total_return_pct']:>9.1f} "
              f"{row['sharpe_ratio']:>7.2f} {row['max_drawdown_pct']:>7.1f} "
              f"{row['total_trades']:>7.0f} {row['win_rate_pct']:>6.1f} {row['profit_factor']:>6.2f}")
    
    # 推奨
    print("\n" + "=" * 80)
    print(" 推奨 (PF > 1.2 & 取引数 >= 100 & MDD > -25%)")
    print("=" * 80)
    
    recommended = results_df[
        (results_df["profit_factor"] > 1.2) & 
        (results_df["total_trades"] >= 100) &
        (results_df["max_drawdown_pct"] > -25)
    ].sort_values("profit_factor", ascending=False)
    
    if len(recommended) > 0:
        for _, row in recommended.iterrows():
            print(f"  {row['symbol']:<8} | PF: {row['profit_factor']:>5.2f} | "
                  f"WR: {row['win_rate_pct']:>5.1f}% | Ret: {row['total_return_pct']:>7.1f}% | "
                  f"MDD: {row['max_drawdown_pct']:>6.1f}%")
    else:
        print("  条件を満たす通貨ペアがありません。条件を緩和:")
        relaxed = results_df[results_df["profit_factor"] > 1.0].sort_values("profit_factor", ascending=False)
        for _, row in relaxed.head(5).iterrows():
            print(f"  {row['symbol']:<8} | PF: {row['profit_factor']:>5.2f} | "
                  f"WR: {row['win_rate_pct']:>5.1f}% | Ret: {row['total_return_pct']:>7.1f}%")
    
    # ルールベースとの比較用データ
    print(f"\n結果保存先: {output_dir / 'ml_strategy_results.csv'}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
