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
    features["price_to_sma20"] = df["close"] / features["sma_20"] - 1
    features["price_to_sma50"] = df["close"] / features["sma_50"] - 1
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
    
    # 欠損値を削除
    features = features.dropna()
    
    return features


def compute_target(df: pd.DataFrame, horizon: int = 6) -> pd.Series:
    """
    予測ターゲット: N時間後のリターン。
    
    Args:
        df: OHLCVデータ
        horizon: 予測ホライズン（時間）
    """
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    return future_return


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """LightGBMモデルを学習。"""
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBMがインストールされていません。")
        print("pip install lightgbm を実行してください。")
        sys.exit(1)
    
    # 分類問題として扱う（上昇/下降）
    y_class = (y_train > 0).astype(int)
    
    params = {
        "objective": "binary",
        "metric": "auc",
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
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
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
    """
    # 共通インデックスに揃える
    common_idx = features.index.intersection(df.index)
    df = df.loc[common_idx]
    features = features.loc[common_idx]
    
    # 予測
    predictions = model.predict(features)
    
    # 結果格納
    equity = [initial_capital]
    trades = []
    position = None
    cost_pips = spread_pips + slippage_pips
    cost_price = cost_pips * pip_value
    
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
        
        # 新規エントリー判定
        confidence = max(pred, 1 - pred)
        if confidence < min_confidence:
            equity.append(equity[-1])
            continue
        
        # ATRベースのSL/TP
        atr = features.iloc[i-1]["atr_14"] if "atr_14" in features.columns else current_price * 0.001
        sl_distance = atr * 2
        tp_distance = atr * 3
        
        # ポジションサイズ計算
        risk_amount = equity[-1] * risk_per_trade
        position_size = risk_amount / sl_distance if sl_distance > 0 else 0
        
        if pred > 0.5:  # Long signal
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
        model = train_model(X_train, y_train)
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
    print(" FX自動売買システム - フル検証パイプライン")
    print("=" * 60)
    
    # 設定
    symbols = ["EURUSD", "USDJPY", "GBPUSD"]
    timeframe = "H1"
    data_dir = project_root / "data" / "raw"
    initial_capital = 1000
    full_period = False
    walk_forward = True
    train_years = 2
    step_months = 6
    spread_pips_by_symbol = {
        "EURUSD": 1.0,
        "USDJPY": 1.5,
        "GBPUSD": 1.8,
    }
    slippage_pips = 0.2
    pip_value_by_symbol = {
        "EURUSD": 0.0001,
        "USDJPY": 0.01,
        "GBPUSD": 0.0001,
    }
    
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
        
        # 特徴量生成
        print("\n[2/4] 特徴量生成...")
        features = compute_features(df)
        print(f"  特徴量数: {len(features.columns)}")
        print(f"  サンプル数: {len(features)}")
        
        # ターゲット計算
        target = compute_target(df, horizon=6)
        
        # 共通インデックス
        common_idx = features.index.intersection(target.dropna().index)
        X = features.loc[common_idx]
        y = target.loc[common_idx]
        
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
        model = train_model(X_train, y_train)
        
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
        spread_pips = spread_pips_by_symbol.get(symbol, 1.0)
        pip_value = pip_value_by_symbol.get(symbol, 0.0001)
        if walk_forward:
            result = walk_forward_backtest(
                df,
                X,
                y,
                initial_capital=initial_capital,
                risk_per_trade=0.01,
                min_confidence=0.6,
                train_years=train_years,
                step_months=step_months,
                spread_pips=spread_pips,
                slippage_pips=slippage_pips,
                pip_value=pip_value,
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
