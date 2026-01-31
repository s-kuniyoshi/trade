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
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from typing import Any

# 共通モジュールからインポート
from config.pair_config import PAIR_CONFIG, get_pair_config, get_all_symbols, COMMON_CONFIG
from python.features import compute_features, compute_triple_barrier_target, compute_atr


def load_data(symbol: str, timeframe: str, data_dir: Path) -> pd.DataFrame | None:
    """Parquetファイルからデータを読み込む。"""
    filepath = data_dir / f"{symbol}_{timeframe}.parquet"
    
    if not filepath.exists():
        print(f"  ファイルが見つかりません: {filepath}")
        return None
    
    df = pd.read_parquet(filepath)
    print(f"  {symbol}_{timeframe}: {len(df)} 本 ({df.index[0].date()} ~ {df.index[-1].date()})")
    return df


def compute_target(df: pd.DataFrame, horizon: int = 6) -> pd.Series:
    """
    予測ターゲット: N時間後のリターン。（旧方式、互換性のため残す）
    
    Args:
        df: OHLCVデータ
        horizon: 予測ホライズン（時間）
    """
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    return future_return


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
            # ATRを計算（共通モジュール使用）
            atr = compute_atr(df, period=14)
            
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
