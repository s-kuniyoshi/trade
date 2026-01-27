"""
レバレッジ最適化分析

ケリー基準とモンテカルロシミュレーションを用いて
最適なレバレッジを計算する。
"""

import numpy as np
import pandas as pd
from pathlib import Path


def kelly_criterion(win_rate: float, win_loss_ratio: float) -> float:
    """
    ケリー基準によるベット比率計算。
    
    f* = (p * b - q) / b
    
    where:
        p = 勝率
        q = 1 - p (敗率)
        b = 平均利益 / 平均損失 (ペイオフレシオ)
    """
    p = win_rate
    q = 1 - p
    b = win_loss_ratio
    
    kelly = (p * b - q) / b
    return max(0, kelly)


def half_kelly(win_rate: float, win_loss_ratio: float) -> float:
    """ハーフケリー（より保守的）"""
    return kelly_criterion(win_rate, win_loss_ratio) / 2


def optimal_f(returns: np.ndarray) -> float:
    """
    Ralph Vinceの最適f計算。
    TWR (Terminal Wealth Relative) を最大化するf値を探索。
    """
    best_f = 0
    best_twr = 1
    
    # 最大損失を基準にする
    max_loss = abs(min(returns))
    if max_loss == 0:
        return 0
    
    for f in np.arange(0.01, 1.0, 0.01):
        twr = 1.0
        for r in returns:
            # HPR = 1 + f * (-return / max_loss)
            hpr = 1 + f * (r / max_loss)
            if hpr <= 0:
                twr = 0
                break
            twr *= hpr
        
        if twr > best_twr:
            best_twr = twr
            best_f = f
    
    return best_f


def simulate_leverage(
    returns: np.ndarray,
    leverage: float,
    initial_capital: float = 1000,
    n_simulations: int = 1000,
) -> dict:
    """
    モンテカルロシミュレーションでレバレッジの効果を分析。
    """
    n_trades = len(returns)
    final_equities = []
    max_drawdowns = []
    ruin_count = 0
    
    for _ in range(n_simulations):
        # リターンをシャッフル
        shuffled_returns = np.random.permutation(returns)
        
        equity = initial_capital
        peak = initial_capital
        max_dd = 0
        
        for r in shuffled_returns:
            # レバレッジ適用
            equity *= (1 + r * leverage)
            
            # 破産チェック
            if equity <= 0:
                ruin_count += 1
                equity = 0
                break
            
            # ドローダウン更新
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        
        final_equities.append(equity)
        max_drawdowns.append(max_dd)
    
    return {
        "mean_return": np.mean(final_equities) / initial_capital - 1,
        "median_return": np.median(final_equities) / initial_capital - 1,
        "std_return": np.std(final_equities) / initial_capital,
        "mean_max_dd": np.mean(max_drawdowns),
        "p95_max_dd": np.percentile(max_drawdowns, 95),
        "ruin_probability": ruin_count / n_simulations,
        "sharpe": (np.mean(final_equities) / initial_capital - 1) / (np.std(final_equities) / initial_capital + 1e-10),
    }


def main():
    print("=" * 60)
    print(" レバレッジ最適化分析")
    print("=" * 60)
    
    # バックテスト結果から取得した統計値
    # EURUSD: PF=1.08, 勝率=44.11%, 331取引
    # USDJPY: PF=1.14, 勝率=45.12%, 246取引
    
    results = {
        "EURUSD": {
            "win_rate": 0.4411,
            "pf": 1.08,
            "avg_win": 16.66,
            "avg_loss": 12.15,
            "total_trades": 331,
            "sharpe": 0.52,
            "max_dd": 0.1515,
        },
        "USDJPY": {
            "win_rate": 0.4512,
            "pf": 1.14,
            "avg_win": 16.32,
            "avg_loss": 11.73,
            "total_trades": 246,
            "sharpe": 0.68,
            "max_dd": 0.1796,
        },
    }
    
    for symbol, stats in results.items():
        print(f"\n{'='*60}")
        print(f" {symbol} レバレッジ分析")
        print(f"{'='*60}")
        
        win_rate = stats["win_rate"]
        win_loss_ratio = stats["avg_win"] / stats["avg_loss"]
        
        print(f"\n【基本統計】")
        print(f"  勝率: {win_rate*100:.2f}%")
        print(f"  ペイオフレシオ (平均利益/平均損失): {win_loss_ratio:.2f}")
        print(f"  Profit Factor: {stats['pf']:.2f}")
        print(f"  Sharpe Ratio: {stats['sharpe']:.2f}")
        print(f"  実績最大DD: {stats['max_dd']*100:.2f}%")
        
        # ケリー基準計算
        full_kelly = kelly_criterion(win_rate, win_loss_ratio)
        half = half_kelly(win_rate, win_loss_ratio)
        quarter = full_kelly / 4
        
        print(f"\n【ケリー基準】")
        print(f"  フルケリー: {full_kelly*100:.2f}% (理論上の最適値、実用は危険)")
        print(f"  ハーフケリー: {half*100:.2f}% (推奨上限)")
        print(f"  クォーターケリー: {quarter*100:.2f}% (保守的推奨)")
        
        # トレードごとのリターンをシミュレート
        n_wins = int(stats["total_trades"] * win_rate)
        n_losses = stats["total_trades"] - n_wins
        
        # 1%リスクでのリターン（基準）
        base_risk = 0.01
        win_return = base_risk * (stats["avg_win"] / stats["avg_loss"])  # 勝ちトレードのリターン
        loss_return = -base_risk  # 負けトレードのリターン
        
        returns = np.concatenate([
            np.full(n_wins, win_return),
            np.full(n_losses, loss_return)
        ])
        
        # 最適f計算
        opt_f = optimal_f(returns)
        
        print(f"\n【最適f (Ralph Vince)】")
        print(f"  最適f: {opt_f*100:.2f}% (TWR最大化)")
        print(f"  ハーフ最適f: {opt_f*50:.2f}% (推奨)")
        
        # レバレッジごとのシミュレーション
        print(f"\n【モンテカルロシミュレーション】(1000回)")
        print(f"  {'レバ':>6} {'平均リターン':>12} {'中央値':>10} {'平均DD':>10} {'95%DD':>10} {'破産確率':>10}")
        print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        
        best_sharpe = 0
        best_leverage = 1
        
        for lev in [1, 2, 3, 5, 10, 15, 20, 25]:
            sim = simulate_leverage(returns, lev)
            
            if sim["sharpe"] > best_sharpe and sim["ruin_probability"] < 0.01:
                best_sharpe = sim["sharpe"]
                best_leverage = lev
            
            print(f"  {lev:>6}x {sim['mean_return']*100:>11.1f}% {sim['median_return']*100:>9.1f}% "
                  f"{sim['mean_max_dd']*100:>9.1f}% {sim['p95_max_dd']*100:>9.1f}% "
                  f"{sim['ruin_probability']*100:>9.2f}%")
        
        print(f"\n【推奨レバレッジ】")
        print(f"  保守的 (初心者向け): 1-3倍")
        print(f"  標準 (経験者向け): 3-5倍")
        print(f"  積極的 (上級者向け): 5-10倍")
        print(f"  シミュレーション最適 (破産率<1%): {best_leverage}倍")
        
        # リスクパーセント換算
        print(f"\n【1トレードあたりリスク (ATR 2倍 SL基準)】")
        print(f"  現在: 1% (レバレッジ1倍相当)")
        print(f"  レバ3倍: 3% リスク/トレード")
        print(f"  レバ5倍: 5% リスク/トレード")
        print(f"  レバ10倍: 10% リスク/トレード")
    
    print(f"\n{'='*60}")
    print(" 総合推奨")
    print(f"{'='*60}")
    print("""
【結論】

1. 初心者・安全重視: レバレッジ 2-3倍
   - 1トレードリスク: 2-3%
   - 年間DD目安: 20-30%
   - 精神的負担が少ない

2. 経験者・バランス: レバレッジ 5倍
   - 1トレードリスク: 5%
   - 年間DD目安: 30-40%
   - リターンとリスクのバランス良好

3. 上級者・積極運用: レバレッジ 10倍
   - 1トレードリスク: 10%
   - 年間DD目安: 50-60%
   - 高リターンだが精神的負担大

【注意事項】
- 上記は理論値。実運用では想定外の事態が起こる
- 最初は低レバレッジで始め、実績を積んでから上げる
- 生活資金でのトレードは絶対に避ける
- 最大DDの2倍は覚悟しておく
""")


if __name__ == "__main__":
    main()
