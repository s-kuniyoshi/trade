"""
MT5が利用できない場合のサンプルデータ生成スクリプト。

実際の市場データに近いランダムウォークデータを生成します。
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

import numpy as np
import pandas as pd


def generate_forex_data(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    periods: int,
    initial_price: float,
    volatility: float = 0.0001,
    trend: float = 0.0,
) -> pd.DataFrame:
    """
    FXデータをシミュレート生成。
    
    Args:
        symbol: 通貨ペア名
        timeframe: 時間軸
        start_date: 開始日時
        periods: 生成する足の数
        initial_price: 初期価格
        volatility: ボラティリティ（リターンの標準偏差）
        trend: トレンド（平均リターン）
    """
    np.random.seed(hash(symbol) % 2**32)
    
    # 時間インデックス生成
    if timeframe == "H1":
        freq = "h"
    elif timeframe == "H4":
        freq = "4h"
    elif timeframe == "D1":
        freq = "D"
    else:
        freq = "h"
    
    dates = pd.date_range(start=start_date, periods=periods, freq=freq, tz="UTC")
    
    # リターン生成（正規分布 + 若干の自己相関）
    returns = np.random.normal(trend, volatility, periods)
    
    # 価格パスを生成
    log_prices = np.cumsum(returns)
    close_prices = initial_price * np.exp(log_prices)
    
    # OHLCV生成
    data = []
    for i, (date, close) in enumerate(zip(dates, close_prices)):
        # 日中の変動をシミュレート
        daily_vol = volatility * np.sqrt(1)  # 1時間分のボラ
        
        # High/Low
        high_offset = abs(np.random.normal(0, daily_vol)) * close
        low_offset = abs(np.random.normal(0, daily_vol)) * close
        
        high = close + high_offset
        low = close - low_offset
        
        # Open（前のcloseに近い）
        if i == 0:
            open_price = initial_price
        else:
            gap = np.random.normal(0, volatility * 0.1) * close_prices[i-1]
            open_price = close_prices[i-1] + gap
        
        # 整合性チェック
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Volume（ランダム）
        volume = int(np.random.exponential(1000))
        
        # Spread（通貨ペアによって異なる）
        if "JPY" in symbol:
            spread = int(np.random.uniform(10, 30))  # pips * 10
        else:
            spread = int(np.random.uniform(5, 20))  # pips * 10
        
        data.append({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "spread": spread,
        })
    
    df = pd.DataFrame(data, index=dates)
    return df


def main():
    print("=" * 60)
    print(" サンプルデータ生成")
    print("=" * 60)
    
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 設定
    configs = [
        {"symbol": "EURUSD", "initial_price": 1.0850, "volatility": 0.0001},
        {"symbol": "USDJPY", "initial_price": 150.50, "volatility": 0.0001},
        {"symbol": "GBPUSD", "initial_price": 1.2650, "volatility": 0.00012},
    ]
    
    timeframes = ["H1", "H4", "D1"]
    periods_map = {"H1": 8760, "H4": 2190, "D1": 365}  # 約1年分
    
    start_date = datetime(2025, 1, 1, 0, 0, 0)
    
    print(f"\n生成設定:")
    print(f"  期間: {start_date.date()} から約1年分")
    print(f"  通貨ペア: {', '.join([c['symbol'] for c in configs])}")
    print(f"  時間軸: {', '.join(timeframes)}")
    
    for config in configs:
        symbol = config["symbol"]
        print(f"\n{symbol}:")
        
        for tf in timeframes:
            periods = periods_map[tf]
            
            df = generate_forex_data(
                symbol=symbol,
                timeframe=tf,
                start_date=start_date,
                periods=periods,
                initial_price=config["initial_price"],
                volatility=config["volatility"],
            )
            
            # 保存
            filepath = output_dir / f"{symbol}_{tf}.parquet"
            df.to_parquet(filepath, compression="snappy")
            
            print(f"  [{tf}] {len(df)} 本 -> {filepath.name}")
    
    print("\n" + "=" * 60)
    print(" 完了")
    print(f" 保存先: {output_dir}")
    print("=" * 60)
    print("\n次のステップ:")
    print("  python scripts/run_backtest.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
