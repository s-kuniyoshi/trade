"""
MT5からヒストリカルデータをダウンロードして保存するスクリプト。

使用方法:
    python scripts/download_data.py

設定:
    - 通貨ペア: EURUSD, USDJPY, GBPUSD
    - 時間軸: H1, H4, D1
    - 期間: 1年分
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

import MetaTrader5 as mt5
import pandas as pd


def initialize_mt5(terminal_path: str | None = None) -> bool:
    """MT5を初期化する。"""
    if terminal_path:
        initialized = mt5.initialize(path=terminal_path)
    else:
        initialized = mt5.initialize()

    if not initialized:
        print(f"MT5初期化失敗: {mt5.last_error()}")
        return False
    
    # 接続情報を表示
    account_info = mt5.account_info()
    if account_info is not None:
        print(f"接続成功: {account_info.server}")
        print(f"アカウント: {account_info.login}")
        print(f"残高: {account_info.balance} {account_info.currency}")
    
    return True


def download_ohlcv(
    symbol: str,
    timeframe: int,
    timeframe_name: str,
    days: int = 365,
) -> pd.DataFrame | None:
    """
    MT5から指定期間のOHLCVデータをダウンロード。
    
    Args:
        symbol: 通貨ペア (例: "EURUSD")
        timeframe: MT5のタイムフレーム定数
        timeframe_name: 保存用の名前 (例: "H1")
        days: 取得日数
        
    Returns:
        OHLCVデータのDataFrame、失敗時はNone
    """
    # シンボルが利用可能か確認
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"  シンボル {symbol} が見つかりません")
        return None
    
    if not symbol_info.visible:
        if not mt5.symbol_select(symbol, True):
            print(f"  シンボル {symbol} を選択できません")
            return None
    
    # 期間を設定
    utc_to = datetime.utcnow()
    utc_from = utc_to - timedelta(days=days)
    
    print(f"  期間: {utc_from.date()} ~ {utc_to.date()}")
    
    # データ取得
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    
    if rates is None or len(rates) == 0:
        print(f"  データ取得失敗: {mt5.last_error()}")
        return None
    
    # DataFrameに変換
    df = pd.DataFrame(rates)
    
    # 時刻をDatetimeIndexに変換
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    
    # カラム名を標準化
    df.rename(columns={
        "tick_volume": "volume",
    }, inplace=True)
    
    # 必要なカラムのみ保持
    columns = ["open", "high", "low", "close", "volume", "spread"]
    df = df[[c for c in columns if c in df.columns]]
    
    print(f"  取得完了: {len(df)} 本")
    
    return df


def save_data(df: pd.DataFrame, symbol: str, timeframe: str, output_dir: Path) -> None:
    """データをParquet形式で保存。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{symbol}_{timeframe}.parquet"
    filepath = output_dir / filename
    
    df.to_parquet(filepath, compression="snappy")
    print(f"  保存完了: {filepath}")


def main():
    """メイン処理。"""
    print("=" * 60)
    print("MT5 ヒストリカルデータ ダウンローダー")
    print("=" * 60)
    
    # 設定
    symbols = ["EURUSD", "USDJPY", "GBPUSD"]
    timeframes = [
        (mt5.TIMEFRAME_H1, "H1"),
        (mt5.TIMEFRAME_H4, "H4"),
        (mt5.TIMEFRAME_D1, "D1"),
    ]
    days = 1825  # 5年分
    output_dir = project_root / "data" / "raw"
    
    # MT5初期化
    print("\n[1/3] MT5に接続中...")
    terminal_path = r"C:\Program Files\XMTrading MT5\terminal64.exe"
    if not initialize_mt5(terminal_path):
        print("MT5への接続に失敗しました。")
        print("MT5が起動しているか確認してください。")
        return 1
    
    try:
        # データダウンロード
        print(f"\n[2/3] データをダウンロード中...")
        print(f"  通貨ペア: {', '.join(symbols)}")
        print(f"  時間軸: {', '.join([tf[1] for tf in timeframes])}")
        print(f"  期間: {days}日分")
        print()
        
        success_count = 0
        total_count = len(symbols) * len(timeframes)
        
        for symbol in symbols:
            print(f"\n{symbol}:")
            for tf_const, tf_name in timeframes:
                print(f"  [{tf_name}]")
                df = download_ohlcv(symbol, tf_const, tf_name, days)
                
                if df is not None and len(df) > 0:
                    save_data(df, symbol, tf_name, output_dir)
                    success_count += 1
        
        # 結果サマリー
        print("\n" + "=" * 60)
        print(f"[3/3] 完了: {success_count}/{total_count} ファイル")
        print(f"保存先: {output_dir}")
        print("=" * 60)
        
        if success_count == total_count:
            print("\n次のステップ:")
            print("  python scripts/run_backtest.py")
            return 0
        else:
            print("\n一部のデータ取得に失敗しました。")
            return 1
            
    finally:
        mt5.shutdown()
        print("\nMT5接続を終了しました。")


if __name__ == "__main__":
    sys.exit(main())
