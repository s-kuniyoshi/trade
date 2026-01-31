"""
FX自動売買スクリプト

バックテストで確立した戦略をMT5で実行する。
デモ口座・ライブ口座の両方に対応。

主な機能:
- MT5からリアルタイムデータ取得
- LightGBMモデルによるシグナル生成
- ニュースフィルター（重要指標前後の取引停止）
- リスク管理（DD制限、連敗制限、ボラティリティ調整）
- 自動売買（エントリー・決済）

使用方法:
    python scripts/run_trading.py
    または
    start_trading.bat をダブルクリック
"""

import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 共通モジュールからインポート
from config.pair_config import PAIR_CONFIG as BASE_PAIR_CONFIG, get_pair_config as get_base_pair_config, COMMON_CONFIG
from python.features import compute_features

# ログ設定
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"demo_trading_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
except ImportError:
    logger.error("MetaTrader5パッケージをインストールしてください: pip install MetaTrader5")
    sys.exit(1)

import lightgbm as lgb


# =============================================================================
# 設定
# =============================================================================

# =============================================================================
# 推奨戦略設定（バックテスト結果に基づく）
# =============================================================================
# バックテスト結果:
# - USDJPY × EMACross: PF 1.08, WR 43.2%, Return 27.2%
# - EURJPY × ML_LightGBM: PF 1.05, WR 43.0%, Return 22.3%  
# - AUDUSD × TripleScreen: PF 1.11, WR 45.1%, Return 9.7%
# - GBPUSD × RSI_Stoch: PF 1.02, WR 42.9%, Return 6.3%

# =============================================================================
# 通貨ペア別設定（バックテスト結果に基づく最適化）
# =============================================================================
# 各ペアに最も成績の良い単一戦略を割り当て
# H1バックテスト結果:
# - USDJPY × EMACross: PF 1.08, Return +27.2%
# M30 Walk-Forward バックテスト結果 (2026-01-31):
# - GBPUSD × ML_Primary: PF 1.17, Return +11.2% ✅
# - USDJPY × EMACross (ルールベース): ML PF 0.92 → ルールベースに変更
# - AUDUSD × TripleScreen (ルールベース): ML PF 0.94 → ルールベースに変更
# - EURJPY/EURUSD: 除外（成績不良）

# 取引固有の設定（戦略情報など）を共通設定に追加
# BASE_PAIR_CONFIGをベースに、取引固有フィールドを追加
TRADING_PAIR_CONFIG = {
    # 取引対象ペアのみ定義（BASE_PAIR_CONFIGの設定をオーバーライド）
    "USDCAD": {
        **get_base_pair_config("USDCAD"),
        "strategy": "ML_Primary",     # 最優秀（PF 1.28@0.60）
        "use_ml_fallback": True,
    },
    "GBPUSD": {
        **get_base_pair_config("GBPUSD"),
        "strategy": "ML_Primary",     # 優秀（PF 1.20@0.50）
        "use_ml_fallback": True,
    },
    "USDJPY": {
        **get_base_pair_config("USDJPY"),
        "strategy": "ML_Primary",     # 安定（PF 1.08@0.50）
        "use_ml_fallback": True,
    },
    "AUDJPY": {
        **get_base_pair_config("AUDJPY"),
        "strategy": "ML_Primary",     # 多取引（PF 1.02@0.50）
        "use_ml_fallback": True,
    },
}

# 後方互換性のため PAIR_CONFIG エイリアスを維持
PAIR_CONFIG = TRADING_PAIR_CONFIG

# 後方互換性のためSYMBOL_STRATEGIESも維持
SYMBOL_STRATEGIES = {pair: [cfg["strategy"]] for pair, cfg in TRADING_PAIR_CONFIG.items()}

CONFIG = {
    # 取引対象（M30最適化ポートフォリオ - 4通貨ペア）
    # 2026-01-31更新: 年間164取引、トータルリターン+238%
    "symbols": ["USDCAD", "GBPUSD", "USDJPY", "AUDJPY"],
    "timeframe": "M30",  # M30（取引頻度向上 + 精度向上）
    
    # モデル設定
    "model_path": project_root / "models",
    
    # 取引設定（M30最適化 - バックテストと同期）
    "initial_capital": 10000.0,
    "risk_per_trade": 0.03,   # バックテストと同期（3%）
    "leverage": 25,           # フォールバック（PAIR_CONFIGで個別設定）
    "min_confidence": 0.50,   # PAIR_CONFIGで個別設定（0.50-0.60）、これはフォールバック
    
    # SL/TP設定（バックテストと同期）
    "sl_atr_mult": 1.5,       # 2.0→1.5（バックテストと同期）
    "tp_atr_mult": 2.5,       # 3.0→2.5（バックテストと同期）
    
    # フィルター設定（PAIR_CONFIGで個別設定、これはフォールバック）
    "use_filters": True,
    "adx_threshold": 15.0,      # PAIR_CONFIGで12-18を使用
    "sma_atr_threshold": 0.3,   # PAIR_CONFIGで0.2-0.4を使用
    
    # リスク管理
    "max_drawdown_pct": 0.25,
    "consecutive_loss_limit": 5,
    "vol_scale_threshold": 1.5,
    
    # シンボル別制限（PAIR_CONFIGと整合性を取る）
    "long_only_symbols": {"USDJPY"},  # USDJPYはロングのみ（H1分析で有効）
    
    # スプレッド
    "max_spread_pips": {
        "EURUSD": 2.0,
        "USDJPY": 2.0,
        "EURJPY": 2.5,
        "AUDUSD": 2.0,
        "GBPUSD": 2.5,
        "USDCAD": 2.5,
        "AUDJPY": 3.0,  # クロス円は広め
    },
    
    # pip値
    "pip_value": {
        "EURUSD": 0.0001,
        "USDJPY": 0.01,
        "EURJPY": 0.01,
        "AUDUSD": 0.0001,
        "GBPUSD": 0.0001,
        "USDCAD": 0.0001,
        "AUDJPY": 0.01,  # JPYペアは0.01
    },
    
    # 実行間隔
    "check_interval_seconds": 60,
    
    # デバッグ
    "dry_run": False,
}


# =============================================================================
# 特徴量計算: python/features.py からインポート済み
# =============================================================================


# =============================================================================
# MT5操作
# =============================================================================

def initialize_mt5() -> bool:
    """MT5を初期化。"""
    if not mt5.initialize():
        logger.error(f"MT5初期化失敗: {mt5.last_error()}")
        return False
    
    account_info = mt5.account_info()
    if account_info is None:
        logger.error("アカウント情報取得失敗")
        return False
    
    logger.info(f"MT5接続成功")
    logger.info(f"  口座番号: {account_info.login}")
    logger.info(f"  サーバー: {account_info.server}")
    logger.info(f"  残高: {account_info.balance:.2f} {account_info.currency}")
    logger.info(f"  レバレッジ: 1:{account_info.leverage}")
    
    return True


def get_ohlcv(symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
    """MT5からOHLCVデータを取得。"""
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    
    mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
    
    if rates is None or len(rates) == 0:
        logger.warning(f"データ取得失敗: {symbol} {timeframe}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "volume"})
    
    return df[["open", "high", "low", "close", "volume"]]


def get_current_spread(symbol: str) -> float:
    """現在のスプレッドをpipsで取得。"""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return 999.0
    
    pip_val = CONFIG["pip_value"].get(symbol, 0.0001)
    spread = (tick.ask - tick.bid) / pip_val
    return spread


def place_order(
    symbol: str,
    direction: str,
    lots: float,
    sl: float,
    tp: float,
) -> dict | None:
    """注文を発注。"""
    if CONFIG["dry_run"]:
        logger.info(f"[DRY RUN] 注文: {symbol} {direction} {lots:.2f}lot SL={sl:.5f} TP={tp:.5f}")
        return {"ticket": -1, "dry_run": True}
    
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"ティック取得失敗: {symbol}")
        return None
    
    price = tick.ask if direction == "buy" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if direction == "buy" else mt5.ORDER_TYPE_SELL
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lots,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 123456,
        "comment": "fx_ml_demo",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"注文失敗: {result.comment}")
        return None
    
    logger.info(f"★★★ 注文成功: {symbol} {direction} {lots:.2f}lot @ {price:.5f} ★★★")
    return {"ticket": result.order, "price": price}


def get_open_positions(symbol: str | None = None) -> list:
    """オープンポジションを取得。"""
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()
    
    if positions is None:
        return []
    
    return list(positions)


# =============================================================================
# 予測とシグナル生成
# =============================================================================

class DemoTrader:
    """デモ取引を管理するクラス。"""
    
    def __init__(self, config: dict):
        self.config = config
        self.models: dict[str, lgb.Booster] = {}
        self.feature_columns: dict[str, list] = {}
        
        # リスク管理状態
        self.peak_equity = config["initial_capital"]
        self.current_equity = config["initial_capital"]
        self.consecutive_losses: dict[str, int] = {s: 0 for s in config["symbols"]}
        self.trading_halted = False
        self.avg_atr: dict[str, float] = {}
        
        # ニュースフィルター初期化
        self.news_filter = None
        try:
            from python.src.news.news_filter import NewsFilter, NewsFilterConfig
            from python.src.news.news_provider import EventImpact
            
            news_config = NewsFilterConfig(
                enabled=True,
                blackout_before_minutes=30,
                blackout_after_minutes=15,
                min_impact=EventImpact.HIGH,
            )
            self.news_filter = NewsFilter(
                config=news_config,
                symbols=config["symbols"],
            )
            self.news_filter.start_auto_refresh()
            logger.info("ニュースフィルター初期化完了")
        except Exception as e:
            logger.warning(f"ニュースフィルター初期化失敗（続行）: {e}")
    
    # =========================================================================
    # ルールベース戦略
    # =========================================================================
    
    def strategy_ema_cross(self, features: pd.DataFrame) -> dict | None:
        """
        EMACross戦略: EMA12/26クロス
        - USDJPY向け（最高リターン）
        """
        if len(features) < 2:
            return None
        
        f = features.iloc[-1]
        f_prev = features.iloc[-2]
        
        # EMAを計算（特徴量から）
        ema_12 = f.get("ema_12") if "ema_12" in features.columns else None
        ema_26 = f.get("ema_26") if "ema_26" in features.columns else None
        
        if ema_12 is None or ema_26 is None:
            # EMAがない場合はMACD代用
            macd_hist = f["macd_hist"]
            macd_hist_prev = f_prev["macd_hist"]
            
            if macd_hist > 0 and macd_hist_prev <= 0:
                return {"direction": "buy", "confidence": 0.6}
            elif macd_hist < 0 and macd_hist_prev >= 0:
                return {"direction": "sell", "confidence": 0.6}
            return None
        
        ema_12_prev = f_prev.get("ema_12", ema_12)
        ema_26_prev = f_prev.get("ema_26", ema_26)
        
        # クロス判定
        if ema_12 > ema_26 and ema_12_prev <= ema_26_prev:
            return {"direction": "buy", "confidence": 0.65}
        elif ema_12 < ema_26 and ema_12_prev >= ema_26_prev:
            return {"direction": "sell", "confidence": 0.65}
        
        return None
    
    def strategy_triple_screen(self, df: pd.DataFrame, features: pd.DataFrame) -> dict | None:
        """
        TripleScreen戦略（エルダー方式・緩和版）
        - AUDUSD向け（低MDD、安定）
        - 条件を緩和して取引頻度向上
        """
        if len(features) < 1:
            return None
        
        f = features.iloc[-1]
        close = df.iloc[-1]["close"]
        
        # Screen 1: 長期トレンド（SMA200）
        sma_200 = f.get("sma_200")
        if sma_200 is None or pd.isna(sma_200):
            return None
        long_trend = close > sma_200
        
        # Screen 2: MACD方向（緩和: ヒストグラム反転も許可）
        macd_hist = f.get("macd_hist", 0)
        macd_signal = f.get("macd_signal", 0)
        macd = f.get("macd", 0)
        macd_bullish = macd_hist > 0 or (macd > macd_signal)  # 条件緩和
        macd_bearish = macd_hist < 0 or (macd < macd_signal)
        
        # Screen 3: オシレーター（閾値緩和: 50/40 → 55/50, 50/60 → 45/50）
        rsi = f.get("rsi_14", 50)
        stoch_k = f.get("stoch_k") if "stoch_k" in features.columns else 50
        
        if long_trend and macd_bullish and rsi < 55 and stoch_k < 50:
            confidence = min(0.70, 0.50 + (55 - rsi) / 110)
            return {"direction": "buy", "confidence": confidence}
        elif not long_trend and macd_bearish and rsi > 45 and stoch_k > 50:
            confidence = min(0.70, 0.50 + (rsi - 45) / 110)
            return {"direction": "sell", "confidence": confidence}
        
        return None
    
    def strategy_rsi_stoch(self, features: pd.DataFrame) -> dict | None:
        """
        RSI + ストキャスティクス戦略（緩和版）
        - GBPUSD向け（レンジ相場）
        - 閾値を緩和して取引頻度向上
        """
        if len(features) < 1:
            return None
        
        f = features.iloc[-1]
        
        rsi = f.get("rsi_14", 50)
        stoch_k = f.get("stoch_k") if "stoch_k" in features.columns else 50
        stoch_d = f.get("stoch_d") if "stoch_d" in features.columns else 50
        adx = f.get("adx_14", 30)
        
        # レンジ相場でのみ有効（閾値緩和: 35 → 40）
        if adx > 40:
            return None
        
        # 閾値緩和: RSI 35/65 → 40/60, Stoch 20/80 → 25/75
        if rsi < 40 and stoch_k < 25 and stoch_k > stoch_d:
            confidence = min(0.75, 0.45 + (40 - rsi) / 80 + (25 - stoch_k) / 50)
            return {"direction": "buy", "confidence": confidence}
        elif rsi > 60 and stoch_k > 75 and stoch_k < stoch_d:
            confidence = min(0.75, 0.45 + (rsi - 60) / 80 + (stoch_k - 75) / 50)
            return {"direction": "sell", "confidence": confidence}
        
        return None
    
    def strategy_breakout(self, df: pd.DataFrame, features: pd.DataFrame) -> dict | None:
        """
        Breakout戦略: BB幅縮小後のブレイク
        - EURJPY向け（PF 1.10、バックテスト2位）
        
        ボリンジャーバンド幅が縮小した後、価格がバンド外に出たらエントリー。
        ボラティリティ収縮後のブレイクアウトを狙う。
        """
        if len(features) < 6:
            return None
        
        f = features.iloc[-1]
        close = df.iloc[-1]["close"]
        
        # BB関連の特徴量を取得
        bb_width = f.get("bb_width")
        bb_upper = f.get("bb_upper")
        bb_lower = f.get("bb_lower")
        
        if bb_width is None or bb_upper is None or bb_lower is None:
            return None
        if pd.isna(bb_width) or pd.isna(bb_upper) or pd.isna(bb_lower):
            return None
        
        # 過去5本のBB幅平均を計算
        bb_width_series = features["bb_width"].iloc[-6:-1]
        if len(bb_width_series) < 5:
            return None
        bb_width_avg = bb_width_series.mean()
        
        # スクイーズ判定: 現在のBB幅が平均の80%未満
        width_squeeze = bb_width < bb_width_avg * 0.8
        
        if not width_squeeze:
            return None
        
        # ブレイクアウト判定
        if close > bb_upper:
            # スクイーズの深さに応じて信頼度を調整
            squeeze_ratio = bb_width / bb_width_avg
            confidence = min(0.75, 0.55 + (0.8 - squeeze_ratio) * 0.5)
            return {"direction": "buy", "confidence": confidence}
        elif close < bb_lower:
            squeeze_ratio = bb_width / bb_width_avg
            confidence = min(0.75, 0.55 + (0.8 - squeeze_ratio) * 0.5)
            return {"direction": "sell", "confidence": confidence}
        
        return None
    
    def strategy_momentum(self, df: pd.DataFrame, features: pd.DataFrame) -> dict | None:
        """
        Momentum戦略: トレンド方向への継続エントリー
        - 強いトレンド相場で有効
        - ADX高め + 価格の勢いを確認
        """
        if len(features) < 5:
            return None
        
        f = features.iloc[-1]
        close = df.iloc[-1]["close"]
        
        # ADXでトレンド強度を確認（閾値: 25以上で強トレンド）
        adx = f.get("adx_14", 0)
        if adx < 25:
            return None
        
        # SMA20/50のトレンド方向
        sma_20 = f.get("sma_20")
        sma_50 = f.get("sma_50")
        if sma_20 is None or sma_50 is None:
            return None
        if pd.isna(sma_20) or pd.isna(sma_50):
            return None
        
        # 過去5本のリターンを確認（モメンタム）
        log_ret_5 = f.get("log_return_5", 0)
        
        # 効率比率（クリーンなトレンドかどうか）
        efficiency = f.get("efficiency_ratio", 0.5)
        if pd.isna(efficiency):
            efficiency = 0.5
        
        # 上昇トレンド + 正のモメンタム
        if sma_20 > sma_50 and close > sma_20 and log_ret_5 > 0 and efficiency > 0.4:
            confidence = min(0.70, 0.40 + adx / 100 + efficiency * 0.2)
            return {"direction": "buy", "confidence": confidence}
        # 下降トレンド + 負のモメンタム
        elif sma_20 < sma_50 and close < sma_20 and log_ret_5 < 0 and efficiency > 0.4:
            confidence = min(0.70, 0.40 + adx / 100 + efficiency * 0.2)
            return {"direction": "sell", "confidence": confidence}
        
        return None
    
    def strategy_mean_reversion(self, df: pd.DataFrame, features: pd.DataFrame) -> dict | None:
        """
        MeanReversion戦略: 平均回帰を狙う逆張り
        - レンジ相場で有効
        - BBからの乖離 + RSI極値を確認
        """
        if len(features) < 1:
            return None
        
        f = features.iloc[-1]
        close = df.iloc[-1]["close"]
        
        # レンジ相場判定（ADX低め）
        adx = f.get("adx_14", 30)
        if adx > 30:  # トレンド相場では使わない
            return None
        
        # Zスコア（平均からの乖離）
        zscore = f.get("zscore", 0)
        if pd.isna(zscore):
            return None
        
        # ボリンジャーバンドポジション
        bb_position = f.get("bb_position", 0.5)
        if pd.isna(bb_position):
            bb_position = 0.5
        
        # RSI
        rsi = f.get("rsi_14", 50)
        
        # BB幅パーセンタイル（低い = 収縮中）
        bw_pct = f.get("bw_percentile", 0.5)
        if pd.isna(bw_pct):
            bw_pct = 0.5
        
        # 売られすぎからの反発を狙う
        if zscore < -1.5 and bb_position < 0.15 and rsi < 35:
            confidence = min(0.65, 0.35 + abs(zscore) * 0.1 + (35 - rsi) / 100)
            return {"direction": "buy", "confidence": confidence}
        # 買われすぎからの反落を狙う
        elif zscore > 1.5 and bb_position > 0.85 and rsi > 65:
            confidence = min(0.65, 0.35 + abs(zscore) * 0.1 + (rsi - 65) / 100)
            return {"direction": "sell", "confidence": confidence}
        
        return None
        
    def load_models(self) -> bool:
        """モデルを読み込む。"""
        model_path = self.config["model_path"]
        
        for symbol in self.config["symbols"]:
            model_file = model_path / f"{symbol}_model.txt"
            features_file = model_path / f"{symbol}_features.txt"
            
            if not model_file.exists():
                logger.error(f"モデルファイルが見つかりません: {model_file}")
                logger.error("先にモデルを学習してください。")
                return False
            
            self.models[symbol] = lgb.Booster(model_file=str(model_file))
            logger.info(f"モデル読み込み: {symbol}")
            
            # 特徴量リストを読み込み
            if features_file.exists():
                with open(features_file) as f:
                    self.feature_columns[symbol] = [line.strip() for line in f]
            
        return True
    
    def train_and_save_models(self) -> bool:
        """モデルを学習して保存（初回用）。"""
        from scripts.run_backtest import compute_triple_barrier_target, train_model, load_data
        
        model_path = self.config["model_path"]
        model_path.mkdir(parents=True, exist_ok=True)
        
        for symbol in self.config["symbols"]:
            logger.info(f"{symbol} のモデルを学習中...")
            
            # データ読み込み
            data_dir = project_root / "data" / "raw"
            df = load_data(symbol, self.config["timeframe"], data_dir)
            
            if df is None or len(df) < 1000:
                logger.warning(f"データ不足: {symbol}")
                continue
            
            # 特徴量計算
            features = compute_features(df)
            
            # ターゲット計算
            tr1 = df["high"] - df["low"]
            tr2 = abs(df["high"] - df["close"].shift(1))
            tr3 = abs(df["low"] - df["close"].shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            
            target = compute_triple_barrier_target(
                df, atr,
                tp_atr=self.config["tp_atr_mult"],
                sl_atr=self.config["sl_atr_mult"],
                max_hold=36,  # バックテストと同期
            )
            
            # 共通インデックス
            common_idx = features.index.intersection(target.dropna().index)
            X = features.loc[common_idx].dropna()
            y = target.loc[X.index]
            
            # 学習
            model = train_model(X, y, use_triple_barrier=True)
            
            # 保存
            model.save_model(str(model_path / f"{symbol}_model.txt"))
            with open(model_path / f"{symbol}_features.txt", "w") as f:
                f.write("\n".join(X.columns))
            
            self.models[symbol] = model
            self.feature_columns[symbol] = list(X.columns)
            
            logger.info(f"モデル保存完了: {symbol}")
        
        return len(self.models) > 0
    
    def _predict_ml(self, symbol: str, features: pd.DataFrame, min_confidence: float) -> dict | None:
        """
        MLモデルを使用して予測を実行。
        
        Args:
            symbol: 通貨ペア
            features: 特徴量DataFrame
            min_confidence: 最低信頼度閾値
            
        Returns:
            予測結果dict、または条件未達でNone
        """
        if symbol not in self.models:
            return None
        
        model = self.models[symbol]
        
        # 特徴量を揃える
        if symbol in self.feature_columns:
            cols = self.feature_columns[symbol]
            missing = set(cols) - set(features.columns)
            if missing:
                logger.warning(f"特徴量不足: {missing}")
                return None
            features_ml = features[cols]
        else:
            features_ml = features
        
        # 欠損値チェック
        if features_ml.iloc[-1].isna().any():
            return None
        
        # 予測（Triple-barrier: 3クラス）
        pred = model.predict(features_ml.iloc[[-1]])
        
        # 確率から方向と信頼度を計算
        prob_sell = pred[0, 0]
        prob_neutral = pred[0, 1]
        prob_buy = pred[0, 2]
        
        direction_strength = max(prob_buy, prob_sell)
        edge_confidence = direction_strength - prob_neutral * 0.5
        
        # 信頼度閾値を適用
        if edge_confidence >= min_confidence:
            if prob_buy > prob_sell:
                direction = "buy"
            else:
                direction = "sell"
            
            return {
                "direction": direction,
                "confidence": edge_confidence,
                "prob_buy": prob_buy,
                "prob_sell": prob_sell,
                "prob_neutral": prob_neutral,
                "strategy": "ML_LightGBM",
            }
        
        return None
    
    def predict(self, symbol: str, df: pd.DataFrame, features: pd.DataFrame) -> dict | None:
        """
        PAIR_CONFIGに基づいてシンボル別に最適な戦略で予測を実行。
        
        各ペアに最も成績の良い単一戦略を使用し、
        必要に応じてMLフォールバックを試す。
        """
        # ペア別設定を取得
        pair_cfg = PAIR_CONFIG.get(symbol, {
            "strategy": "EMACross",
            "min_confidence": 0.50,
            "use_ml_fallback": True,
        })
        
        primary_strategy = pair_cfg.get("strategy", "EMACross")
        min_confidence = pair_cfg.get("min_confidence", 0.50)
        use_ml_fallback = pair_cfg.get("use_ml_fallback", True)
        
        # 戦略名から実行メソッドへのマッピング
        strategy_methods = {
            "EMACross": lambda: self.strategy_ema_cross(features),
            "TripleScreen": lambda: self.strategy_triple_screen(df, features),
            "RSI_Stoch": lambda: self.strategy_rsi_stoch(features),
            "Breakout": lambda: self.strategy_breakout(df, features),
            "Momentum": lambda: self.strategy_momentum(df, features),
            "MeanReversion": lambda: self.strategy_mean_reversion(df, features),
        }
        
        best_result = None
        
        # ML_Primary戦略の場合はMLを先に試す
        if primary_strategy == "ML_Primary" and symbol in self.models:
            best_result = self._predict_ml(symbol, features, min_confidence)
            if best_result:
                return best_result
        
        # ルールベース戦略を実行
        if primary_strategy in strategy_methods:
            result = strategy_methods[primary_strategy]()
            if result and result.get("confidence", 0) >= min_confidence:
                result["strategy"] = primary_strategy
                best_result = result
        
        # プライマリ戦略がシグナルを出さなかった場合、MLフォールバック
        if best_result is None and use_ml_fallback and primary_strategy != "ML_Primary":
            best_result = self._predict_ml(symbol, features, min_confidence)
        
        return best_result
    
    def apply_filters(
        self,
        symbol: str,
        direction: str,
        features: pd.DataFrame,
        current_time: datetime,
    ) -> str | None:
        """
        PAIR_CONFIGに基づいてペア別フィルターを適用。
        通過ならNone、拒否なら理由を返す。
        """
        if not self.config["use_filters"]:
            return None
        
        # ペア別設定を取得
        pair_cfg = PAIR_CONFIG.get(symbol, {})
        adx_threshold = pair_cfg.get("adx_threshold", self.config["adx_threshold"])
        sma_atr_threshold = pair_cfg.get("sma_atr_threshold", self.config["sma_atr_threshold"])
        long_only = pair_cfg.get("long_only", False)
        
        # ロングオンリーチェック（ペア別設定優先）
        if long_only and direction == "sell":
            return f"long_only: {symbol} はロングのみ"
        
        # 時間帯フィルター
        hour = current_time.hour
        if hour >= 21 or hour < 1:
            return f"time_filter: {hour}時は取引除外時間"
        
        # ADXフィルター（ペア別閾値）
        if "adx_14" in features.columns:
            adx = features.iloc[-1]["adx_14"]
            if pd.notna(adx) and adx < adx_threshold:
                return f"adx_filter: {adx:.1f} < {adx_threshold}"
        
        # SMA200距離フィルター（ペア別閾値）
        if "sma200_atr_distance" in features.columns:
            dist = features.iloc[-1]["sma200_atr_distance"]
            if pd.notna(dist) and dist < sma_atr_threshold:
                return f"sma_distance_filter: {dist:.2f} < {sma_atr_threshold}"
        
        return None
    
    def get_risk_multiplier(self, symbol: str, current_atr: float) -> float:
        """リスク調整倍率を計算。"""
        multiplier = 1.0
        
        # DD制限
        if self.peak_equity > 0:
            dd = (self.peak_equity - self.current_equity) / self.peak_equity
            if dd >= self.config["max_drawdown_pct"]:
                self.trading_halted = True
                logger.warning(f"取引停止: DD {dd*100:.1f}% >= {self.config['max_drawdown_pct']*100:.1f}%")
                return 0.0
            elif self.trading_halted and dd < self.config["max_drawdown_pct"] * 0.5:
                self.trading_halted = False
                logger.info("取引再開: DD回復")
        
        if self.trading_halted:
            return 0.0
        
        # 連敗制限
        if self.consecutive_losses.get(symbol, 0) >= self.config["consecutive_loss_limit"]:
            multiplier *= 0.5
            logger.info(f"リスク半減: {symbol} {self.consecutive_losses[symbol]}連敗")
        
        # ボラティリティ調整
        if symbol in self.avg_atr and self.avg_atr[symbol] > 0:
            vol_ratio = current_atr / self.avg_atr[symbol]
            if vol_ratio > self.config["vol_scale_threshold"]:
                vol_adj = max(0.5, 1.0 / vol_ratio)
                multiplier *= vol_adj
                logger.info(f"ボラ調整: {symbol} vol_ratio={vol_ratio:.2f}")
        
        return multiplier
    
    def calculate_position_size(
        self,
        symbol: str,
        sl_distance: float,
        risk_multiplier: float,
        confidence: float = 0.5,
    ) -> float:
        """
        ポジションサイズを計算。
        
        PAIR_CONFIGのレバレッジ設定を使用:
        - base_leverage: 基本レバレッジ
        - max_leverage: 高確信度時の最大レバレッジ
        - 確信度0.65以上でレバレッジが増加
        """
        # ペア別レバレッジ設定を取得
        pair_cfg = PAIR_CONFIG.get(symbol, {})
        base_leverage = pair_cfg.get("base_leverage", 25.0)
        max_leverage = pair_cfg.get("max_leverage", 80.0)
        leverage_threshold = 0.65  # この確信度以上でレバレッジ増加
        
        # 確信度に応じたレバレッジ計算
        if confidence >= leverage_threshold:
            # 確信度0.65-1.00をbase-maxにマッピング
            confidence_ratio = (confidence - leverage_threshold) / (1.0 - leverage_threshold)
            current_leverage = base_leverage + (max_leverage - base_leverage) * confidence_ratio
        else:
            current_leverage = base_leverage
        
        # リスク計算
        base_risk = self.config["risk_per_trade"] * current_leverage
        adjusted_risk = base_risk * risk_multiplier
        
        risk_amount = self.current_equity * adjusted_risk
        
        # ロットサイズ計算（簡易版）
        pip_val = self.config["pip_value"].get(symbol, 0.0001)
        sl_pips = sl_distance / pip_val
        
        if sl_pips <= 0:
            return 0.0
        
        # 1ロット = 100,000通貨、1pip = $10相当と仮定
        lot_value_per_pip = 10.0
        lots = risk_amount / (sl_pips * lot_value_per_pip)
        
        # ロットサイズ制限
        lots = max(0.01, min(lots, 1.0))  # 0.01〜1.0ロット
        
        logger.debug(f"  レバレッジ: {current_leverage:.1f}x (確信度: {confidence:.2f})")
        
        return round(lots, 2)
    
    def update_trade_result(self, symbol: str, pnl: float, is_win: bool):
        """取引結果を更新。"""
        self.current_equity += pnl
        
        if is_win:
            self.consecutive_losses[symbol] = 0
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
        else:
            self.consecutive_losses[symbol] = self.consecutive_losses.get(symbol, 0) + 1
        
        logger.info(f"取引結果: {symbol} PnL={pnl:.2f} {'勝' if is_win else '負'}")
        logger.info(f"  資産: {self.current_equity:.2f} (ピーク: {self.peak_equity:.2f})")
        logger.info(f"  連敗: {self.consecutive_losses[symbol]}")
    
    def run_once(self):
        """1回のシグナルチェック・取引実行。"""
        current_time = datetime.now(timezone.utc)
        logger.info(f"{'='*60}")
        logger.info(f"チェック時刻: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info(f"{'='*60}")
        
        for symbol in self.config["symbols"]:
            logger.info(f"--- {symbol} ---")
            
            # ニュースブラックアウトチェック
            if self.news_filter is not None:
                blackout = self.news_filter.check_blackout(symbol)
                if blackout.is_blocked:
                    events = ", ".join(e.title for e in blackout.blocking_events[:2])
                    logger.warning(f"ニュースブラックアウト中: {events}")
                    if blackout.blackout_ends:
                        logger.info(f"  解除予定: {blackout.blackout_ends.strftime('%H:%M')} UTC")
                    continue
            
            # 既存ポジションチェック
            positions = get_open_positions(symbol)
            if positions:
                logger.info(f"既存ポジションあり: {len(positions)}件")
                continue
            
            # スプレッドチェック
            spread = get_current_spread(symbol)
            max_spread = self.config["max_spread_pips"].get(symbol, 3.0)
            if spread > max_spread:
                logger.info(f"スプレッド過大: {spread:.1f} > {max_spread}")
                continue
            
            # データ取得
            df = get_ohlcv(symbol, self.config["timeframe"], bars=500)
            if df.empty or len(df) < 250:
                logger.warning(f"データ不足: {len(df)}件")
                continue
            
            # 特徴量計算
            features = compute_features(df)
            features = features.dropna()
            
            if len(features) < 50:
                logger.warning(f"特徴量不足")
                continue
            
            # ATR更新
            current_atr = features.iloc[-1]["atr_14"]
            if symbol not in self.avg_atr:
                self.avg_atr[symbol] = features["atr_14"].tail(20).mean()
            else:
                self.avg_atr[symbol] = 0.95 * self.avg_atr[symbol] + 0.05 * current_atr
            
            # 予測（シンボル別の最適戦略を使用）
            pred = self.predict(symbol, df, features)
            if pred is None:
                logger.info("シグナルなし")
                continue
            
            strategy_name = pred.get("strategy", "Unknown")
            logger.info(f"予測: {pred['direction']} 信頼度={pred['confidence']:.3f} ({strategy_name})")
            if "prob_buy" in pred:
                logger.debug(f"  P(buy)={pred['prob_buy']:.3f} P(sell)={pred['prob_sell']:.3f} P(neutral)={pred['prob_neutral']:.3f}")
            
            # 信頼度チェック（PAIR_CONFIGを優先）
            pair_min_conf = PAIR_CONFIG.get(symbol, {}).get("min_confidence", self.config["min_confidence"])
            if pred["confidence"] < pair_min_conf:
                logger.info(f"信頼度不足: {pred['confidence']:.3f} < {pair_min_conf}")
                continue
            
            # フィルター適用
            filter_reason = self.apply_filters(symbol, pred["direction"], features, current_time)
            if filter_reason:
                logger.info(f"フィルター: {filter_reason}")
                continue
            
            # リスク調整
            risk_mult = self.get_risk_multiplier(symbol, current_atr)
            if risk_mult == 0:
                logger.info("取引停止中")
                continue
            
            # SL/TP計算
            current_price = df.iloc[-1]["close"]
            # SL/TP計算（PAIR_CONFIGを優先、なければグローバル設定）
            pair_cfg = PAIR_CONFIG.get(symbol, {})
            sl_mult = pair_cfg.get("sl_atr_mult", self.config["sl_atr_mult"])
            tp_mult = pair_cfg.get("tp_atr_mult", self.config["tp_atr_mult"])
            sl_distance = current_atr * sl_mult
            tp_distance = current_atr * tp_mult
            
            if pred["direction"] == "buy":
                sl = current_price - sl_distance
                tp = current_price + tp_distance
            else:
                sl = current_price + sl_distance
                tp = current_price - tp_distance
            
            # ポジションサイズ（確信度に応じたレバレッジ適用）
            lots = self.calculate_position_size(symbol, sl_distance, risk_mult, pred["confidence"])
            
            # 適用されたレバレッジを計算して表示
            base_lev = pair_cfg.get("base_leverage", 25.0)
            max_lev = pair_cfg.get("max_leverage", 80.0)
            if pred["confidence"] >= 0.65:
                conf_ratio = (pred["confidence"] - 0.65) / 0.35
                current_lev = base_lev + (max_lev - base_lev) * conf_ratio
            else:
                current_lev = base_lev
            
            logger.info(f"★ シグナル発生! {symbol} {pred['direction']}")
            logger.info(f"  価格: {current_price:.5f}")
            logger.info(f"  SL: {sl:.5f} ({sl_distance/self.config['pip_value'][symbol]:.1f}pips)")
            logger.info(f"  TP: {tp:.5f} ({tp_distance/self.config['pip_value'][symbol]:.1f}pips)")
            logger.info(f"  ロット: {lots}")
            logger.info(f"  レバレッジ: {current_lev:.1f}x (確信度={pred['confidence']:.2f})")
            logger.info(f"  リスク調整: {risk_mult:.2f}")
            
            # 注文実行
            result = place_order(symbol, pred["direction"], lots, sl, tp)
            if result:
                logger.info(f"注文発注完了")
    
    def run(self):
        """メインループ。"""
        logger.info("="*60)
        logger.info(" デモ取引開始")
        logger.info("="*60)
        logger.info(f"対象: {self.config['symbols']}")
        logger.info(f"タイムフレーム: {self.config['timeframe']}")
        logger.info(f"初期資金: {self.config['initial_capital']}")
        # ペア別レバレッジ設定を表示
        for sym in self.config['symbols']:
            pair_cfg = PAIR_CONFIG.get(sym, {})
            base_lev = pair_cfg.get("base_leverage", 25.0)
            max_lev = pair_cfg.get("max_leverage", 80.0)
            logger.info(f"  {sym}: レバレッジ {base_lev:.0f}-{max_lev:.0f}x")
        logger.info(f"DRY RUN: {self.config['dry_run']}")
        logger.info("="*60)
        
        # モデル読み込み
        if not self.load_models():
            logger.info("モデルを学習します...")
            if not self.train_and_save_models():
                logger.error("モデル学習失敗")
                return
        
        logger.info(f"{self.config['check_interval_seconds']}秒ごとにチェック中... (Ctrl+C で停止)")
        
        try:
            while True:
                self.run_once()
                time.sleep(self.config["check_interval_seconds"])
        except KeyboardInterrupt:
            logger.info("停止しました。")
        finally:
            # ニュースフィルター停止
            if self.news_filter is not None:
                self.news_filter.stop_auto_refresh()
                logger.info("ニュースフィルター停止")


# =============================================================================
# メイン
# =============================================================================

def main():
    # MT5初期化
    if not initialize_mt5():
        return 1
    
    try:
        # トレーダー作成・実行
        trader = DemoTrader(CONFIG)
        trader.run()
    finally:
        mt5.shutdown()
        logger.info("MT5切断完了")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
