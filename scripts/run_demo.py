"""
デモ取引スクリプト

バックテストで確立した戦略をMT5デモ口座で実行する。

主な機能:
- MT5からリアルタイムデータ取得
- LightGBMモデルによるシグナル生成
- USDJPYロングオンリー、フィルター適用
- リスク管理（DD制限、連敗制限、ボラティリティ調整）
- 自動売買（エントリー・決済）

使用方法:
    python scripts/run_demo.py
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

CONFIG = {
    # 取引対象
    "symbols": ["EURUSD", "USDJPY"],
    "timeframe": "H1",
    
    # モデル設定
    "model_path": project_root / "models",  # モデル保存先
    
    # 取引設定
    "initial_capital": 10000.0,  # 初期資金（デモ口座）
    "risk_per_trade": 0.01,  # 1トレードあたりリスク（1%）
    "leverage": 3,  # レバレッジ倍率
    "min_confidence": 0.50,  # 最小信頼度
    
    # SL/TP設定
    "sl_atr_mult": 2.0,  # SL距離のATR倍率
    "tp_atr_mult": 3.0,  # TP距離のATR倍率
    
    # フィルター設定
    "use_filters": True,
    "adx_threshold": 20.0,
    "sma_atr_threshold": 0.5,
    
    # リスク管理
    "max_drawdown_pct": 0.25,  # 最大DD制限
    "consecutive_loss_limit": 5,  # 連敗制限
    "vol_scale_threshold": 1.5,  # ボラティリティ閾値
    
    # シンボル別制限
    "long_only_symbols": {"USDJPY"},  # ロングオンリー
    
    # スプレッド・スリッページ
    "max_spread_pips": {
        "EURUSD": 2.0,
        "USDJPY": 3.0,
    },
    
    # pip値
    "pip_value": {
        "EURUSD": 0.0001,
        "USDJPY": 0.01,
    },
    
    # 実行間隔
    "check_interval_seconds": 60,  # シグナルチェック間隔
    
    # デバッグ
    "dry_run": False,  # False=実際の注文を出す（デモ口座）
}


# =============================================================================
# 特徴量計算（run_backtest.pyと同じロジック）
# =============================================================================

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    バックテストと同じ特徴量を計算。
    """
    features = pd.DataFrame(index=df.index)
    
    # 基本価格特徴量
    features["returns"] = df["close"].pct_change()
    features["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    
    # ボラティリティ
    features["volatility_20"] = features["returns"].rolling(20).std()
    
    # ATR
    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift(1))
    tr3 = abs(df["low"] - df["close"].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    features["atr_14"] = tr.rolling(14).mean()
    
    # 移動平均
    for period in [5, 10, 20, 50, 100, 200]:
        features[f"sma_{period}"] = df["close"].rolling(period).mean()
        features[f"ema_{period}"] = df["close"].ewm(span=period).mean()
    
    # 価格とSMAの関係
    features["price_to_sma20"] = df["close"] / features["sma_20"] - 1
    features["price_to_sma50"] = df["close"] / features["sma_50"] - 1
    features["price_to_sma200"] = df["close"] / features["sma_200"] - 1
    
    # SMA200からの乖離（ATR単位）
    features["sma200_atr_distance"] = abs(df["close"] - features["sma_200"]) / features["atr_14"]
    
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features["rsi_14"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    features["macd"] = ema12 - ema26
    features["macd_signal"] = features["macd"].ewm(span=9).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]
    
    # ボリンジャーバンド
    sma20 = features["sma_20"]
    std20 = df["close"].rolling(20).std()
    features["bb_upper"] = sma20 + 2 * std20
    features["bb_lower"] = sma20 - 2 * std20
    features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / sma20
    features["bb_position"] = (df["close"] - features["bb_lower"]) / (features["bb_upper"] - features["bb_lower"])
    
    # ADX
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr14 = features["atr_14"]
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    features["adx_14"] = dx.rolling(14).mean()
    features["plus_di"] = plus_di
    features["minus_di"] = minus_di
    
    # セッション特徴量
    if hasattr(df.index, 'hour'):
        hour = df.index.hour
    else:
        hour = pd.to_datetime(df.index).hour
    
    features["hour"] = hour
    features["is_tokyo"] = ((hour >= 0) & (hour < 9)).astype(int)
    features["is_london"] = ((hour >= 7) & (hour < 16)).astype(int)
    features["is_ny"] = ((hour >= 13) & (hour < 22)).astype(int)
    
    # セッション累積リターン
    tokyo_ret = features["returns"].where(features["is_tokyo"] == 1, 0)
    london_ret = features["returns"].where(features["is_london"] == 1, 0)
    ny_ret = features["returns"].where(features["is_ny"] == 1, 0)
    
    features["tokyo_cumret"] = tokyo_ret.rolling(24).sum()
    features["london_cumret"] = london_ret.rolling(24).sum()
    features["ny_cumret"] = ny_ret.rolling(24).sum()
    
    # トレンド品質
    features["efficiency_ratio"] = abs(df["close"].diff(10)) / tr.rolling(10).sum()
    features["sma_slope"] = features["sma_20"].diff(5) / features["atr_14"]
    
    # Parkinson volatility
    features["parkinson_vol"] = np.sqrt(np.log(df["high"] / df["low"]) ** 2 / (4 * np.log(2))).rolling(20).mean()
    
    # Tail asymmetry
    ret_5 = features["returns"].rolling(5).apply(lambda x: x.quantile(0.95) - x.quantile(0.05), raw=False)
    ret_50 = features["returns"].rolling(5).apply(lambda x: x.quantile(0.5), raw=False)
    features["tail_asymmetry"] = ret_5 - 2 * ret_50
    
    # ミーンリバージョン指標
    features["zscore"] = (df["close"] - features["sma_20"]) / std20
    features["bw_percentile"] = features["bb_position"]
    
    # 欠損値処理
    features = features.replace([np.inf, -np.inf], np.nan)
    
    return features


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
                max_hold=48,
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
    
    def predict(self, symbol: str, features: pd.DataFrame) -> dict | None:
        """予測を実行。"""
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
            features = features[cols]
        
        # 欠損値チェック
        if features.iloc[-1].isna().any():
            return None
        
        # 予測（Triple-barrier: 3クラス）
        pred = model.predict(features.iloc[[-1]])
        
        # 確率から方向と信頼度を計算
        prob_sell = pred[0, 0]
        prob_neutral = pred[0, 1]
        prob_buy = pred[0, 2]
        
        direction_strength = max(prob_buy, prob_sell)
        edge_confidence = direction_strength - prob_neutral * 0.5
        
        if prob_buy > prob_sell:
            direction = "buy"
            confidence = edge_confidence
        else:
            direction = "sell"
            confidence = edge_confidence
        
        return {
            "direction": direction,
            "confidence": confidence,
            "prob_buy": prob_buy,
            "prob_sell": prob_sell,
            "prob_neutral": prob_neutral,
        }
    
    def apply_filters(
        self,
        symbol: str,
        direction: str,
        features: pd.DataFrame,
        current_time: datetime,
    ) -> str | None:
        """フィルターを適用。通過ならNone、拒否なら理由を返す。"""
        
        if not self.config["use_filters"]:
            return None
        
        # ロングオンリーチェック
        if symbol in self.config["long_only_symbols"] and direction == "sell":
            return f"long_only: {symbol} はロングのみ"
        
        # 時間帯フィルター
        hour = current_time.hour
        if hour >= 21 or hour < 1:
            return f"time_filter: {hour}時は取引除外時間"
        
        # ADXフィルター
        if "adx_14" in features.columns:
            adx = features.iloc[-1]["adx_14"]
            if pd.notna(adx) and adx < self.config["adx_threshold"]:
                return f"adx_filter: {adx:.1f} < {self.config['adx_threshold']}"
        
        # SMA200距離フィルター
        if "sma200_atr_distance" in features.columns:
            dist = features.iloc[-1]["sma200_atr_distance"]
            if pd.notna(dist) and dist < self.config["sma_atr_threshold"]:
                return f"sma_distance_filter: {dist:.2f} < {self.config['sma_atr_threshold']}"
        
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
    ) -> float:
        """ポジションサイズを計算。"""
        base_risk = self.config["risk_per_trade"] * self.config["leverage"]
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
            
            # 予測
            pred = self.predict(symbol, features)
            if pred is None:
                logger.warning("予測失敗")
                continue
            
            logger.info(f"予測: {pred['direction']} 信頼度={pred['confidence']:.3f}")
            logger.debug(f"  P(buy)={pred['prob_buy']:.3f} P(sell)={pred['prob_sell']:.3f} P(neutral)={pred['prob_neutral']:.3f}")
            
            # 信頼度チェック
            if pred["confidence"] < self.config["min_confidence"]:
                logger.info(f"信頼度不足: {pred['confidence']:.3f} < {self.config['min_confidence']}")
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
            sl_distance = current_atr * self.config["sl_atr_mult"]
            tp_distance = current_atr * self.config["tp_atr_mult"]
            
            if pred["direction"] == "buy":
                sl = current_price - sl_distance
                tp = current_price + tp_distance
            else:
                sl = current_price + sl_distance
                tp = current_price - tp_distance
            
            # ポジションサイズ
            lots = self.calculate_position_size(symbol, sl_distance, risk_mult)
            
            logger.info(f"★ シグナル発生! {symbol} {pred['direction']}")
            logger.info(f"  価格: {current_price:.5f}")
            logger.info(f"  SL: {sl:.5f} ({sl_distance/self.config['pip_value'][symbol]:.1f}pips)")
            logger.info(f"  TP: {tp:.5f} ({tp_distance/self.config['pip_value'][symbol]:.1f}pips)")
            logger.info(f"  ロット: {lots}")
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
        logger.info(f"レバレッジ: {self.config['leverage']}倍")
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
