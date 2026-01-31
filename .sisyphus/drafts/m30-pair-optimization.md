# Draft: M30タイムフレーム切り替え + 通貨ペア別モデル/戦略最適化

## Requirements (confirmed)

### ユーザーの目標
- M30タイムフレームに変更して精度を上げる
- 通貨ペア別にモデルや戦略を変える

### 現状のバックテスト結果

**H1, MLモデル (ユーザー提供):**
| ペア | PF | Return | 状態 |
|------|-----|--------|------|
| USDJPY | 1.10 | +14.59% | OK |
| EURJPY | 0.95 | -16.25% | 要改善 |
| AUDUSD | 0.88 | -54.52% | 要改善 |
| GBPUSD | 0.99 | -8.85% | 要改善 |
| EURUSD | 0.97 | -14.17% | 要改善 |

**ルールベース戦略（以前の結果）:**
| ペア | 戦略 | PF |
|------|------|-----|
| USDJPY | EMACross | 1.08 |
| EURJPY | Breakout | 1.10 |
| AUDUSD | TripleScreen | 1.11 |
| GBPUSD | RSI_Stoch | 1.02 |

**既存MLバックテスト結果 (data/backtest/ml_strategy_results.csv):**
| ペア | PF | Return | MDD | 最適閾値 |
|------|-----|--------|-----|---------|
| EURJPY | 1.05 | +22.3% | -28.0% | 0.45 |
| USDJPY (Long-only) | 1.02 | +4.8% | -17.8% | 0.35 |
| CADJPY | 0.95 | -12.2% | -38.6% | 0.50 |
| AUDJPY | 0.94 | -25.7% | -51.4% | 0.30 |
| AUDUSD | 0.93 | -39.7% | -49.5% | 0.35 |
| GBPUSD | 0.92 | -36.7% | -42.2% | 0.35 |

## Technical Decisions

### 確認済み
- M30データ: 12ペア分が `data/raw/{SYMBOL}_M30.parquet` に存在
  - EURUSD, USDJPY, EURJPY, GBPUSD, AUDUSD
  - GBPJPY, AUDJPY, CADJPY, EURAUD, EURGBP, USDCAD, USDCHF
- データ形式: Parquet (snappy圧縮)
- 既存モデル: `models/{SYMBOL}_model.txt` (USDJPY, EURJPY, AUDUSD, EURUSD の4ペア分、H1用)
- LightGBM: 3クラス分類 (up/down/neutral)、46特徴量

### 主要ファイル構造

1. **scripts/run_trading.py** (ライブ取引)
   - CONFIG dict (L80-134): timeframe, symbols, risk設定
   - SYMBOL_STRATEGIES (L72-78): ペア別戦略マッピング
   - DemoTrader.predict() (L806-884): 複数戦略を評価、最高信頼度を選択
   - 戦略メソッド6種: EMACross, TripleScreen, RSI_Stoch, Breakout, Momentum, MeanReversion
   - get_ohlcv(): tf_mapにM30追加必要 (L347-354)

2. **scripts/run_backtest.py** (バックテスト)
   - timeframe変数 (L859): 現在 "H1"
   - symbols変数 (L858): ["USDJPY", "EURJPY", "AUDUSD", "GBPUSD", "EURUSD"]
   - フィルター設定 (L883-885): ADX閾値=15.0, SMA乖離閾値=0.3
   - walk_forward_backtest(): 拡張ウィンドウ、6か月ごと再学習

3. **config/trading.yaml**
   - timeframes.primary: "H1" (L51)
   - signals.min_confidence: 0.35

4. **config/risk.yaml**
   - trade_risk.risk_per_trade_pct: 1.0
   - stop_loss.atr.multiplier: 2.0
   - take_profit.risk_reward_ratio: 1.5

5. **python/src/data/loader.py**
   - TIMEFRAME_MAP: M30=30 (対応済み)
   - FileDataLoader: Parquetからの読み込みをサポート

## Research Findings

### 戦略選択ロジック詳細 (run_trading.py L806-884)

1. SYMBOL_STRATEGIESで各ペアに複数戦略を割り当て
2. 全戦略を評価して最高信頼度のシグナルを採用
3. ルールベースがシグナルを出さない場合のみMLモデルをフォールバック使用
4. 現行マッピング:
   - USDJPY: [EMACross, Momentum]
   - EURJPY: [Breakout, MeanReversion]
   - AUDUSD: [TripleScreen, Momentum]
   - GBPUSD: [RSI_Stoch, MeanReversion]
   - EURUSD: [EMACross, Breakout]

### 各戦略の特性

| 戦略 | 特性 | 適した相場 | 信頼度範囲 |
|------|------|-----------|-----------|
| EMACross | MACDヒストグラムのクロス | トレンド相場 | 0.60-0.65 |
| TripleScreen | SMA200+MACD+RSI複合 | 安定トレンド | 0.50-0.70 |
| RSI_Stoch | RSI+ストキャスオシレーター | レンジ相場 (ADX<40) | 0.45-0.75 |
| Breakout | BB幅縮小後のブレイク | 収縮後のブレイク | 0.55-0.75 |
| Momentum | ADX+SMA+効率比率 | 強トレンド (ADX>25) | 0.40-0.70 |
| MeanReversion | Zスコア+BB+RSI | レンジ相場 (ADX<30) | 0.35-0.65 |

### タイムフレーム変更の影響箇所

| ファイル | 変更箇所 | 内容 |
|---------|---------|------|
| run_trading.py | L83 | CONFIG["timeframe"] = "M30" |
| run_trading.py | L347-354 | tf_mapにM30追加 |
| run_backtest.py | L859 | timeframe = "M30" |
| config/trading.yaml | L51 | timeframes.primary: "M30" |

## Decisions Made (2026-01-31)

### 1. 戦略選択方針: 単一戦略固定
- 各ペアで最も成績が良い単一戦略を使用
- シンプルに最適な戦略1つを固定

### 2. パラメータ最適化範囲: 戦略+閾値のみ
- 戦略タイプ: ペア別に固定
- 信頼度閾値: ペア別に最適化（バックテストで決定）
- フィルター/SL/TP: まずは共通設定維持

### 3. M30への期待
- シグナル頻度の増加 + エントリータイミングの精度向上
- パラメータ調整: まず変更なしで検証 → 必要なら調整

### 4. 対象ペア: 現行5ペア
- USDJPY, EURJPY, AUDUSD, GBPUSD, EURUSD
- CADJPYは後から追加可能

## Scope Boundaries

### INCLUDE
- M30タイムフレーム対応（コード変更）
- 通貨ペア別設定構造（PAIR_CONFIG）の設計・実装
- バックテスト実行による検証（PF > 1.0目標）

### EXCLUDE
- フィルター/SL/TPのペア別最適化（将来対応）
- 対象ペア拡張（CADJPYなど）（将来対応）
- pytest等のテスト基盤（バックテストで検証）
