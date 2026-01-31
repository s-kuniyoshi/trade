# M30タイムフレーム切り替え + 通貨ペア別戦略最適化

## TL;DR

> **Quick Summary**: M30タイムフレームに切り替え、各通貨ペアに最適な単一戦略を割り当て、バックテストでPF > 1.0を検証する
> 
> **Deliverables**:
> - M30対応のコード変更（run_trading.py, run_backtest.py, config/trading.yaml）
> - PAIR_CONFIG dict（ペア別戦略・閾値設定）
> - M30バックテスト結果（5ペア x PF > 1.0目標）
> 
> **Estimated Effort**: Medium (3-4時間)
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 → Task 2 → Task 4 → Task 5 → Task 6

---

## Context

### Original Request
- M30タイムフレームに変更して精度を上げる
- 通貨ペア別にモデルや戦略を変える

### Interview Summary
**Key Discussions**:
- 戦略選択: 各ペアで最も成績が良い単一戦略を使用
- 最適化範囲: 戦略タイプ + 信頼度閾値のみ（フィルター/SL/TPは共通維持）
- M30への期待: シグナル頻度増加 + エントリータイミング精度向上
- 対象ペア: 現行5ペア（USDJPY, EURJPY, AUDUSD, GBPUSD, EURUSD）

**Research Findings**:
- M30データ: 12ペア分が`data/raw/{SYMBOL}_M30.parquet`に存在（確認済み）
- 既存バックテスト結果（H1）:
  - AUDUSD × TripleScreen: PF 1.11
  - EURJPY × Breakout: PF 1.10
  - USDJPY × EMACross: PF 1.08
  - GBPUSD × RSI_Stoch: PF 1.02
  - EURUSD: 要検証

### Metis Review
**Identified Gaps** (addressed):
- tf_mapにM30追加必要 → Task 1で対応
- 複数戦略から単一戦略への変更 → Task 2で対応
- EURUSD戦略未定 → Task 4で全ペアバックテスト時に決定
- 受け入れ基準未定義 → 下記で定義

---

## Work Objectives

### Core Objective
M30タイムフレームで各通貨ペアに最適な単一戦略を割り当て、バックテストでPF > 1.0を達成する

### Concrete Deliverables
1. `scripts/run_trading.py`: M30対応 + PAIR_CONFIG導入
2. `scripts/run_backtest.py`: M30対応
3. `config/trading.yaml`: timeframes.primary = "M30"
4. バックテスト結果: 5ペア全てPF > 1.0

### Definition of Done
- [ ] `python scripts/run_backtest.py`が正常完了
- [ ] 全5ペアでPF >= 1.0
- [ ] 全5ペアで取引数 >= 100（統計的有意性）

### Must Have
- M30タイムフレーム対応
- ペア別単一戦略の割り当て
- ペア別信頼度閾値の最適化
- バックテスト検証

### Must NOT Have (Guardrails)
- 複数戦略のアンサンブル（単一戦略のみ）
- フィルター/SL/TPのペア別最適化（Phase 2へ延期）
- 新規通貨ペアの追加（5ペア固定）
- H4/D1のマルチタイムフレーム分析（Phase 2へ延期）

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (run_backtest.py)
- **User wants tests**: Manual verification via backtest
- **Framework**: Existing backtest infrastructure

### Automated Verification

Each TODO includes EXECUTABLE verification:
- **バックテスト**: `python scripts/run_backtest.py` 実行結果で検証
- **コード変更**: Python構文チェック + バックテスト実行で検証

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: M30タイムフレーム対応（コード変更）
└── Task 3: config/trading.yaml更新

Wave 2 (After Wave 1):
├── Task 2: PAIR_CONFIG構造導入
└── (Task 1完了後)

Wave 3 (After Wave 2):
└── Task 4: M30バックテスト実行

Wave 4 (After Wave 3):
├── Task 5: 閾値最適化
└── Task 6: 最終検証・ライブ取引設定更新

Critical Path: Task 1 → Task 2 → Task 4 → Task 5 → Task 6
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 4 | 3 |
| 2 | 1 | 4 | None |
| 3 | None | 6 | 1 |
| 4 | 1, 2 | 5 | None |
| 5 | 4 | 6 | None |
| 6 | 3, 5 | None | None (final) |

---

## TODOs

- [ ] 1. M30タイムフレーム対応（基盤変更）

  **What to do**:
  - `scripts/run_trading.py`: tf_mapにM30追加（L347-354）
    ```python
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,  # 追加
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    ```
  - `scripts/run_backtest.py`: timeframe変数を"M30"に変更（L859）
    ```python
    timeframe = "M30"  # "H1"から変更
    ```

  **Must NOT do**:
  - 他のタイムフレーム設定の変更
  - ローリング期間の変更（まずそのまま検証）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 2箇所の単純な変更のみ
  - **Skills**: [`git-master`]
    - `git-master`: コミット時に使用

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 3)
  - **Blocks**: Tasks 2, 4
  - **Blocked By**: None

  **References**:
  - `scripts/run_trading.py:347-356` - tf_map定義、M30追加箇所
  - `scripts/run_backtest.py:859` - timeframe変数定義

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "import scripts.run_backtest as rb; print(rb.timeframe)"
  # Assert: Output is "M30"
  
  # Agent runs:
  grep -n "M30" scripts/run_trading.py
  # Assert: M30がtf_mapに存在
  ```

  **Commit**: YES
  - Message: `feat(trading): add M30 timeframe support`
  - Files: `scripts/run_trading.py`, `scripts/run_backtest.py`

---

- [ ] 2. PAIR_CONFIG構造導入（戦略割り当て）

  **What to do**:
  - `scripts/run_trading.py`: PAIR_CONFIG dict作成（L70-80付近）
    ```python
    # 通貨ペア別設定（バックテスト結果に基づく）
    PAIR_CONFIG = {
        "USDJPY": {
            "strategy": "EMACross",      # PF 1.08
            "min_confidence": 0.40,      # 要最適化
        },
        "EURJPY": {
            "strategy": "Breakout",      # PF 1.10
            "min_confidence": 0.38,      # 要最適化
        },
        "AUDUSD": {
            "strategy": "TripleScreen",  # PF 1.11
            "min_confidence": 0.40,      # 要最適化
        },
        "GBPUSD": {
            "strategy": "RSI_Stoch",     # PF 1.02
            "min_confidence": 0.35,      # 要最適化
        },
        "EURUSD": {
            "strategy": "EMACross",      # 仮（バックテストで決定）
            "min_confidence": 0.35,      # 要最適化
        },
    }
    ```
  - `SYMBOL_STRATEGIES`を削除または単一戦略に変更
  - `DemoTrader.predict()`を修正: PAIR_CONFIGから単一戦略を取得
    ```python
    def predict(self, symbol: str, df: pd.DataFrame, features: pd.DataFrame) -> dict | None:
        config = PAIR_CONFIG.get(symbol, {"strategy": "EMACross", "min_confidence": 0.35})
        strategy_name = config["strategy"]
        
        strategy_methods = {
            "EMACross": lambda: self.strategy_ema_cross(features),
            "TripleScreen": lambda: self.strategy_triple_screen(df, features),
            "RSI_Stoch": lambda: self.strategy_rsi_stoch(features),
            "Breakout": lambda: self.strategy_breakout(df, features),
            "Momentum": lambda: self.strategy_momentum(df, features),
            "MeanReversion": lambda: self.strategy_mean_reversion(df, features),
        }
        
        if strategy_name in strategy_methods:
            result = strategy_methods[strategy_name]()
            if result:
                result["strategy"] = strategy_name
                return result
        
        return None
    ```
  - `run_once()`の信頼度チェックをPAIR_CONFIG参照に変更

  **Must NOT do**:
  - MLモデルのフォールバック維持（単一戦略のみ使用）
  - 複数戦略の評価ロジック維持

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: predict()メソッドのリファクタリングが中心
  - **Skills**: []
    - 特別なスキル不要

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (after Task 1)
  - **Blocks**: Task 4
  - **Blocked By**: Task 1

  **References**:
  - `scripts/run_trading.py:72-78` - SYMBOL_STRATEGIES（削除/変更対象）
  - `scripts/run_trading.py:806-884` - predict()メソッド（リファクタリング対象）
  - `scripts/run_trading.py:1063-1064` - 信頼度チェック（PAIR_CONFIG参照に変更）
  - `data/backtest/FINAL_REPORT.md:14-20` - 最適戦略の根拠

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "from scripts.run_trading import PAIR_CONFIG; print(PAIR_CONFIG['USDJPY']['strategy'])"
  # Assert: Output is "EMACross"
  
  # Agent runs:
  grep -c "SYMBOL_STRATEGIES" scripts/run_trading.py
  # Assert: 0 (削除済み) または参照なし
  ```

  **Commit**: YES
  - Message: `refactor(trading): introduce PAIR_CONFIG for per-pair strategy assignment`
  - Files: `scripts/run_trading.py`

---

- [ ] 3. config/trading.yaml更新

  **What to do**:
  - `config/trading.yaml`: timeframes.primary を"M30"に変更（L51）
    ```yaml
    timeframes:
      primary: "M30"  # "H1"から変更
    ```

  **Must NOT do**:
  - higher/lowerタイムフレームの変更

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 1行の単純な変更
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 6
  - **Blocked By**: None

  **References**:
  - `config/trading.yaml:49-59` - timeframes設定

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  grep "primary:" config/trading.yaml
  # Assert: Contains "M30"
  ```

  **Commit**: YES (groups with Task 1)
  - Message: `feat(trading): add M30 timeframe support`
  - Files: `config/trading.yaml`

---

- [ ] 4. M30バックテスト実行

  **What to do**:
  - `python scripts/run_backtest.py`を実行
  - 各ペアの結果を確認:
    - PF (Profit Factor) >= 1.0
    - 取引数 >= 100
    - MDD (Max Drawdown) <= 30%
  - EURUSDの最適戦略を決定（EMACross, Breakout, TripleScreenを比較）
  - 結果をPAIR_CONFIGに反映

  **Must NOT do**:
  - フィルター/SL/TPパラメータの変更（共通設定維持）
  - H1との比較分析（Phase 2）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: バックテスト実行と結果分析
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential)
  - **Blocks**: Task 5
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `scripts/run_backtest.py:845-1132` - main()関数、バックテスト実行ロジック
  - `data/backtest/FINAL_REPORT.md` - 過去のバックテスト結果（参考）

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python scripts/run_backtest.py
  # Assert: 正常終了（exit code 0）
  # Assert: 各ペアのPF >= 1.0がターミナル出力に表示
  
  # 期待される出力形式:
  # Symbol     Return%    Sharpe    MaxDD%    Trades    WinRate%    PF
  # USDJPY      XX.XX      X.XX     -XX.XX       XXX       XX.XX    1.XX
  # EURJPY      XX.XX      X.XX     -XX.XX       XXX       XX.XX    1.XX
  # AUDUSD      XX.XX      X.XX     -XX.XX       XXX       XX.XX    1.XX
  # GBPUSD      XX.XX      X.XX     -XX.XX       XXX       XX.XX    1.XX
  # EURUSD      XX.XX      X.XX     -XX.XX       XXX       XX.XX    1.XX
  ```

  **Commit**: NO (バックテスト実行のみ)

---

- [ ] 5. 閾値最適化

  **What to do**:
  - Task 4のバックテスト結果から各ペアの最適閾値を確認
  - `scripts/run_backtest.py`の閾値探索結果を参照:
    ```
    閾値探索中...
    閾値 0.20: PF=X.XX, 取引数=XXX, リターン=XX.X%
    閾値 0.25: PF=X.XX, 取引数=XXX, リターン=XX.X%
    ...
    最適閾値: 0.XX (PF=X.XX)
    ```
  - PAIR_CONFIGのmin_confidenceを最適値に更新
  - EURUSDの戦略も確定（最も高いPFの戦略を選択）

  **Must NOT do**:
  - 閾値 < 0.20 または > 0.60 の採用
  - 取引数 < 100 の閾値の採用

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: バックテスト結果からPAIR_CONFIGを更新するのみ
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (after Task 4)
  - **Blocks**: Task 6
  - **Blocked By**: Task 4

  **References**:
  - `scripts/run_backtest.py:1004-1062` - 閾値探索ロジック
  - `scripts/run_trading.py:PAIR_CONFIG` - 更新対象

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "from scripts.run_trading import PAIR_CONFIG; print({k: v['min_confidence'] for k, v in PAIR_CONFIG.items()})"
  # Assert: 各ペアに0.20-0.60の範囲の閾値が設定されている
  ```

  **Commit**: YES
  - Message: `feat(trading): optimize confidence thresholds per pair based on M30 backtest`
  - Files: `scripts/run_trading.py`

---

- [ ] 6. 最終検証・ライブ取引設定更新

  **What to do**:
  - `scripts/run_trading.py`: CONFIG["timeframe"]を"M30"に変更（L83）
  - 最終バックテスト実行で全ペアPF >= 1.0を確認
  - ライブ取引準備完了の確認

  **Must NOT do**:
  - dry_run設定の変更（現状維持）
  - 本番デプロイ（別タスク）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 設定変更と最終確認のみ
  - **Skills**: [`git-master`]
    - `git-master`: 最終コミット

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (final)
  - **Blocks**: None (final task)
  - **Blocked By**: Tasks 3, 5

  **References**:
  - `scripts/run_trading.py:83` - CONFIG["timeframe"]
  - `scripts/run_trading.py:133` - dry_run設定

  **Acceptance Criteria**:
  ```bash
  # Agent runs:
  python -c "from scripts.run_trading import CONFIG; print(CONFIG['timeframe'])"
  # Assert: Output is "M30"
  
  # Agent runs:
  python scripts/run_backtest.py
  # Assert: 全ペアPF >= 1.0
  ```

  **Commit**: YES
  - Message: `feat(trading): complete M30 migration with optimized per-pair strategies`
  - Files: `scripts/run_trading.py`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1, 3 | `feat(trading): add M30 timeframe support` | run_trading.py, run_backtest.py, trading.yaml | grep M30 |
| 2 | `refactor(trading): introduce PAIR_CONFIG for per-pair strategy assignment` | run_trading.py | Python import check |
| 5 | `feat(trading): optimize confidence thresholds per pair based on M30 backtest` | run_trading.py | Python import check |
| 6 | `feat(trading): complete M30 migration with optimized per-pair strategies` | run_trading.py | backtest run |

---

## Success Criteria

### Verification Commands
```bash
# M30対応確認
grep -n "M30" scripts/run_trading.py scripts/run_backtest.py config/trading.yaml

# PAIR_CONFIG確認
python -c "from scripts.run_trading import PAIR_CONFIG; print(PAIR_CONFIG)"

# バックテスト実行
python scripts/run_backtest.py
# Expected: 全ペアでPF >= 1.0
```

### Final Checklist
- [ ] All "Must Have" present:
  - [ ] M30タイムフレーム対応
  - [ ] ペア別単一戦略の割り当て
  - [ ] ペア別信頼度閾値の最適化
  - [ ] バックテスト検証（PF >= 1.0）
- [ ] All "Must NOT Have" absent:
  - [ ] 複数戦略のアンサンブルなし
  - [ ] フィルター/SL/TPのペア別最適化なし
  - [ ] 新規通貨ペアの追加なし

---

## Fallback Plan

### If M30 PF < 1.0 for any pair:
1. **PF 0.95-0.99**: 許容（ボーダーライン）、モニタリング継続
2. **PF < 0.95**: そのペアのみH1に戻す or 異なる戦略を試行
3. **全ペアPF < 0.95**: M30移行を中止、H1に完全回帰

### If EURUSD strategy selection fails:
1. EMACross, Breakout, TripleScreenを全て試行
2. 最もPFが高い戦略を採用
3. 全てPF < 1.0の場合、EURUSDを対象から除外（4ペアで運用）
