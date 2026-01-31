# FX通貨ペア別最適化 - PAIR_CONFIG実装計画

## TL;DR

> **Quick Summary**: run_backtest.pyにPAIR_CONFIGを追加し、12通貨ペアそれぞれに最適なパラメータでバックテストを実行。PF > 1.0のペアを特定して取引頻度と利益を最大化する。
> 
> **Deliverables**:
> - run_backtest.pyにPAIR_CONFIG追加
> - 12ペア全てのバックテスト実行
> - ペア別最適パラメータの特定
> - run_trading.py用の最終PAIR_CONFIG推奨値
> 
> **Estimated Effort**: Medium (4-6時間)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4

---

## Context

### Original Request
通貨ペアごとに戦略やレバレッジを変えて、なるべく多くの通貨ペアで取引を増やし利益を最大化したい。

### Interview Summary
**Key Discussions**:
- 現在run_backtest.pyはGBPUSD, USDJPYのみハードコード
- run_trading.pyには成功パターンのPAIR_CONFIGが存在
- グローバル設定（base_leverage=25, max_leverage=100）を使用中
- 最新結果: GBPUSD(PF 1.17✅), USDJPY(PF 1.02⚠️), AUDUSD(PF 0.64❌)

**Research Findings**:
- run_backtest.py: 1250行超のバックテストスクリプト
- run_trading.py PAIR_CONFIG: 6キー（strategy, min_confidence, use_ml_fallback, long_only, adx_threshold, sma_atr_threshold）
- backtest()関数: leverage, filters, risk managementパラメータを受け入れ
- 閾値探索: [0.50, 0.55, 0.60]でPF≥1.0かつtrades>30を選択

### Self-Review Gap Analysis
**Identified Gaps** (addressed in plan):
1. データファイル存在確認 → Task 1で確認
2. 全12ペアのspread/pip_value定義 → Task 1で定義
3. ペア別レバレッジ適用 → Task 2でbacktest()呼び出し時に適用
4. エラーハンドリング → Task 2でデータ不足ペアをスキップ

---

## Work Objectives

### Core Objective
run_backtest.pyにPAIR_CONFIGを実装し、12通貨ペアそれぞれに最適なパラメータでバックテストを実行して、PF > 1.0のペアを特定する。

### Concrete Deliverables
- `scripts/run_backtest.py` にPAIR_CONFIG定義を追加
- `scripts/run_backtest.py` のメインループをPAIR_CONFIGベースに修正
- 12ペアのバックテスト実行結果（コンソール出力）
- 推奨PAIR_CONFIG値（run_trading.py用）

### Definition of Done
- [ ] 12ペア全てでバックテストが実行される（データ存在ペアのみ）
- [ ] 各ペアでPF, Return, Trades, MaxDDが表示される
- [ ] PF > 1.0のペアが5つ以上特定される
- [ ] 合計年間取引数が100以上になる
- [ ] 実行コマンド: `python scripts/run_backtest.py` が成功する

### Must Have
- PAIR_CONFIG定義（12ペア分）
- ペア別のadx_threshold, sma_atr_threshold
- ペア別のbase_leverage, max_leverage
- ペア別の閾値探索
- spread_pips, pip_valueの12ペア分定義

### Must NOT Have (Guardrails)
- backtest.pyエンジンの変更（エンジンは触らない）
- run_trading.pyの自動更新（結果は手動で反映）
- MLモデルの再学習（既存モデルを使用）
- 新しい戦略の追加（既存ML戦略のみ）
- compute_features()関数の変更

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **User wants tests**: Manual-only
- **Framework**: Manual verification via backtest execution

### Manual Verification Procedures

Each TODO includes EXECUTABLE verification procedures:

**For Script Changes** (using Bash):
```bash
# Agent runs:
python scripts/run_backtest.py
# Assert: No Python errors
# Assert: Output contains "バックテスト結果" for multiple pairs
# Assert: PF values are displayed for each pair
```

**Evidence to Capture:**
- [ ] コンソール出力のスクリーンショットまたはログ
- [ ] 各ペアのPF, Return, Trades値

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: PAIR_CONFIG定義を追加
└── (Sequential within Task 1)

Wave 2 (After Wave 1):
├── Task 2: バックテストループを修正
└── (Sequential within Task 2)

Wave 3 (After Wave 2):
└── Task 3: 12ペアバックテスト実行

Wave 4 (After Wave 3):
└── Task 4: 結果分析と推奨値出力

Critical Path: Task 1 → Task 2 → Task 3 → Task 4
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2 | None |
| 2 | 1 | 3 | None |
| 3 | 2 | 4 | None |
| 4 | 3 | None | None |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1 | delegate_task(category="quick", load_skills=[], run_in_background=false) |
| 2 | 2 | delegate_task(category="unspecified-low", load_skills=[], run_in_background=false) |
| 3 | 3 | Manual execution by user or agent |
| 4 | 4 | delegate_task(category="quick", load_skills=[], run_in_background=false) |

---

## TODOs

- [ ] 1. PAIR_CONFIG定義をrun_backtest.pyに追加

  **What to do**:
  - run_backtest.pyの先頭（import文の後）にPAIR_CONFIG辞書を定義
  - 12ペア全ての設定を含める
  - spread_pips_by_symbol, pip_value_by_symbolをPAIR_CONFIGに統合
  - ボラティリティベースのレバレッジ設定を定義

  **Must NOT do**:
  - compute_features()関数を変更しない
  - backtest()関数のシグネチャを変更しない
  - 既存のグローバル変数を削除しない（後方互換性）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 単純な辞書定義の追加、明確なパターンあり
  - **Skills**: `[]`
    - Reason: 特殊なスキル不要

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 1)
  - **Blocks**: Task 2
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `scripts/run_trading.py:82-117` - PAIR_CONFIG定義パターン（strategy, min_confidence, use_ml_fallback, long_only, adx_threshold, sma_atr_threshold）
  - `scripts/run_backtest.py:989-1003` - 現在のspread_pips_by_symbol, pip_value_by_symbol定義

  **Implementation Details**:
  ```python
  # 追加位置: run_backtest.py の import文の後（約line 28付近）
  
  PAIR_CONFIG = {
      # === メジャーペア ===
      "EURUSD": {
          "enabled": True,
          "spread_pips": 1.0,
          "pip_value": 0.0001,
          "adx_threshold": 15.0,
          "sma_atr_threshold": 0.3,
          "long_only": False,
          "base_leverage": 25.0,  # 中ボラ
          "max_leverage": 80.0,
          "threshold_search": [0.50, 0.55, 0.60],
      },
      "USDJPY": {
          "enabled": True,
          "spread_pips": 1.2,
          "pip_value": 0.01,
          "adx_threshold": 15.0,
          "sma_atr_threshold": 0.3,
          "long_only": True,  # ロングのみ
          "base_leverage": 25.0,
          "max_leverage": 80.0,
          "threshold_search": [0.50, 0.55, 0.60],
      },
      "GBPUSD": {
          "enabled": True,
          "spread_pips": 1.5,
          "pip_value": 0.0001,
          "adx_threshold": 12.0,  # 低め
          "sma_atr_threshold": 0.2,
          "long_only": False,
          "base_leverage": 15.0,  # 高ボラ
          "max_leverage": 50.0,
          "threshold_search": [0.50, 0.55, 0.60],
      },
      "USDCHF": {
          "enabled": True,
          "spread_pips": 1.5,
          "pip_value": 0.0001,
          "adx_threshold": 15.0,
          "sma_atr_threshold": 0.3,
          "long_only": False,
          "base_leverage": 30.0,  # 低ボラ
          "max_leverage": 100.0,
          "threshold_search": [0.50, 0.55, 0.60],
      },
      "USDCAD": {
          "enabled": True,
          "spread_pips": 1.5,
          "pip_value": 0.0001,
          "adx_threshold": 15.0,
          "sma_atr_threshold": 0.3,
          "long_only": False,
          "base_leverage": 25.0,
          "max_leverage": 80.0,
          "threshold_search": [0.50, 0.55, 0.60],
      },
      "AUDUSD": {
          "enabled": True,
          "spread_pips": 1.2,
          "pip_value": 0.0001,
          "adx_threshold": 15.0,
          "sma_atr_threshold": 0.3,
          "long_only": False,
          "base_leverage": 25.0,
          "max_leverage": 80.0,
          "threshold_search": [0.50, 0.55, 0.60],
      },
      # === クロスペア ===
      "EURJPY": {
          "enabled": True,
          "spread_pips": 1.5,
          "pip_value": 0.01,
          "adx_threshold": 15.0,
          "sma_atr_threshold": 0.3,
          "long_only": False,
          "base_leverage": 20.0,
          "max_leverage": 70.0,
          "threshold_search": [0.50, 0.55, 0.60],
      },
      "GBPJPY": {
          "enabled": True,
          "spread_pips": 2.0,
          "pip_value": 0.01,
          "adx_threshold": 12.0,
          "sma_atr_threshold": 0.2,
          "long_only": False,
          "base_leverage": 15.0,  # 高ボラ
          "max_leverage": 50.0,
          "threshold_search": [0.50, 0.55, 0.60],
      },
      "AUDJPY": {
          "enabled": True,
          "spread_pips": 1.5,
          "pip_value": 0.01,
          "adx_threshold": 15.0,
          "sma_atr_threshold": 0.3,
          "long_only": False,
          "base_leverage": 20.0,
          "max_leverage": 70.0,
          "threshold_search": [0.50, 0.55, 0.60],
      },
      "CADJPY": {
          "enabled": True,
          "spread_pips": 1.8,
          "pip_value": 0.01,
          "adx_threshold": 15.0,
          "sma_atr_threshold": 0.3,
          "long_only": False,
          "base_leverage": 20.0,
          "max_leverage": 70.0,
          "threshold_search": [0.50, 0.55, 0.60],
      },
      "EURGBP": {
          "enabled": True,
          "spread_pips": 1.5,
          "pip_value": 0.0001,
          "adx_threshold": 15.0,
          "sma_atr_threshold": 0.3,
          "long_only": False,
          "base_leverage": 25.0,
          "max_leverage": 80.0,
          "threshold_search": [0.50, 0.55, 0.60],
      },
      "EURAUD": {
          "enabled": True,
          "spread_pips": 2.0,
          "pip_value": 0.0001,
          "adx_threshold": 15.0,
          "sma_atr_threshold": 0.3,
          "long_only": False,
          "base_leverage": 20.0,
          "max_leverage": 70.0,
          "threshold_search": [0.50, 0.55, 0.60],
      },
  }
  ```

  **Acceptance Criteria**:

  **Automated Verification**:
  ```bash
  # Agent runs:
  python -c "import sys; sys.path.insert(0, 'scripts'); from run_backtest import PAIR_CONFIG; print(f'PAIR_CONFIG defined: {len(PAIR_CONFIG)} pairs')"
  # Assert: Output is "PAIR_CONFIG defined: 12 pairs"
  # Assert: Exit code 0
  ```

  **Commit**: YES
  - Message: `feat(backtest): add PAIR_CONFIG for 12 currency pairs`
  - Files: `scripts/run_backtest.py`
  - Pre-commit: `python -c "from scripts.run_backtest import PAIR_CONFIG"`

---

- [ ] 2. バックテストループをPAIR_CONFIGベースに修正

  **What to do**:
  - main()関数のsymbols定義をPAIR_CONFIGから取得するよう変更
  - 各ペアのループ内でPAIR_CONFIGから設定を取得
  - backtest()およびwalk_forward_backtest()呼び出し時にペア別パラメータを適用
  - 閾値探索をPAIR_CONFIGのthreshold_searchを使用
  - データファイルが存在しないペアはスキップ

  **Must NOT do**:
  - backtest()関数のシグネチャを変更しない
  - walk_forward_backtest()関数のシグネチャを変更しない
  - compute_features()を変更しない

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: ループ修正とパラメータ適用、中程度の複雑さ
  - **Skills**: `[]`
    - Reason: 特殊なスキル不要

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 2)
  - **Blocks**: Task 3
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `scripts/run_trading.py:910-957` - predict()でのPAIR_CONFIG使用パターン（pair_cfg = PAIR_CONFIG.get(symbol, {...})）
  - `scripts/run_trading.py:959-1000` - apply_filters()でのペア別閾値適用パターン
  - `scripts/run_backtest.py:966-1000` - 現在のmain()関数の構造
  - `scripts/run_backtest.py:1042-1100` - 現在のペアループ構造

  **Implementation Details**:
  
  1. **symbols定義の変更** (約line 981):
  ```python
  # Before:
  symbols = ["GBPUSD", "USDJPY"]
  
  # After:
  symbols = [s for s, cfg in PAIR_CONFIG.items() if cfg.get("enabled", True)]
  ```

  2. **ペアループ内でPAIR_CONFIG取得** (約line 1042):
  ```python
  for symbol in symbols:
      # ペア別設定を取得
      pair_cfg = PAIR_CONFIG.get(symbol, {})
      spread_pips = pair_cfg.get("spread_pips", 2.0)
      pip_value = pair_cfg.get("pip_value", 0.0001)
      adx_threshold = pair_cfg.get("adx_threshold", 15.0)
      sma_atr_threshold = pair_cfg.get("sma_atr_threshold", 0.3)
      base_leverage = pair_cfg.get("base_leverage", 25.0)
      max_leverage = pair_cfg.get("max_leverage", 100.0)
      long_only = pair_cfg.get("long_only", False)
      threshold_search_values = pair_cfg.get("threshold_search", [0.50, 0.55, 0.60])
      
      # データ読み込み（存在しない場合はスキップ）
      df = load_data(symbol, timeframe, data_dir)
      if df is None:
          print(f"  {symbol}: データファイルなし、スキップ")
          continue
  ```

  3. **閾値探索でペア別設定を使用** (約line 1134):
  ```python
  # Before:
  thresholds = [0.50, 0.55, 0.60]
  
  # After:
  thresholds = threshold_search_values
  ```

  4. **backtest/walk_forward呼び出しでペア別パラメータ適用** (約line 1143-1164):
  ```python
  test_result = walk_forward_backtest(
      df, X, y,
      # ... 既存パラメータ ...
      adx_threshold=adx_threshold,  # ← ペア別
      sma_atr_threshold=sma_atr_threshold,  # ← ペア別
      base_leverage=base_leverage,  # ← ペア別
      max_leverage=max_leverage,  # ← ペア別
      # ...
  )
  ```

  5. **long_only制約の適用** (backtest関数内で既に実装済み):
  ```python
  # run_backtest.py line 670-672 で既に実装されている
  # symbol引数でUSDJPYを渡せばロングのみになる
  # PAIR_CONFIGのlong_onlyに基づいて動的に適用する場合は
  # backtest()呼び出し前にチェック
  ```

  **Acceptance Criteria**:

  **Automated Verification**:
  ```bash
  # Agent runs:
  python -c "
  import sys
  sys.path.insert(0, 'scripts')
  from run_backtest import PAIR_CONFIG, main
  symbols = [s for s, cfg in PAIR_CONFIG.items() if cfg.get('enabled', True)]
  print(f'Enabled symbols: {len(symbols)}')
  assert len(symbols) == 12, f'Expected 12, got {len(symbols)}'
  print('Loop modification verified')
  "
  # Assert: Output contains "Enabled symbols: 12"
  # Assert: Exit code 0
  ```

  **Commit**: YES
  - Message: `feat(backtest): use PAIR_CONFIG in backtest loop for per-pair optimization`
  - Files: `scripts/run_backtest.py`
  - Pre-commit: `python -c "from scripts.run_backtest import PAIR_CONFIG, main"`

---

- [ ] 3. 12ペアバックテスト実行

  **What to do**:
  - `python scripts/run_backtest.py` を実行
  - 各ペアのバックテスト結果を確認
  - 結果をファイルまたはコンソールから記録

  **Must NOT do**:
  - コードの追加変更（実行のみ）
  - 結果に基づくrun_trading.pyの自動更新

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: スクリプト実行のみ
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 3)
  - **Blocks**: Task 4
  - **Blocked By**: Task 2

  **References**:
  - `scripts/run_backtest.py` - 実行対象スクリプト
  - `data/raw/` - データファイルの格納場所

  **Acceptance Criteria**:

  **Automated Verification**:
  ```bash
  # Agent runs:
  cd C:\Users\kuniyoshi\Desktop\projects\trade
  python scripts/run_backtest.py 2>&1 | tee backtest_results.log
  # Assert: Exit code 0
  # Assert: Output contains "バックテスト結果" for multiple pairs
  # Assert: Output contains "Profit Factor" values
  ```

  **Evidence to Capture:**
  - [ ] backtest_results.log ファイル
  - [ ] 各ペアのPF, Return, Trades, MaxDD値

  **Commit**: NO (実行結果のみ、コード変更なし)

---

- [ ] 4. 結果分析と推奨PAIR_CONFIG出力

  **What to do**:
  - バックテスト結果からPF > 1.0のペアを抽出
  - 各ペアの最適閾値を特定
  - run_trading.py用の推奨PAIR_CONFIG値を出力
  - 結果サマリーを作成

  **Must NOT do**:
  - run_trading.pyを自動で更新しない
  - run_backtest.pyを追加変更しない

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 結果分析と出力のみ
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 4)
  - **Blocks**: None
  - **Blocked By**: Task 3

  **References**:
  - Task 3の実行結果（backtest_results.log）
  - `scripts/run_trading.py:82-117` - 推奨PAIR_CONFIGのフォーマット

  **Expected Output Format**:
  ```
  === 結果サマリー ===
  
  PF > 1.0 のペア (採用推奨):
  | Pair    | PF   | Return | Trades | 最適閾値 | 推奨レバレッジ |
  |---------|------|--------|--------|---------|--------------|
  | GBPUSD  | 1.17 | +67.2% | 236    | 0.60    | 15-50        |
  | ...     | ...  | ...    | ...    | ...     | ...          |
  
  PF < 1.0 のペア (除外推奨):
  | Pair    | PF   | Return | Trades | 備考 |
  |---------|------|--------|--------|------|
  | AUDUSD  | 0.64 | -46.1% | 76     | 除外 |
  | ...     | ...  | ...    | ...    | ...  |
  
  === run_trading.py 推奨PAIR_CONFIG ===
  PAIR_CONFIG = {
      "GBPUSD": {
          "strategy": "ML_Primary",
          "min_confidence": 0.60,
          ...
      },
      ...
  }
  ```

  **Acceptance Criteria**:

  **Automated Verification**:
  ```bash
  # Agent runs:
  grep -c "PF" backtest_results.log
  # Assert: Count > 0 (results exist)
  
  # Extract PF > 1.0 pairs
  grep "Profit Factor" backtest_results.log | grep -E "[1-9]\.[0-9]+"
  # Assert: At least 5 lines (5+ pairs with PF > 1.0)
  ```

  **Commit**: NO (分析結果のみ、コード変更なし)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(backtest): add PAIR_CONFIG for 12 currency pairs` | run_backtest.py | python import check |
| 2 | `feat(backtest): use PAIR_CONFIG in backtest loop for per-pair optimization` | run_backtest.py | python import check |
| 3 | (no commit - execution only) | - | - |
| 4 | (no commit - analysis only) | - | - |

---

## Success Criteria

### Verification Commands
```bash
# 1. PAIR_CONFIG定義確認
python -c "from scripts.run_backtest import PAIR_CONFIG; print(len(PAIR_CONFIG))"
# Expected: 12

# 2. バックテスト実行
python scripts/run_backtest.py
# Expected: 12ペアの結果出力

# 3. PF > 1.0のペア数確認
grep "Profit Factor" backtest_results.log | grep -cE "[1-9]\.[0-9]+"
# Expected: >= 5
```

### Final Checklist
- [ ] 12ペア分のPAIR_CONFIGが定義されている
- [ ] バックテストループがPAIR_CONFIGベースで動作する
- [ ] 各ペアでペア別パラメータが適用される
- [ ] PF > 1.0のペアが5つ以上特定される
- [ ] 合計年間取引数が100以上になる
- [ ] run_trading.py用の推奨PAIR_CONFIGが出力される

---

## Risk Mitigation

### Data Availability
- **Risk**: 一部のペアでデータファイルが存在しない可能性
- **Mitigation**: データ不足ペアはスキップしてログ出力

### Performance
- **Risk**: 12ペアのバックテストに時間がかかる可能性
- **Mitigation**: Walk-forward期間を維持、並列化は見送り

### Overfitting
- **Risk**: ペア別最適化による過学習リスク
- **Mitigation**: 閾値探索を3値に限定、Walk-forwardで検証
