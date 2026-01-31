# Draft: FX Trading Frequency Tuning

## Requirements (confirmed)
- User wants to increase trade frequency significantly
- Current situation: Sometimes 0 entries per day
- User selected "optimal balance tuning" approach

## Current System Analysis

### Architecture
- Main trading script: `scripts/run_trading.py`
- Config-driven: `config/trading.yaml`, `config/risk.yaml`, `config/model.yaml`
- Backtest infrastructure: `python/src/training/backtest.py`

### Current Bottlenecks Identified

#### 1. Timeframe: H1 (Only 24 candles/day)
- **Impact**: Limited signal opportunities
- **Location**: `CONFIG["timeframe"]` in `run_trading.py:81`

#### 2. Symbol Count: Only 3 pairs
- **Current**: USDJPY, EURJPY, AUDUSD
- **Location**: `CONFIG["symbols"]` in `run_trading.py:80`

#### 3. Strategy Entry Conditions (Very Restrictive)

| Strategy | Current Condition | Why It's Rare |
|----------|------------------|---------------|
| **EMACross** | MACD histogram crosses zero | Happens ~1-2x/day per symbol |
| **TripleScreen** | SMA200 trend + MACD bullish + RSI<50 + Stoch<40 | All 4 conditions must align (very rare) |
| **RSI_Stoch** | RSI<35 AND Stoch<20 | Extreme oversold only |
| **Breakout** | BB squeeze + price breakout | Requires specific volatility pattern |

#### 4. Cumulative Filter Impact

| Filter | Setting | Est. Rejection Rate |
|--------|---------|-------------------|
| `min_confidence` | 0.40 | ~30% |
| `adx_threshold` | 20.0 | ~25% |
| `sma_atr_threshold` | 0.5 | ~20% |
| Time filter (21:00-01:00 UTC excluded) | Hard-coded | ~17% |
| News blackout | 30min before, 15min after | Variable |
| USDJPY long-only | Direction restriction | 50% for 1 symbol |
| Max 1 position per symbol | Position limit | Blocks opportunities |

**Estimated Combined Effect**: Only 10-20% of potential signals pass all filters.

## Key Files to Modify

1. **`scripts/run_trading.py`**:
   - Lines 78-132: CONFIG dictionary
   - Lines 482-628: Strategy implementations
   - Lines 794-827: Filter application

2. **`config/trading.yaml`**:
   - Timeframe settings
   - Symbol list
   - Trading hours

3. **`config/risk.yaml`**:
   - Filter thresholds
   - Position limits

## Open Questions
- [RESOLVED] User selected "optimal balance" approach

## Scope Boundaries
- INCLUDE: All frequency optimization tasks
- INCLUDE: Backtest validation
- EXCLUDE: Fundamental changes to risk management (keep max DD < 30%)
- EXCLUDE: Addition of completely new ML models

## Technical Decisions (Pending)
1. Timeframe change: M30 preferred over M15 (balance quality/quantity)
2. Symbol expansion: Add 2-3 pairs (GBPUSD, EURUSD, GBPJPY?)
3. Strategy relaxation levels: TBD
4. Filter threshold adjustments: TBD
5. New strategies to add: TBD
