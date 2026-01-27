# Draft: Walk-Forward PnL (2021-Now)

## Requirements (confirmed)
- User wants walk-forward evaluation (not in-sample).
- Initial capital: 1000 USD.
- Start date requested: 2021-01-01 (current data begins 2021-01-27).
- Include transaction costs.
  - Defaults accepted: EURUSD 1.0 pips, USDJPY 1.5 pips, GBPUSD 1.8 pips, slippage 0.2 pips.
  - Initial training window: 2 years.

## Technical Decisions
- None yet (need walk-forward configuration).

## Research Findings
- Current data range available in MT5 download: 2021-01-27 to 2026-01-26.
- Existing backtest script uses a single 80/20 split; walk-forward not implemented yet.

## Open Questions
- Walk-forward scheme and retrain cadence (expanding vs rolling, step size).
- Exact transaction cost model (spread, commission, slippage values).
- Initial training window length for first model (e.g., 1y/2y/3y) and evaluation window size.

## Scope Boundaries
- INCLUDE: Walk-forward backtest across full period with periodic retraining and OOS evaluation.
- EXCLUDE: Live trading changes; feature engineering changes unless requested.
