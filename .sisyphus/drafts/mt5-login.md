# Draft: MT5 Login / Data Download

## Requirements (confirmed)
- User reports MT5 shows "Connected" in UI.
- Need to complete MT5 data download verification task.

## Technical Decisions
- Prefer identifying correct MT5 terminal instance via explicit terminal path for mt5.initialize.

## Research Findings
- scripts/download_data.py uses mt5.initialize() without params, which relies on default terminal session.
- python/src/data/loader.py supports mt5.initialize(path/login/password/server) but is not used by download script.
- config/trading.yaml contains broker server/account but no password or terminal path.

## Open Questions
- Is only one MT5 terminal installed/running? If multiple, what is the terminal64.exe path for the logged-in instance?

## Scope Boundaries
- INCLUDE: MT5 connection verification and data download execution steps.
- EXCLUDE: Code changes to add terminal path/password (unless explicitly requested).
