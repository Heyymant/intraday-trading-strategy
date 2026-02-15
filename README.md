# Optimised P3 Mean-Reversion Alpha

Intraday trading strategy for P3 using structural mean-reversion at long horizons.

**Performance (96 OOS days)**: Sharpe 3.27 | Return +10.48% | Net PnL +$1,048 | Max DD $280

## Requirements

```
Python >= 3.9
numpy
pandas
```

No ML libraries required. The strategy uses only NumPy and Pandas.

## Usage

### Single Day Execution (Primary Interface)

```bash
python strategy.py --input train/1.csv --output trades_1.csv
```

This reads a single day CSV, runs the strategy causally (bar-by-bar), and outputs the trade log.

### Batch Mode (Walk-Forward)

```bash
python strategy.py --train-dir train --output-dir output_final
```

Runs rolling walk-forward execution across all days. The first 15 days are used for calibration; days 16+ are executed out-of-sample.

### Custom Starting Capital

```bash
python strategy.py --input train/1.csv --output trades_1.csv --capital 50000
```

## Output Format

The trade log CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `ts` | Timestamp (nanoseconds) |
| `price` | P3 price at this bar |
| `signal` | Combined MR z-score signal |
| `mr_signal` | Raw mean-reversion signal |
| `position` | Current position (+N long, -N short, 0 flat) |
| `entry_price` | Price at which current position was entered |
| `realized_pnl` | Realized PnL from any trade at this bar |
| `mtm_pnl` | Mark-to-market unrealized PnL |
| `trade_tc` | Transaction cost incurred at this bar |
| `cum_tc` | Cumulative transaction costs |
| `cum_realized_pnl` | Cumulative realized PnL |
| `cum_net_pnl` | Cumulative net PnL (realized - TC) |
| `equity` | Current equity (capital + net PnL + MTM) |

## File Structure

```
final_strategy/
  strategy.py        # Main executable strategy
  WRITEUP.md         # Quant theory doc (modeling rationale)
  README.md          # This file
  analysis.ipynb     # Jupyter notebook with full analysis
  Problem statement.pdf
  output_final/      # Pre-computed results (96 OOS days)
    results.json     # Summary performance metrics
    daily_summary.csv
    trades_*.csv     # Per-day trade logs
```

## Strategy Overview

1. **Signal**: Z-score of P3 relative to blended rolling mean (800 + 2000 bar windows)
2. **Entry**: Only at extreme deviations (|z| > 3.0) with momentum confirmation
3. **Hold**: ~8000 bars to capture full mean-reversion cycle
4. **Exit**: Time-based (target hold) / stop-loss (10x volatility) / EOD (99%)
5. **Sizing**: Vol-targeted, max 150 units, drawdown-scaled
6. **TC**: 0.01% per side on all entries, exits, and flips

## Causality Enforcement

- All rolling windows use only past and current data (`prices[:t+1]`)
- No forward-looking normalization
- Walk-forward execution: OLS coefficients re-estimated every 5 days using only past data
- Signal warm-up: first 1500 bars are observation-only (no trading)
