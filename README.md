# Stablecoin Basis / Arbitrage Monitor

## Disclaimer
This project is for educational and research purposes only. Cryptocurrency markets are volatile and risky. Do not trade with funds you cannot afford to lose. Nothing in this repository is financial advice.

## Overview
This bot monitors stablecoin prices across centralized exchanges (CEXs), looking for small basis/arb gaps between the same pair (e.g., USDT/USD or USDC/USDT). It:
- pulls level-2 order books via `ccxt`,
- computes VWAP buy/sell prices for a configurable trade size,
- accounts for taker fees,
- flags opportunities whose net edge exceeds a threshold,
- optionally records order books to CSV and replays them offline for backtesting.

The code is deliberately simple: a single engine script, a YAML config, and a CSV-based recorder/replayer. I really don't recommend actually using this to trade.

## Key Features
- Cross-venue detection for the same symbol (e.g., `USDT/USD` on multiple venues).
- VWAP math on depth (not just top-of-book).
- Fee-aware edge calculation in basis points (bps).
- De-peg kill-switch (suppresses entries when a stable trades away from 1.00 for a sustained window).
- Recording and offline replay with time bucketing to approximate “fresh” quotes across venues.
- No API keys required for public order books.

## Repo Layout
```
scripts/            # optional helpers (bash)
  record.sh
  replay.sh
  run_live.sh
config.yaml         # venues, risk limits, and logging
requirements.txt    # python deps
stablecoin_arb_engine.py
```


## Installation
Prereqs: Python 3.10+ recommended.

```bash
git clone <your-repo-url>
cd stable-arb-bot
python -m venv .venv
# Windows (cmd):
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration
Edit `config.yaml`. Keep at least two venues sharing the same symbol for cross-venue comparisons to actually work. My example uses USDT/USD everywhere:

```yaml
base_units: 100.0
poll_interval_ms: 750
log_csv: snapshots/snapshots.csv

venues:
  - name: binanceus
    symbol: "USDT/USD"
    taker_fee_bps: 0.0
    depth_levels: 20

  - name: kraken
    symbol: "USDT/USD"
    taker_fee_bps: 0.0
    depth_levels: 20

  - name: bitstamp
    symbol: "USDT/USD"
    taker_fee_bps: 0.0
    depth_levels: 20

  - name: coinbase
    symbol: "USDT/USD"
    taker_fee_bps: 0.0
    depth_levels: 20

risk:
  edge_entry_bps: 0.1
  edge_exit_bps: 2.0
  max_abs_depeg_bps: 150.0
  depeg_window_sec: 60
  price_ttl_ms: 5000
  max_trade_units: 10000.0
```

Important fields:
- `base_units`: Notional base size used to compute VWAP (e.g., 100.0 means “simulate trading $100 notional”).
- `taker_fee_bps`: Per-venue taker fee in basis points (100 bps = 1%). Set to your tier.
- `edge_entry_bps`: Minimum net edge to flag as an opportunity.
- `price_ttl_ms`: Quotes older than this are considered stale (affects both live scan and replay bucketing).
- `log_csv`: Where live snapshots are written (created if missing).

## Running
Live (paper):
```bash
python stablecoin_arb_engine.py --config config.yaml run-live
```
You’ll see lines like:
```
[...timestamp...] OPP USDT/USD: BUY kraken@0.99990 -> SELL binanceus@1.00010 | size=100.00 | edge=1.50 bps | pnl=$0.0015
```

Record snapshots for N seconds:
```bash
python stablecoin_arb_engine.py --config config.yaml record --outfile snapshots/snapshots.csv --duration 180
```

Replay from CSV (backtest):
```bash
python stablecoin_arb_engine.py --config config.yaml replay --infile snapshots/snapshots.csv
```
Replay aggregates snapshots into time buckets (≤ `price_ttl_ms`) so venues within the same bucket are evaluated together. It supports partial fills if the requested size exceeds available depth.

## CSV Format
`stablecoin_arb_engine.py` writes bids and asks up to N levels per venue per second (N defaults to 30 in live/record). Columns:
```
ts_ms, venue, symbol, side, price, size
```
Example:
```
1724720645123,kraken,USDT/USD,bid,1.0001000000,25000.000000
1724720645123,kraken,USDT/USD,ask,1.0002000000,18000.000000
...
```

## How PnL Is Calculated
For a candidate size `Q`:
- Buy VWAP from venue A’s asks, apply buy taker fee.
- Sell VWAP into venue B’s bids, apply sell taker fee.
- Edge (bps) = ((sell_revenue - buy_cost) / buy_cost) * 10,000.
- Paper PnL = `sell_revenue - buy_cost`.
If `edge >= edge_entry_bps`, it is counted as an opportunity in replay and printed in live mode.

## Architecture
- Data: `ccxt` public order books (no keys). Rate-limited per venue.
- Math: VWAP over depth, fee multipliers, edge in bps.
- Risk: De-peg kill-switch, stale-quote TTL, per-trade size cap.
- I/O: CSV writer/reader + time-bucketed replay with optional partial-fill logic.

## Tips
- Make sure symbols match across venues (e.g., `USDT/USD` on all of them) to get cross-venue edges.
- Increase `base_units` only if there’s sufficient depth; otherwise, partial fills will reduce effective size.
- Start with `edge_entry_bps` low to inspect signals, then raise it toward realistic net-profit thresholds after fees.
- If a venue blocks your region, replace it with one that lists the same pair.

## License
MIT. See `LICENSE`.
