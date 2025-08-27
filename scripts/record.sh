#!/usr/bin/env bash
set -euo pipefail
mkdir -p snapshots
python stablecoin_arb_poc.py record --config config.yaml --outfile snapshots/snapshots.csv --duration "${1:-180}"
