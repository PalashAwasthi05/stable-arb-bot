#!/usr/bin/env bash
set -euo pipefail
python stablecoin_arb_poc.py replay --config config.yaml --infile snapshots/snapshots.csv
