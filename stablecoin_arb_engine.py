from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import time
from datetime import datetime, timezone
import math
import signal
import ccxt  
import pandas as pd  
import numpy as np  
import yaml  


# --------------- structures and config, dont touch -----------------

@dataclass
class CEXVenue:
    name: str #ccxt exchange
    symbol: str              
    taker_fee_bps: float
    maker_fee_bps: float = 0.0
    depth_levels: int = 20
    rate_limit_ms: int = 200


@dataclass
class DEXPool:
    name: str
    base_token: str
    quote_token: str
    reserve_base: float
    reserve_quote: float
    lp_fee_bps: float = 3.0
    gas_usd: float = 0.0

@dataclass
class RiskLimits:
    edge_entry_bps: float = 8.0
    edge_exit_bps: float = 2.0
    max_abs_depeg_bps: float = 150.0
    depeg_window_sec: int = 60
    price_ttl_ms: int = 2500
    max_trade_units: float = 10000.0

@dataclass
class ProjectConfig:
    base_units: float = 1000.0
    venues: List[CEXVenue] = field(default_factory=list)
    dex_pools: List[DEXPool] = field(default_factory=list)
    risk: RiskLimits = field(default_factory=RiskLimits)
    poll_interval_ms: int = 750
    log_csv: Optional[str] = None

    @staticmethod
    def from_yaml(path: str) -> "ProjectConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        venues = [
            CEXVenue(
                name=str(v["name"]).strip(),
                symbol=str(v["symbol"]).strip(),
                taker_fee_bps=float(v.get("taker_fee_bps", 5.0)),
                maker_fee_bps=float(v.get("maker_fee_bps", 0.0)),
                depth_levels=int(v.get("depth_levels", 20)),
                rate_limit_ms=int(v.get("rate_limit_ms", 200)),
            )
            for v in raw.get("venues", [])
        ]

        dex_pools = [
            DEXPool(
                name=str(p["name"]).strip(),
                base_token=str(p["base_token"]).strip(),
                quote_token=str(p["quote_token"]).strip(),
                reserve_base=float(p["reserve_base"]),
                reserve_quote=float(p["reserve_quote"]),
                lp_fee_bps=float(p.get("lp_fee_bps", 3.0)),
                gas_usd=float(p.get("gas_usd", 0.0)),
            )
            for p in raw.get("dex_pools", [])
        ]

        r = raw.get("risk", {})
        risk = RiskLimits(
            edge_entry_bps=float(r.get("edge_entry_bps", 8.0)),
            edge_exit_bps=float(r.get("edge_exit_bps", 2.0)),
            max_abs_depeg_bps=float(r.get("max_abs_depeg_bps", 150.0)),
            depeg_window_sec=int(r.get("depeg_window_sec", 60)),
            price_ttl_ms=int(r.get("price_ttl_ms", 2500)),
            max_trade_units=float(r.get("max_trade_units", 10000.0)),
        )

        return ProjectConfig(
            base_units=float(raw.get("base_units", 1000.0)),
            venues=venues,
            dex_pools=dex_pools,
            risk=risk,
            poll_interval_ms=int(raw.get("poll_interval_ms", 750)),
            log_csv=raw.get("log_csv"),
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def ts_ms_now() -> int:
    return int(time.time() * 1000)

def log(msg: str) -> None:
    now = utc_now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{now}] {msg}", flush=True)


# --------------- math ops (vwap, fees calc, edge) -----------------


@dataclass
class VWAPResult:
    vwap: Optional[float]
    filled: float

def vwap_price(order_side: List[Tuple[float, float]], amount: float) -> VWAPResult:
    if amount <= 0:
        return VWAPResult(vwap=None, filled=0.0)
    remaining = amount
    notional = 0.0
    filled = 0.0
    for price, size in order_side:
        if size <= 0:
            continue
        take = min(remaining, size)
        notional += take * price
        filled += take
        remaining -= take
        if remaining <= 1e-12:
            break
    if filled <= 0:
        return VWAPResult(vwap=None, filled=0.0)
    if filled + 1e-12 < amount:
        return VWAPResult(vwap=None, filled=filled)
    return VWAPResult(vwap=notional / filled, filled=filled)

def bps(x: float) -> float:
    return x * 10000.0

def edge_bps(buy_cost_usd: float, sell_revenue_usd: float) -> float:
    if buy_cost_usd <= 0:
        return -math.inf
    return bps((sell_revenue_usd - buy_cost_usd) / buy_cost_usd)

def taker_fee_multiplier(fee_bps: float, side: str) -> float:
    """side='buy' -> pay (1+fee), side='sell' -> receive (1-fee)."""
    if fee_bps < 0:
        raise ValueError("fee_bps must be >= 0")
    if side == "buy":
        return 1.0 + (fee_bps / 10000.0)
    if side == "sell":
        return 1.0 - (fee_bps / 10000.0)
    raise ValueError("side must be 'buy' or 'sell'")

# --------------- gathering exchange adapters -----------------

class CEXAdapter: #had to adjust to enable 3 fields
    def __init__(self, venue: CEXVenue):
        self.v = venue
        cls = getattr(ccxt, self.v.name)
        self.ex = cls({"enableRateLimit": True})

    async def load_markets(self) -> None:
        await asyncio.to_thread(self.ex.load_markets)

    async def fetch_order_book(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], int]:
        def _fetch():
            return self.ex.fetch_order_book(self.v.symbol, limit=self.v.depth_levels)
        ob = await asyncio.to_thread(_fetch)

        def _norm(levels):
            out = []
            for lvl in levels:
                # lvl can be [price, amount] or [price, amount, ...] or a dict
                if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                    price, size = float(lvl[0]), float(lvl[1])
                elif isinstance(lvl, dict):
                    price = float(lvl.get("price", lvl.get(0)))
                    size  = float(lvl.get("amount", lvl.get(1)))
                else:
                    continue
                out.append((price, size))
            return out

        bids = _norm(ob.get("bids", []))
        asks = _norm(ob.get("asks", []))
        nowms = ts_ms_now()
        return bids, asks, nowms


# --------------- engine (vroom) -----------------


@dataclass
class Quote:
    venue: str
    symbol: str
    ts_ms: int
    bid_levels: List[Tuple[float, float]]
    ask_levels: List[Tuple[float, float]]

@dataclass
class Opportunity:
    buy_venue: str
    sell_venue: str
    symbol: str
    size_base: float
    buy_vwap: float
    sell_vwap: float
    buy_fee_bps: float
    sell_fee_bps: float
    buy_cost_usd: float
    sell_revenue_usd: float
    edge_bps: float
    created_at: datetime

class StableArbEngine:
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        self.adapters: Dict[str, CEXAdapter] = {}
        self._stop = False
        self._recent_prices: Dict[str, List[Tuple[int, float]]] = {}  # symbol -> [(ts_ms, mid)]

    async def init_adapters(self) -> None:
        for v in self.cfg.venues:
            ad = CEXAdapter(v)
            try:
                await ad.load_markets()
            except Exception as e:
                log(f"Skipping venue {v.name}: {type(e).__name__}: {e}")
                continue
            self.adapters[v.name] = ad
            log(f"Initialized venue: {v.name} {v.symbol}")

        if not self.adapters:
            raise RuntimeError("No venues initialized successfully.") #adjusted bc binance global blocks US, switched to US but realized should make fallbac


    def stop(self) -> None:
        self._stop = True

    def _record_mid(self, symbol: str, bid: Optional[float], ask: Optional[float], ts_ms: int) -> None:
        if bid is None or ask is None:
            return
        mid = 0.5 * (bid + ask)
        buf = self._recent_prices.setdefault(symbol, [])
        buf.append((ts_ms, mid))
        cutoff = ts_ms - self.cfg.risk.depeg_window_sec * 1000
        self._recent_prices[symbol] = [(t, m) for (t, m) in buf if t >= cutoff]

    def _depeg_triggered(self, symbol: str) -> bool:
        data = self._recent_prices.get(symbol, [])
        if not data:
            return False
        threshold = self.cfg.risk.max_abs_depeg_bps / 10000.0
        # if ANY mid within window is within threshold, don't trigger
        for _, mid in data:
            if abs(mid - 1.0) <= threshold:
                return False
        return True  # all mids outside threshold for the whole window

    async def scan_once(self) -> Tuple[List[Quote], List[Opportunity]]:
        quotes: List[Quote] = []

        async def fetch_one(v: CEXVenue) -> Optional[Quote]:
            try:
                ad = self.adapters.get(v.name)
                if ad is None:
                    return None
                bids, asks, nowms = await ad.fetch_order_book()
                b0 = bids[0][0] if bids else None
                a0 = asks[0][0] if asks else None
                self._record_mid(v.symbol, b0, a0, nowms)
                return Quote(venue=v.name, symbol=v.symbol, ts_ms=nowms, bid_levels=bids, ask_levels=asks)
            except Exception as e:
                log(f"WARN: fetch_order_book failed for {v.name}: {type(e).__name__}: {e}")
                return None

        tasks = [fetch_one(v) for v in self.cfg.venues if v.name in self.adapters]
        results = await asyncio.gather(*tasks)
        for q in results:
            if q is not None:
                quotes.append(q)

        blocked_symbols = {v.symbol for v in self.cfg.venues if self._depeg_triggered(v.symbol)}
        for s in blocked_symbols:
            log(f"KILL-SWITCH: sustained de-peg detected on {s}; suppressing entries.")

        opps: List[Opportunity] = []
        for i in range(len(quotes)):
            for j in range(len(quotes)):
                if i == j:
                    continue
                qi, qj = quotes[i], quotes[j]
                if qi.symbol != qj.symbol or qi.symbol in blocked_symbols:
                    continue
                nowms = ts_ms_now()
                if (nowms - qi.ts_ms) > self.cfg.risk.price_ttl_ms:
                    continue
                if (nowms - qj.ts_ms) > self.cfg.risk.price_ttl_ms:
                    continue
                size = min(self.cfg.base_units, self.cfg.risk.max_trade_units)

                buy = vwap_price(qi.ask_levels, size)   # buy at asks
                sell = vwap_price(qj.bid_levels, size)  # sell to bids
                if buy.vwap is None or sell.vwap is None:
                    continue

                buy_fee = next(v.taker_fee_bps for v in self.cfg.venues if v.name == qi.venue)
                sell_fee = next(v.taker_fee_bps for v in self.cfg.venues if v.name == qj.venue)

                buy_cost = size * buy.vwap * taker_fee_multiplier(buy_fee, "buy")
                sell_rev = size * sell.vwap * taker_fee_multiplier(sell_fee, "sell")
                e = edge_bps(buy_cost, sell_rev)

                if e >= self.cfg.risk.edge_entry_bps:
                    opps.append(Opportunity(
                        buy_venue=qi.venue,
                        sell_venue=qj.venue,
                        symbol=qi.symbol,
                        size_base=size,
                        buy_vwap=buy.vwap,
                        sell_vwap=sell.vwap,
                        buy_fee_bps=buy_fee,
                        sell_fee_bps=sell_fee,
                        buy_cost_usd=buy_cost,
                        sell_revenue_usd=sell_rev,
                        edge_bps=e,
                        created_at=utc_now(),
                    ))
        return quotes, opps

# --------------- csv i/o -----------------


CSV_FIELDS = ["ts_ms", "venue", "symbol", "side", "price", "size"]

def os_path_exists(p: str) -> bool:
    try:
        import os
        return os.path.exists(p)
    except Exception:
        return False

def write_snapshot_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    # ensure parent directory exists bc i low key forgot to make one first run
    import os
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    new_file = not os_path_exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def quotes_to_rows(quotes: List[Quote], max_levels: int = 10) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for q in quotes:
        bids = q.bid_levels[:max_levels]
        asks = q.ask_levels[:max_levels]
        for price, size in bids:
            rows.append({"ts_ms": q.ts_ms, "venue": q.venue, "symbol": q.symbol,
                         "side": "bid", "price": f"{price:.10f}", "size": f"{size:.6f}"})
        for price, size in asks:
            rows.append({"ts_ms": q.ts_ms, "venue": q.venue, "symbol": q.symbol,
                         "side": "ask", "price": f"{price:.10f}", "size": f"{size:.6f}"})
    return rows

# --------------- commands -----------------

async def cmd_run_live(cfg: ProjectConfig) -> None:
    engine = StableArbEngine(cfg)
    await engine.init_adapters()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, engine.stop)
        except NotImplementedError:
            pass  

    log("Starting live scan (paper)")
    last_record_ms = 0
    while not engine._stop:
        quotes, opps = await engine.scan_once()
        for o in opps:
            pnl = o.sell_revenue_usd - o.buy_cost_usd
            log(
                f"OPP {o.symbol}: BUY {o.buy_venue}@{o.buy_vwap:.6f} -> "
                f"SELL {o.sell_venue}@{o.sell_vwap:.6f} | size={o.size_base:.2f} | "
                f"edge={o.edge_bps:.2f} bps | pnl=${pnl:.4f}"
            )
        if cfg.log_csv:
            nowms = ts_ms_now()
            if nowms - last_record_ms >= 1000:
                rows = quotes_to_rows(quotes, max_levels=30)
                write_snapshot_rows(cfg.log_csv, rows)
                last_record_ms = nowms
        await asyncio.sleep(max(0.05, cfg.poll_interval_ms / 1000.0))
    log("Stopped.")

async def cmd_record(cfg: ProjectConfig, outfile: str, duration_sec: int) -> None:
    engine = StableArbEngine(cfg)
    await engine.init_adapters()
    log(f"Recording snapshots to {outfile} for {duration_sec}s…")
    start = ts_ms_now()
    while (ts_ms_now() - start) < duration_sec * 1000:
        quotes, _ = await engine.scan_once()
        rows = quotes_to_rows(quotes, max_levels=30)
        write_snapshot_rows(outfile, rows)
        await asyncio.sleep(max(0.05, cfg.poll_interval_ms / 1000.0))
    log("Recording complete.")

async def cmd_replay(cfg: ProjectConfig, infile: str) -> None:
    """
    Replay using time-bucketing so cross-venue books align within a window,
    mirroring live TTL behavior. Also supports optional partial-fill logic.
    """
    log(f"Replaying from {infile} …")
    df = pd.read_csv(infile, dtype={"venue": str, "symbol": str, "side": str})

    for col in CSV_FIELDS:
        if col not in df.columns:
            raise RuntimeError(f"CSV missing required column: {col}")

    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    df["ts_ms"] = df["ts_ms"].astype(np.int64)

    fee_map = {v.name: float(v.taker_fee_bps) for v in cfg.venues}
    warned_missing: set[str] = set()

    # ------- key change: bucket timestamps so venues align -------
    # Choose a bucket <= TTL to cluster quotes that would be considered "fresh" together.
    bucket_ms = int(cfg.risk.price_ttl_ms)
    df["bucket"] = (df["ts_ms"] // bucket_ms) * bucket_ms

    # optional change: allow partial fills in replay to inspect "latent" edge even when top-N depth is insufficient. set True to enable
    ALLOW_PARTIAL_FILL = True

    total_opps = 0
    total_pnl = 0.0

    for bucket, g in df.groupby("bucket"):
        quotes: List[Quote] = []

        for (venue, symbol), gg in g.groupby(["venue", "symbol"]):
            # pick the latest timestamp inside the bucket for this venue/symbol
            latest_ts = gg["ts_ms"].max()
            snap = gg[gg["ts_ms"] == latest_ts]

            side_bid = snap[snap["side"] == "bid"][["price", "size"]]
            side_ask = snap[snap["side"] == "ask"][["price", "size"]]

            side_bid = side_bid[(side_bid["size"] > 0) & np.isfinite(side_bid["price"])]
            side_ask = side_ask[(side_ask["size"] > 0) & np.isfinite(side_ask["price"])]

            bids = list(zip(side_bid["price"].tolist(), side_bid["size"].tolist()))
            asks = list(zip(side_ask["price"].tolist(), side_ask["size"].tolist()))
            if not bids and not asks:
                continue

            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])
            quotes.append(Quote(venue=venue, symbol=symbol, ts_ms=int(latest_ts),
                                bid_levels=bids, ask_levels=asks))

        for i in range(len(quotes)):
            for j in range(len(quotes)):
                if i == j:
                    continue
                qi, qj = quotes[i], quotes[j]
                if qi.symbol != qj.symbol:
                    continue

                target_size = min(cfg.base_units, cfg.risk.max_trade_units)

                buy = vwap_price(qi.ask_levels, target_size)    # buy at asks
                sell = vwap_price(qj.bid_levels, target_size)   # sell to bids

                if buy.vwap is None or sell.vwap is None:
                    if not ALLOW_PARTIAL_FILL:
                        continue
                    eff_size = min(buy.filled if buy.filled else 0.0,
                                   sell.filled if sell.filled else 0.0)
                    if eff_size <= 0:
                        continue
                    buy = vwap_price(qi.ask_levels, eff_size)
                    sell = vwap_price(qj.bid_levels, eff_size)
                    if buy.vwap is None or sell.vwap is None:
                        continue
                    size = eff_size
                else:
                    size = target_size

                buy_fee = fee_map.get(qi.venue)
                if buy_fee is None:
                    if qi.venue not in warned_missing:
                        log(f"WARN: venue '{qi.venue}' not in config; assuming 0 bps for replay.")
                        warned_missing.add(qi.venue)
                    buy_fee = 0.0

                sell_fee = fee_map.get(qj.venue)
                if sell_fee is None:
                    if qj.venue not in warned_missing:
                        log(f"WARN: venue '{qj.venue}' not in config; assuming 0 bps for replay.")
                        warned_missing.add(qj.venue)
                    sell_fee = 0.0

                buy_cost = size * buy.vwap * taker_fee_multiplier(buy_fee, "buy")
                sell_rev = size * sell.vwap * taker_fee_multiplier(sell_fee, "sell")
                e = edge_bps(buy_cost, sell_rev)

                if e >= cfg.risk.edge_entry_bps:
                    total_opps += 1
                    total_pnl += (sell_rev - buy_cost)

    log(f"Replay complete. Opportunities: {total_opps}, Aggregated paper PnL: ${total_pnl:.4f}")


# --------------- config template and dependency helper -----------------


CONFIG_TEMPLATE = """
base_units: 1000.0
poll_interval_ms: 750
log_csv: snapshots/snapshots.csv

venues:
  - name: binance
    symbol: "USDC/USDT"   
    taker_fee_bps: 1.0
    depth_levels: 20
    rate_limit_ms: 200

  - name: kraken
    symbol: "USDT/USD"
    taker_fee_bps: 26.0
    depth_levels: 20
    rate_limit_ms: 300

dex_pools:
  - name: SimPool-USDT-USDC
    base_token: USDT
    quote_token: USDC
    reserve_base: 5000000.0
    reserve_quote: 5000000.0
    lp_fee_bps: 3.0
    gas_usd: 0.2

risk:
  edge_entry_bps: 8.0
  edge_exit_bps: 2.0
  max_abs_depeg_bps: 150.0
  depeg_window_sec: 60
  price_ttl_ms: 2500
  max_trade_units: 10000.0
""".strip() + "\n"

REQUIREMENTS_TXT = """
ccxt
pandas
numpy
PyYAML
""".strip()

# --------------- cli setup -----------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stablecoin Basis / De-Peg Monitor (paper trading)")
    sub = p.add_subparsers(dest="cmd", required=False)

    p.add_argument("--init-config", metavar="PATH", help="Write a sample config.yaml to PATH and exit")
    p.add_argument("--print-requirements", action="store_true", help="Print pip requirements and exit")
    p.add_argument("--config", metavar="FILE", help="Path to config.yaml", default=None)

    sub.add_parser("run-live", help="Run live paper scanner")

    sp_rec = sub.add_parser("record", help="Record snapshots to CSV")
    sp_rec.add_argument("--outfile", required=True, help="CSV path for snapshots")
    sp_rec.add_argument("--duration", type=int, default=120, help="Recording duration in seconds (default 120)")

    sp_rep = sub.add_parser("replay", help="Replay backtest from CSV")
    sp_rep.add_argument("--infile", required=True, help="CSV snapshots to replay")

    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.print_requirements:
        print(REQUIREMENTS_TXT)
        return

    if args.init_config:
        with open(args.init_config, "w", encoding="utf-8") as f:
            f.write(CONFIG_TEMPLATE)
        print(f"Wrote sample config to {args.init_config}")
        return

    if not args.cmd:
        print("Nothing to do. Use --init-config, --print-requirements, or a subcommand (run-live, record, replay).")
        return

    if not args.config:
        print("--config is required for run-live/record/replay")
        return

    cfg = ProjectConfig.from_yaml(args.config)

    if args.cmd == "run-live":
        asyncio.run(cmd_run_live(cfg)); return
    if args.cmd == "record":
        asyncio.run(cmd_record(cfg, args.outfile, args.duration)); return
    if args.cmd == "replay":
        asyncio.run(cmd_replay(cfg, args.infile)); return

if __name__ == "__main__":
    main()
