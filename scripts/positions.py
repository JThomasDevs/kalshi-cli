#!/usr/bin/env python3
"""Unified positions dashboard for Kalshi CLI.

Shows every active position with:
  - Current bid/ask for the contract you hold
  - Unrealized P&L vs cost basis
  - Time to close
  - Active scalp_monitor (if any) for that position
  - Recent scalp_monitor log tail

Also surfaces BTC spot for context. Designed for repeated polling in a
separate terminal — print, sleep, repeat.

Usage:
  python3 scripts/positions.py --watch 30    # refresh every 30s
  python3 scripts/positions.py               # one-shot
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLI_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, CLI_DIR)

from cli import api, load_env, load_key  # noqa: E402


def fetch_btc_spot() -> float | None:
    """Same fallback chain as scalp_monitor."""
    import json
    import urllib.request
    sources = [
        ("https://api.coinbase.com/v2/prices/BTC-USD/spot",
         lambda d: float(d["data"]["amount"])),
        ("https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
         lambda d: float(list(d["result"].values())[0]["c"][0])),
        ("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
         lambda d: float(d["bitcoin"]["usd"])),
    ]
    for url, parser in sources:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kalshi-positions/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            return parser(data)
        except Exception:
            continue
    return None


def find_monitor_for_ticker(ticker: str) -> tuple[int | None, str | None]:
    """Find scalp_monitor process for a given ticker. Returns (pid, log_path)."""
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,command"], text=True)
    except Exception:
        return None, None
    for line in out.splitlines():
        if "scalp_monitor.py" not in line:
            continue
        m = re.search(r"--ticker\s+(\S+)", line)
        if m and m.group(1).upper() == ticker.upper():
            pid_m = re.match(r"\s*(\d+)", line)
            pid = int(pid_m.group(1)) if pid_m else None
            # The monitor writes its log to the most recent /tmp/scalp_*.log or /tmp/scalp<idx>.log
            # For now, prefer the per-ticker log
            log_path = f"/tmp/scalp_{ticker}.log"
            if not os.path.exists(log_path):
                # Fall back to scalp3.log, watch4.log, etc — find any log containing this ticker
                for f in os.listdir("/tmp"):
                    if f.startswith(("scalp", "watch")) and f.endswith(".log"):
                        full = f"/tmp/{f}"
                        try:
                            with open(full) as fp:
                                if ticker.upper() in fp.read():
                                    log_path = full
                                    break
                        except Exception:
                            continue
            return pid, log_path
    return None, None


def tail_log(path: str | None, lines: int = 3) -> str:
    if not path or not os.path.exists(path):
        return "  (no log)"
    try:
        with open(path) as f:
            content = f.readlines()
        return "".join(content[-lines:])
    except Exception as e:
        return f"  (log read error: {e})"


def show_positions(spot: float | None):
    """Print the dashboard."""
    # Get event positions (these are the held contracts)
    r = api("GET", "portfolio/positions")
    event_positions = r.get("event_positions", [])

    # Get open orders (should be 0 if all resting bids filled)
    orders = api("GET", "portfolio/orders?status=resting").get("orders", [])

    # Balance — prefer balance_dollars (string in dollars), fall back to balance (integer cents)
    bal_resp = api("GET", "portfolio/balance")
    if "balance_dollars" in bal_resp:
        bal = float(bal_resp["balance_dollars"])
    else:
        bal = float(bal_resp.get("balance", 0)) / 100

    now = datetime.now(timezone.utc)
    print(f"\n{'='*78}")
    print(f"  KALSHI POSITIONS DASHBOARD — {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Balance: ${bal:,.2f}    BTC spot: {f'${spot:,.2f}' if spot else 'n/a'}")
    print(f"  Open resting orders: {len(orders)}")
    print(f"{'='*78}")

    # For each event with non-zero cost or shares, fetch market detail
    open_positions = []
    for ep in event_positions:
        cost = float(ep.get("total_cost_dollars") or 0)
        shares = float(ep.get("total_cost_shares_fp") or 0)
        realized = float(ep.get("realized_pnl_dollars") or 0)
        if cost == 0 and shares == 0:
            continue
        open_positions.append((ep, cost, shares, realized))

    if not open_positions:
        print("\n  No open positions.")
    else:
        # Sort: BTC events first, then alphabetical
        open_positions.sort(key=lambda x: (not x[0].get("event_ticker", "").startswith("KXBTC"), x[0].get("event_ticker", "")))

        for ep, cost, shares, realized in open_positions:
            event = ep.get("event_ticker", "?")
            tag = "  [BTC]" if "KXBTC" in event else "       "
            print(f"\n  {tag} Event: {event}   shares={shares:.2f}   cost=${cost:.4f}   realized=${realized:+.4f}")

            # Try to find the specific market within this event for current pricing
            # We'll look up markets via the trades-driven endpoint
            try:
                trades = api("GET", "markets/trades?limit=200").get("trades", [])
                seen = list({t.get("ticker", "") for t in trades if t.get("ticker", "").startswith(event)})[:5]
                if seen:
                    details = api("GET", f"markets?tickers={','.join(seen)}&status=open").get("markets", [])
                    for m in details:
                        tk = m.get("ticker", "")
                        exp = m.get("expected_expiration_time") or m.get("close_time") or ""
                        try:
                            dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                            secs_to_close = (dt - now).total_seconds()
                            if secs_to_close > 0:
                                if secs_to_close > 3600:
                                    when = f"{secs_to_close/3600:.1f}h"
                                else:
                                    when = f"{int(secs_to_close//60)}m{int(secs_to_close%60)}s"
                            else:
                                when = "EXPIRED"
                        except Exception:
                            when = "?"

                        yb = float(m.get("yes_bid_dollars") or 0)
                        ya = float(m.get("yes_ask_dollars") or 0)
                        nb = float(m.get("no_bid_dollars") or 0)
                        na = float(m.get("no_ask_dollars") or 0)

                        pid, log_path = find_monitor_for_ticker(tk)
                        mon_status = f"monitor pid={pid}" if pid else "no monitor"

                        print(f"\n    {tk}  [{when} to close]  [{mon_status}]")
                        print(f"      YES bid/ask: {yb:.2f}/{ya:.2f}    NO bid/ask: {nb:.2f}/{na:.2f}")

                        if log_path and pid:
                            print(f"      Last 3 monitor lines ({log_path}):")
                            for line in tail_log(log_path, 3).splitlines():
                                if line.strip():
                                    print(f"        {line.strip()[:100]}")
            except Exception as e:
                print(f"    (could not fetch markets for {event}: {e})")

    print(f"\n{'='*78}\n")


def main():
    p = argparse.ArgumentParser(description="Unified Kalshi positions dashboard")
    p.add_argument("--watch", type=int, default=0,
                   help="Refresh every N seconds (0 = one-shot, default)")
    args = p.parse_args()

    load_env()
    load_key()

    if args.watch > 0:
        try:
            while True:
                spot = fetch_btc_spot()
                show_positions(spot)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nstopped.")
            return
    else:
        spot = fetch_btc_spot()
        show_positions(spot)


if __name__ == "__main__":
    main()
