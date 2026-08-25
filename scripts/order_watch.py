#!/usr/bin/env python3
"""Watch resting orders and launch scalp_monitor when one fills.

Polls `kalshi orders` and detects when a resting order transitions to
filled/executed. When that happens, spawns scalp_monitor.py as a subprocess
to manage the resulting position.

Usage:
  python3 scripts/order_watch.py \
    --ticker KXBTCD-26AUG2514-T78899.99 \
    --strike 78899 \
    --target-bid 0.90 \
    --panic-cushion 30 \
    --poll-secs 5
"""

import argparse
import os
import subprocess
import sys
import time

# Make cli.py importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLI_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, CLI_DIR)

import cli  # noqa: E402
from cli import api, load_env, load_key  # noqa: E402


def fetch_orders() -> list:
    """Get all resting orders from Kalshi."""
    r = api("GET", "portfolio/orders?status=resting")
    return r.get("orders", [])


def watch_and_spawn(
    target_ticker: str,
    strike: float,
    target_bid: float,
    panic_cushion: float,
    poll_secs: float,
    close_grace_secs: float,
    count: int,
    side: str,
    scalp_script: str,
):
    """Poll until target_ticker shows in resting orders, then spawn scalp_monitor."""
    print(f"[order_watch] watching {target_ticker} for fill", flush=True)
    while True:
        orders = fetch_orders()
        # Check if our order is still resting
        still_resting = any(
            o.get("ticker", "").upper() == target_ticker.upper() and o.get("status") == "resting"
            for o in orders
        )

        if still_resting:
            # Print compact status
            my_order = next(o for o in orders if o.get("ticker", "").upper() == target_ticker.upper())
            ts = time.strftime("%H:%M:%S")
            print(
                f"  [{ts}] still resting on {target_ticker} "
                f"(order_id={my_order.get('order_id', '?')[:8]}...)",
                flush=True,
            )
            time.sleep(poll_secs)
            continue

        # Order not in resting list → either filled or cancelled
        # Check if it was filled (look at all order statuses)
        all_orders_resp = api("GET", "portfolio/orders?status=executed")
        executed = all_orders_resp.get("orders", [])
        filled = any(o.get("ticker", "").upper() == target_ticker.upper() for o in executed)

        if filled:
            print(f"[order_watch] {target_ticker} FILLED — launching scalp_monitor", flush=True)
            # Spawn scalp_monitor as detached subprocess
            cmd = [
                sys.executable,
                scalp_script,
                "--ticker", target_ticker,
                "--strike", str(strike),
                "--target-bid", str(target_bid),
                "--panic-cushion", str(panic_cushion),
                "--poll-secs", str(poll_secs),
                "--close-grace-secs", str(close_grace_secs),
                "--count", str(count),
                "--side", side,
            ]
            print(f"  [spawn] {' '.join(cmd)}", flush=True)
            # Redirect scalp_monitor stdout/stderr to a log file so the user can see progress
            log_path = f"/tmp/scalp_{target_ticker}.log"
            log_file = open(log_path, "w")
            subprocess.Popen(cmd, cwd=CLI_DIR, stdout=log_file, stderr=subprocess.STDOUT)
            print(f"  [log] output: {log_path}", flush=True)
            return

        # Order not resting, not executed → must have been cancelled externally
        print(f"[order_watch] {target_ticker} no longer resting and not executed — exiting", flush=True)
        return


def main():
    p = argparse.ArgumentParser(description="Watch a resting order and spawn scalp_monitor when it fills")
    p.add_argument("--ticker", required=True, help="Ticker to watch for fill")
    p.add_argument("--strike", type=float, required=True, help="Strike price (for the panic-cushion calculation)")
    p.add_argument("--target-bid", type=float, default=0.90)
    p.add_argument("--panic-cushion", type=float, default=30.0)
    p.add_argument("--poll-secs", type=float, default=5.0)
    p.add_argument("--close-grace-secs", type=float, default=60.0)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--side", choices=["yes", "no"], default="yes")
    args = p.parse_args()

    load_env()
    load_key()

    watch_and_spawn(
        target_ticker=args.ticker,
        strike=args.strike,
        target_bid=args.target_bid,
        panic_cushion=args.panic_cushion,
        poll_secs=args.poll_secs,
        close_grace_secs=args.close_grace_secs,
        count=args.count,
        side=args.side,
        scalp_script=os.path.join(SCRIPT_DIR, "scalp_monitor.py"),
    )


if __name__ == "__main__":
    main()