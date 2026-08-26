#!/usr/bin/env python3
"""Hyperspeed trade — capture micro-spreads on volatile markets.

Designed for BTC strike markets where prices can move 5-10% within seconds.

Strategy:
1. Place BUY order at the current bid (maker, gets the spread)
2. Wait for fill (up to --entry-timeout seconds)
3. On fill: immediately place SELL order at bid + (cost * --profit-margin)
4. Wait for exit fill (up to --exit-timeout seconds)
5. If exit doesn't fill, cancel and report

This is a tight loop — designed to capture 5-10% profits in seconds-to-minutes.
NOT for holding positions. Use scalp_monitor.py for that.

Usage:
  python3 scripts/hyperspeed_trade.py --ticker KXBTCD-26AUG2521-T78699.99 \\
    --side yes --profit-margin 0.08 --entry-timeout 60 --exit-timeout 180
"""

import argparse
import os
import sys
import time
import uuid
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLI_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, CLI_DIR)

from cli import api, load_env, load_key, _build_v2_order_body, ApiError  # noqa: E402


def fetch_bid_ask(ticker: str, side: str) -> tuple:
    """Return (bid_cents, ask_cents) for the requested side."""
    r = api("GET", f"markets/{ticker}").get("market", {})
    if side == "yes":
        bid = float(r.get("yes_bid_dollars") or 0)
        ask = float(r.get("yes_ask_dollars") or 0)
    else:
        # For NO: NO bid/ask is the price for NO contracts directly
        bid = float(r.get("no_bid_dollars") or 0)
        ask = float(r.get("no_ask_dollars") or 0)
    return int(round(bid * 100)), int(round(ask * 100))


def place_v2_order(ticker: str, side: str, count: int, price_cents: int, action: str = "buy") -> dict:
    """Place V2 order. action='buy' or 'sell' (mine)."""
    if action == "buy":
        api_side = "bid"
        my_side_for_build = side
    else:  # sell
        api_side = "ask"
        # For NO: when I sell NO, V2 reverse maps to bid-side YES-buy
        # We use the same helper but it expects the user-facing side (yes/no)
        my_side_for_build = side
    body = _build_v2_order_body(
        ticker=ticker,
        side=my_side_for_build,
        count=count,
        price_cents=price_cents,
        action=action,
        client_order_id=str(uuid.uuid4()),
    )
    return api("POST", "portfolio/events/orders", body=body)


def cancel_order(order_id: str):
    try:
        api("DELETE", f"portfolio/events/orders/{order_id}")
    except ApiError as e:
        # Already filled or canceled — fine
        pass


def wait_for_fill(order_id: str, timeout_secs: float, poll_secs: float = 1.0) -> bool:
    """Poll order status until filled, canceled, or timeout."""
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            r = api("GET", f"portfolio/orders/{order_id}")
            o = r.get("order", {})
            remaining = float(o.get("remaining_count_fp") or 0)
            fill = float(o.get("fill_count_fp") or 0)
            if remaining == 0 and fill > 0:
                return True
            status = o.get("status", "")
            if status in ("canceled", "cancelled", "expired"):
                return False
        except ApiError:
            pass
        time.sleep(poll_secs)
    return False


def hyperspeed(
    ticker: str,
    side: str,
    profit_margin: float,
    entry_timeout: float,
    exit_timeout: float,
    count: int,
    dry_run: bool,
):
    """Execute one hyperspeed trade cycle."""
    print(f"[hyperspeed] starting on {ticker}", flush=True)
    print(f"  side={side} count={count} profit_margin={profit_margin*100:.1f}% entry={entry_timeout}s exit={exit_timeout}s", flush=True)

    # Step 1: fetch current bid
    bid_cents, ask_cents = fetch_bid_ask(ticker, side)
    if bid_cents <= 0 or ask_cents <= 0:
        print(f"  [err] no live book: bid={bid_cents}¢ ask={ask_cents}¢", flush=True)
        return None

    entry_cents = bid_cents  # buy at the bid (maker)
    target_exit_cents = int(round(entry_cents * (1 + profit_margin)))
    target_exit_cents = max(target_exit_cents, entry_cents + 1)  # at least +1¢
    target_exit_cents = min(target_exit_cents, 99)  # never above 99¢

    print(f"  bid={bid_cents}¢ ask={ask_cents}¢  entry_at={entry_cents}¢  exit_target={target_exit_cents}¢", flush=True)

    # Step 2: place entry order (buy at bid)
    if dry_run:
        print(f"  [dry-run] would BUY {count}x {side} @ {entry_cents}¢", flush=True)
        print(f"  [dry-run] would SELL {count}x {side} @ {target_exit_cents}¢ after fill", flush=True)
        return None

    print(f"  [step 1] placing BUY {count}x {side} @ {entry_cents}¢", flush=True)
    try:
        entry_order = place_v2_order(ticker, side, count, entry_cents, action="buy")
    except ApiError as e:
        print(f"  [err] entry order failed: {e}", flush=True)
        return None

    entry_order_id = entry_order.get("order_id")
    print(f"  [step 2] waiting for entry fill (timeout {entry_timeout}s)...", flush=True)

    # Step 3: wait for entry fill
    if not wait_for_fill(entry_order_id, entry_timeout):
        print(f"  [timeout] entry order not filled, canceling", flush=True)
        cancel_order(entry_order_id)
        return None

    actual_fill_price = int(round(float(entry_order.get("yes_price_dollars") or entry_order.get("no_price_dollars") or entry_cents/100) * 100))
    print(f"  [filled] entry at {actual_fill_price}¢", flush=True)

    # Step 4: place exit order (sell at target)
    print(f"  [step 3] placing SELL {count}x {side} @ {target_exit_cents}¢", flush=True)
    try:
        exit_order = place_v2_order(ticker, side, count, target_exit_cents, action="sell")
    except ApiError as e:
        print(f"  [err] exit order failed: {e}", flush=True)
        return None

    exit_order_id = exit_order.get("order_id")
    print(f"  [step 4] waiting for exit fill (timeout {exit_timeout}s)...", flush=True)

    # Step 5: wait for exit fill
    if not wait_for_fill(exit_order_id, exit_timeout):
        print(f"  [timeout] exit order not filled, canceling (position still open)", flush=True)
        cancel_order(exit_order_id)
        return {"entry": actual_fill_price, "exit": None, "status": "timeout"}

    actual_exit_price = int(round(float(exit_order.get("yes_price_dollars") or exit_order.get("no_price_dollars") or target_exit_cents/100) * 100))
    profit_cents = actual_exit_price - actual_fill_price
    profit_pct = (profit_cents / actual_fill_price * 100) if actual_fill_price > 0 else 0
    print(f"  [done] exited at {actual_exit_price}¢  profit={profit_cents}¢/ct ({profit_pct:.1f}%)  total=${profit_cents * count / 100:.2f}", flush=True)
    return {"entry": actual_fill_price, "exit": actual_exit_price, "status": "filled", "profit_cents": profit_cents}


def main():
    p = argparse.ArgumentParser(description="Hyperspeed trade — capture micro-spreads on volatile markets")
    p.add_argument("--ticker", required=True, help="Market ticker")
    p.add_argument("--side", choices=["yes", "no"], default="yes", help="Side to buy (default yes)")
    p.add_argument("--count", type=int, default=1, help="Number of contracts (default 1)")
    p.add_argument("--profit-margin", type=float, default=0.08, help="Profit margin as decimal (0.08 = 8%%, default 8%%)")
    p.add_argument("--entry-timeout", type=float, default=60.0, help="Seconds to wait for entry fill (default 60)")
    p.add_argument("--exit-timeout", type=float, default=180.0, help="Seconds to wait for exit fill (default 180 = 3min)")
    p.add_argument("--dry-run", action="store_true", help="Print plan without placing orders")
    p.add_argument("--loop", action="store_true", help="Keep running new hyperspeed trades (Ctrl-C to stop)")
    p.add_argument("--loop-interval", type=float, default=5.0, help="Seconds between loop iterations (default 5)")
    args = p.parse_args()

    load_env()
    load_key()

    if args.loop:
        iteration = 0
        try:
            while True:
                iteration += 1
                print(f"\n[hyperspeed loop] iteration {iteration} @ {datetime.now(timezone.utc).strftime('%H:%M:%S')}", flush=True)
                hyperspeed(
                    ticker=args.ticker,
                    side=args.side,
                    profit_margin=args.profit_margin,
                    entry_timeout=args.entry_timeout,
                    exit_timeout=args.exit_timeout,
                    count=args.count,
                    dry_run=args.dry_run,
                )
                if not args.dry_run:
                    time.sleep(args.loop_interval)
        except KeyboardInterrupt:
            print("\nstopped.")
            return
    else:
        hyperspeed(
            ticker=args.ticker,
            side=args.side,
            profit_margin=args.profit_margin,
            entry_timeout=args.entry_timeout,
            exit_timeout=args.exit_timeout,
            count=args.count,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
