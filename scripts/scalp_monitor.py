#!/usr/bin/env python3
"""Scalp monitor for open Kalshi positions.

Watches an open YES position and auto-sells when the contract is sufficiently
in-the-money, or panics out if BTC moves against the strike.

Strategy (for BTC strike markets — adjust per asset):
  - If YES bid >= TARGET_PROFIT_BID (default 90c): sell to lock profit
  - If asset spot drops within PANIC_CUSHION_USD (default $20) of strike: panic sell
  - If time-to-close < CLOSE_GRACE_SECS (default 60s) and still ITM: hold to settle
  - Otherwise: keep polling

Assumes V2 endpoint migration (1.2.76+). Uses cli.py as a library.

Usage:
  python3 scripts/scalp_monitor.py --ticker KXBTCD-26AUG2513-T79099.99 \\
    --strike 79100 --target-bid 0.90 --panic-cushion 20 --poll-secs 5
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

# Make cli.py importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLI_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, CLI_DIR)

import cli  # noqa: E402

# Reuse the order helpers from cli.py
from cli import api, load_env, load_key, _build_v2_order_body, ApiError  # noqa: E402


def fetch_btc_spot() -> float | None:
    """Fetch BTC spot from multiple sources with fallback. Returns None on total failure."""
    import urllib.request
    import json

    sources = [
        # CoinGecko (free, rate-limited)
        ("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
         lambda data: float(data["bitcoin"]["usd"])),
        # Coinbase (no rate limit for spot)
        ("https://api.coinbase.com/v2/prices/BTC-USD/spot",
         lambda data: float(data["data"]["amount"])),
        # Kraken (no rate limit for public ticker)
        ("https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
         lambda data: float(list(data["result"].values())[0]["c"][0])),
    ]

    for url, parser in sources:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kalshi-scalp-monitor/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            price = parser(data)
            return float(price)
        except Exception as e:
            print(f"  [warn] {url.split('/')[2]} spot fetch failed: {e}", flush=True)
            continue
    return None


def fetch_market_state(ticker: str) -> dict:
    """Pull detail + best bid/ask from the orderbook."""
    detail = api("GET", f"markets/{ticker}").get("market", {})
    book_resp = api("GET", f"markets/{ticker}/orderbook")
    book = book_resp.get("orderbook", {}).get("orderbook_fp", {})

    # The orderbook's yes_dollars/no_dollars levels can be inverted depending on
    # version. Prefer the detail fields when present (they're normalized).
    return {
        "ticker": ticker,
        "status": detail.get("status"),
        "yes_bid": float(detail.get("yes_bid_dollars") or 0),
        "yes_ask": float(detail.get("yes_ask_dollars") or 0),
        "no_bid": float(detail.get("no_bid_dollars") or 0),
        "no_ask": float(detail.get("no_ask_dollars") or 0),
        "last_price": float(detail.get("last_price_dollars") or 0),
        "close_time": detail.get("close_time"),
        "expected_expiration_time": detail.get("expected_expiration_time"),
        "raw_book": book,
    }


def time_to_close(state: dict) -> float:
    """Seconds until close_time. Negative = past close."""
    ct = state.get("close_time")
    if not ct:
        return float("inf")
    try:
        dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
    except Exception:
        return float("inf")
    return (dt - datetime.now(timezone.utc)).total_seconds()


def place_v2_sell(ticker: str, count: int, price_cents: int, side: str = "yes") -> dict:
    """Place a V2 sell order. Returns the API response."""
    body = _build_v2_order_body(
        ticker=ticker,
        side=side,
        count=count,
        price_cents=price_cents,
        action="sell",
        client_order_id=str(__import__("uuid").uuid4()),
    )
    return api("POST", "portfolio/events/orders", body=body)


def monitor(
    ticker: str,
    strike: float,
    target_bid: float,
    panic_cushion: float,
    poll_secs: float,
    close_grace_secs: float,
    count: int,
    side: str,
    dry_run: bool,
    no_panic: bool = False,
):
    """Main monitor loop. Returns when position is sold or expired."""
    print(f"[scalp_monitor] starting on {ticker}", flush=True)
    print(f"  strike={strike} target_bid={target_bid:.2f} panic_cushion=${panic_cushion}{'  [DISABLED]' if no_panic else ''}", flush=True)
    print(f"  count={count} side={side} dry_run={dry_run}", flush=True)

    while True:
        try:
            state = fetch_market_state(ticker)
        except ApiError as e:
            print(f"  [err] market fetch failed: {e}", flush=True)
            time.sleep(poll_secs)
            continue

        secs_to_close = time_to_close(state)
        spot = fetch_btc_spot()
        cushion = (spot - strike) if (spot is not None) else None

        # Pick the bid/ask for our held side
        if side == "yes":
            our_bid = state["yes_bid"]
            our_ask = state["yes_ask"]
        else:
            our_bid = state["no_bid"]
            our_ask = state["no_ask"]

        print(
            f"  [{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
            f"{side.upper()} bid={our_bid:.2f} ask={our_ask:.2f} "
            f"spot={spot} cushion={cushion} "
            f"close_in={secs_to_close:.0f}s status={state['status']}",
            flush=True,
        )

        # Decision logic
        should_sell = False
        reason = ""

        # 1. Profit target hit
        if our_bid >= target_bid:
            should_sell = True
            reason = f"target_bid reached ({our_bid:.2f} >= {target_bid:.2f})"

        # 2. Panic: spot too close to (or past) strike in the wrong direction
        # Only fires when --no-panic is NOT set
        elif not no_panic and cushion is not None:
            distance_to_strike = abs(cushion)
            if side == "yes" and cushion < panic_cushion:
                should_sell = True
                reason = f"panic: spot ${cushion:.0f} above strike (< ${panic_cushion})"
            elif side == "no" and distance_to_strike < panic_cushion:
                should_sell = True
                reason = f"panic: spot ${cushion:+.0f} from strike (|${distance_to_strike:.0f}| < ${panic_cushion})"

        # 3. Past close: stop trying, position will auto-handle
        if secs_to_close < 0:
            print(f"[scalp_monitor] past close_time — exiting monitor", flush=True)
            return

        # 4. Within close_grace: don't panic sell, let it settle
        if secs_to_close < close_grace_secs and not should_sell:
            print(f"  [hold] within close_grace ({secs_to_close:.0f}s < {close_grace_secs}s), letting position settle", flush=True)
            time.sleep(poll_secs)
            continue

        if should_sell:
            print(f"[scalp_monitor] SELL triggered: {reason}", flush=True)
            if dry_run:
                print(f"  [dry-run] would sell {count}x {side} @ market (bid {our_bid:.2f})", flush=True)
                return
            try:
                # Re-fetch the latest bid just before placing the sell, to avoid
                # the race where the orderbook moves between detection and
                # execution. Place at bid-1¢ to ensure we CROSS the book and
                # become a taker — a limit at the bid price can sit unfilled if
                # the bid drops before our order lands.
                latest = fetch_market_state(ticker)
                latest_bid = latest["yes_bid"] if side == "yes" else latest["no_bid"]
                sell_price_cents = max(1, int(round(latest_bid * 100)) - 1)
                print(f"  [exec] placing sell {count}x {side} @ {sell_price_cents}¢ (current bid {latest_bid:.2f})", flush=True)
                res = place_v2_sell(ticker, count, sell_price_cents, side=side)
                print(f"  [ok] filled: {res}", flush=True)
                return
            except ApiError as e:
                print(f"  [err] sell failed: {e}", flush=True)
                time.sleep(poll_secs)
                continue

        time.sleep(poll_secs)


def main():
    p = argparse.ArgumentParser(description="Scalp monitor for open Kalshi positions")
    p.add_argument("--ticker", required=True, help="Market ticker to watch")
    p.add_argument("--strike", type=float, required=True, help="Strike price (e.g. 79100 for BTC strike markets)")
    p.add_argument("--target-bid", type=float, default=0.90, help="Sell when bid >= this (default 0.90)")
    p.add_argument("--panic-cushion", type=float, default=20.0, help="Panic sell when spot is within $X of strike (default 20)")
    p.add_argument("--poll-secs", type=float, default=5.0, help="Seconds between polls (default 5)")
    p.add_argument("--close-grace-secs", type=float, default=60.0, help="Stop trading within this many seconds of close_time (default 60)")
    p.add_argument("--count", type=int, default=1, help="Number of contracts to monitor/sell (default 1)")
    p.add_argument("--side", choices=["yes", "no"], default="yes", help="Side of the position (default yes)")
    p.add_argument("--dry-run", action="store_true", help="Print what would happen without placing orders")
    p.add_argument("--no-panic", action="store_true", help="Disable panic-cushion exit (only target-exit or hold to settlement)")
    args = p.parse_args()

    # Load Kalshi credentials
    load_env()
    load_key()

    monitor(
        ticker=args.ticker,
        strike=args.strike,
        target_bid=args.target_bid,
        panic_cushion=args.panic_cushion,
        poll_secs=args.poll_secs,
        close_grace_secs=args.close_grace_secs,
        count=args.count,
        side=args.side,
        dry_run=args.dry_run,
        no_panic=args.no_panic,
    )


if __name__ == "__main__":
    main()