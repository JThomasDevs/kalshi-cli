# scripts/ — operational scripts for kalshi-cli

This directory holds standalone operational tools that wrap the CLI for
specific workflows.

## scalp_monitor.py

Watches an open YES position and auto-sells when:

1. **Profit target hit** — bid ≥ `--target-bid` (default 90¢)
2. **Panic** — asset spot drops within `--panic-cushion` of the strike
   (default $20 for BTC strike markets)

Designed for **BTC strike markets** but the strike/cushion are configurable.
Assumes V2 endpoint migration (1.2.76+).

```
python3 scripts/scalp_monitor.py \
  --ticker KXBTCD-26AUG2513-T79099.99 \
  --strike 79100 \
  --target-bid 0.95 \
  --panic-cushion 30 \
  --poll-secs 5 \
  --dry-run
```

Add `--dry-run` to print decisions without placing orders.

## Adding new scripts

Keep scripts thin — they should orchestrate `cli.py` (which does the heavy
lifting + signing + error handling). Don't duplicate API client code.
