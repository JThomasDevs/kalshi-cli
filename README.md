# kalshi-cli

A fast, interactive CLI for trading prediction markets on [Kalshi](https://kalshi.com).

Built with Python, [Typer](https://typer.tiangolo.com/), and [Rich](https://rich.readthedocs.io/) for a clean terminal experience.

## Features

- **Smart search** — find markets by keyword, category, tag, or ticker with multi-strategy matching
- **Interactive drill-down** — browse series → markets → market detail without re-running commands
- **Smart outcome labels** — shows real outcome names (e.g. "Norway", "Trump", "$79,500 or above") instead of raw tickers
- **Up/down markets** — Yes/No columns become "Up ↑" / "Down ↓" for price direction markets
- **Enriched positions** — fetches market details to show titles, outcome picks, return %, and expiry
- **Live orderbook** — view bid/ask depth for any market
- **Trade with confirmation** — buy/sell with cost summaries and confirmation prompts
- **Expiry sorting** — prioritize markets expiring soonest with `-e`
- **Min-odds filtering** — hide dead markets where either side is below a threshold with `-m`
- **Human-readable output** — prices as `$0.68 (68%)`, expiry as `8h 35m` or `Fri 04:00PM`

## Installation

```bash
npm install -g kalshi-cli
```

This installs the `kalshi` command globally. The postinstall script automatically clones the repo, sets up a Python virtual environment, and installs dependencies.

**Requirements:** Python 3.10+ and Node.js/npm.

### API Credentials

1. Generate API credentials at [kalshi.com/api](https://kalshi.com/api)
2. Place your RSA private key at `~/.kalshi/private_key.pem`
3. Set your access key in `~/.kalshi/.env`:

```
KALSHI_ACCESS_KEY=your_access_key_id
```

Or run `kalshi setup-shell` to export it to your shell config automatically.

Template files are created at `~/.kalshi/` during installation if they don't exist.

## Usage

### Browse Markets

```bash
# List open markets
kalshi markets
kalshi markets -l 50 --status settled

# Browse active series (interactive)
kalshi series
kalshi series soccer
kalshi series -e              # sort by soonest expiry
kalshi series --all           # include inactive series
```

### Search

```bash
# Search by keyword, category, or ticker
kalshi search soccer
kalshi search "Super Bowl"
kalshi search hockey -e           # sort by soonest expiry
kalshi search politics -m 5       # hide markets where either side < 5%
kalshi search KXWO-GOLD-26        # direct ticker lookup
kalshi search soccer -e -m 2 -l 20
```

Search uses a multi-strategy approach:
1. **Direct ticker** — tries as market, event, or series ticker
2. **Series matching** — searches series titles, categories, and tags dynamically
3. **Market title fallback** — searches open market titles if nothing else matches

When many series match, an interactive numbered list is displayed — enter a number to drill into that series. Markets are also numbered for a second-level drill-down to market detail.

### Market Details

```bash
# Detailed view of a single market
kalshi detail KXWO-GOLD-26-NOR

# View orderbook depth
kalshi orderbook KXWO-GOLD-26-NOR
```

### Portfolio

```bash
kalshi balance            # account balance
kalshi positions          # current positions with titles, return %, expiry
kalshi positions -e       # sort positions by soonest expiry
kalshi orders             # open/resting orders
```

### Trading

```bash
# Buy 10 YES contracts at 68¢
kalshi buy KXSB-26 10 68

# Buy NO contracts
kalshi buy KXWO-GOLD-26-NOR 5 32 --side no

# Sell
kalshi sell KXWO-GOLD-26-NOR 5 40 --side no

# Skip confirmation prompt
kalshi buy KXSB-26 10 68 --force

# Cancel an order
kalshi cancel <order-id>
```

### Flags Reference

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--limit` | `-l` | Max results to display | 10 (search), 20 (markets) |
| `--min-odds` | `-m` | Hide markets where either side < N% | 0.5 |
| `--expiring` | `-e` | Sort by soonest expiry, show expiry column | off |
| `--side` | `-s` | Contract side: `yes` or `no` | yes |
| `--force` | `-f` | Skip trade confirmation | off |
| `--status` | `-s` | Filter markets by status | open |
| `--all` | `-a` | Include inactive series (series cmd) | off |

## Market Display

Market tables auto-detect the type of market:

- **Multi-outcome** (Olympics, IPO, mentions): Outcome column shows the pick (e.g. "Norway", "Trump")
- **Range** (Bitcoin price): Outcome shows the range (e.g. "$79,500 or above")
- **Up/down** (15-min price): Columns labeled "Up ↑" / "Down ↓" with target price
- **Binary** (Will X happen?): Standard Yes/No columns
- **Parlay**: Legs shown on separate lines

## Notes

- Prices are in **cents**: `68` = $0.68 = 68% implied probability
- Event tickers start with `KX` (e.g. `KXWO-GOLD-26`); market tickers have additional segments (e.g. `KXWO-GOLD-26-NOR`)
- Common search aliases are expanded automatically (e.g. "nfl" also matches "football")

## License

MIT
