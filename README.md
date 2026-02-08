# kalshi-cli

A fast, interactive CLI for trading prediction markets on [Kalshi](https://kalshi.com).

Built with Python, [Typer](https://typer.tiangolo.com/), and [Rich](https://rich.readthedocs.io/) for a clean terminal experience.

## Features

- **Smart search** — find markets by keyword, category, tag, or ticker with multi-strategy matching
- **Interactive drill-down** — browse series as numbered lists, pick one to load its markets inline
- **Live orderbook** — view bid/ask depth for any market
- **Trade with confirmation** — buy/sell with cost summaries and confirmation prompts
- **Expiry sorting** — prioritize markets expiring soonest with `-e`
- **Min-odds filtering** — hide dead markets where either side is below a threshold with `-m`
- **Human-readable output** — prices as `$0.68 (68%)`, expiry as `8h 35m` or `Fri 04:00PM`

## Installation

### pipx (recommended)

```bash
# Install pipx if you don't have it
python3 -m pip install --user pipx
pipx ensurepath

# Install kalshi-cli
pipx install git+https://github.com/JThomasDevs/kalshi-cli.git
```

The `kalshi` command will be available globally.

### Manual Setup

```bash
# Clone the repo
git clone https://github.com/JThomasDevs/kalshi-cli.git
cd kalshi-cli

# Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### API Credentials

1. Generate API credentials at [kalshi.com/api](https://kalshi.com/api)
2. Save your RSA private key:

```bash
mkdir -p ~/.kalshi
# Place your private key at ~/.kalshi/private_key.pem
```

3. Set your access key in `~/.kalshi/.env`:

```
KALSHI_ACCESS_KEY=your_access_key_id
```

Or run `python cli.py setup-shell` to export it to your shell config automatically.

### Shell Alias (optional)

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
alias kalshi="/path/to/kalshi-cli/.venv/bin/python3 /path/to/kalshi-cli/cli.py"
```

Then use `kalshi` from anywhere.

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

When many series match, an interactive numbered list is displayed — enter a number to drill into that series.

### Market Details

```bash
# Detailed view of a single market
kalshi detail KXWO-GOLD-26-NOR

# View orderbook depth
kalshi orderbook KXWO-GOLD-26-NOR
```

### Portfolio

```bash
kalshi balance        # account balance
kalshi positions      # current positions
kalshi orders         # open/resting orders
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

## Notes

- Prices are in **cents**: `68` = $0.68 = 68% implied probability
- Event tickers start with `KX` (e.g. `KXWO-GOLD-26`); market tickers have additional segments (e.g. `KXWO-GOLD-26-NOR`)
- Common search aliases are expanded automatically (e.g. "nfl" also matches "football")

## License

MIT
