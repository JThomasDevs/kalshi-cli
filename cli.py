#!/usr/bin/env python3
"""Kalshi CLI - Prediction market trading tool"""

import os
import sys
import time
import base64

import requests
import typer
from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich import box
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

app = typer.Typer(help="Kalshi prediction market CLI")
BASE_URL = "https://api.elections.kalshi.com"
console = Console()


class ApiError(Exception):
    """Raised on non-success API responses (catchable, unlike typer.Exit)"""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")


# ── Auth & API ──────────────────────────────────────────────


def load_env():
    """Load credentials from ~/.kalshi/.env"""
    env_path = os.path.expanduser("~/.kalshi/.env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r") as f:
        for line in f:
            if line.startswith("export "):
                line = line[7:]
            if "=" in line:
                key, val = line.strip().split("=", 1)
                os.environ[key] = val.strip('"').strip("'")


def load_key():
    """Load RSA private key from env var or file"""
    raw = os.getenv("KALSHI_ACCESS_PRIVATE_KEY")
    if raw:
        try:
            return serialization.load_pem_private_key(raw.encode(), password=None)
        except Exception:
            pass
    for p in ["~/.kalshi/private_key.pem", "private_key.pem"]:
        expanded = os.path.expanduser(p)
        if os.path.exists(expanded):
            with open(expanded, "rb") as f:
                return serialization.load_pem_private_key(f.read(), password=None)
    console.print("[red]Error:[/red] No private key found. Place your RSA key at ~/.kalshi/private_key.pem")
    raise typer.Exit(1)


def sign_request(ts: str, method: str, path: str, key) -> str:
    """Create API request signature"""
    msg = ts + method + path
    sig = key.sign(
        msg.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode()


def api(method: str, endpoint: str, body: dict = None) -> dict:
    """Make an authenticated Kalshi API request"""
    load_env()
    key_id = os.getenv("KALSHI_ACCESS_KEY")
    if not key_id:
        console.print(
            "[red]Error:[/red] KALSHI_ACCESS_KEY not set. "
            "Run [bold]kalshi setup-shell[/bold] or set it in ~/.kalshi/.env"
        )
        raise typer.Exit(1)

    key = load_key()
    ts = str(int(time.time() * 1000))
    path = "/trade-api/v2/" + endpoint
    path_no_query = path.split("?")[0]
    sig = sign_request(ts, method, path_no_query, key)
    headers = {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type": "application/json",
    }
    url = BASE_URL + path

    if method == "GET":
        r = requests.get(url, headers=headers)
    elif method == "POST":
        r = requests.post(url, headers=headers, json=body)
    elif method == "DELETE":
        r = requests.delete(url, headers=headers)
    else:
        console.print(f"[red]Unsupported HTTP method:[/red] {method}")
        raise typer.Exit(1)

    if r.status_code not in (200, 201, 204):
        raise ApiError(r.status_code, r.text)
    if r.status_code == 204:
        return {}
    return r.json()


# ── Display Helpers ─────────────────────────────────────────


def fmt_price(dollars) -> str:
    """Format a dollar price as '$0.68 (68%)'"""
    if dollars is None or dollars == "N/A":
        return "—"
    try:
        d = float(dollars)
        pct = d * 100
        return f"${d:.2f} ({pct:.0f}%)"
    except (ValueError, TypeError):
        return str(dollars)


def fmt_dollars(val) -> str:
    """Format a plain dollar amount"""
    if val is None:
        return "—"
    try:
        return f"${float(val):.2f}"
    except (ValueError, TypeError):
        return str(val)


def filter_by_min_odds(markets: list, min_odds: float) -> list:
    """Filter out markets where either yes or no bid is below min_odds (as %)"""
    if min_odds <= 0:
        return markets
    threshold = min_odds / 100  # convert % to dollar value (e.g. 0.5% → 0.005)
    filtered = []
    for m in markets:
        yes = float(m.get("yes_bid_dollars", 0) or 0)
        no = float(m.get("no_bid_dollars", 0) or 0)
        if yes >= threshold and no >= threshold:
            filtered.append(m)
    return filtered


def sort_by_expiry(markets: list) -> list:
    """Filter out expired markets, then sort by expiration ascending (soonest first).
    Markets without an expiration are pushed to the end.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    future = []
    for m in markets:
        exp = m.get("expiration_time") or m.get("close_time") or ""
        if exp and exp < now:
            continue  # skip expired
        future.append(m)

    def key(m):
        exp = m.get("expiration_time") or m.get("close_time") or ""
        return exp if exp else "9999"
    return sorted(future, key=key)


def fmt_expiry(raw) -> str:
    """Format an expiration timestamp into a short human-readable string."""
    if not raw:
        return "—"
    try:
        from datetime import datetime, timezone
        # Handle ISO format (e.g. "2026-02-08T23:30:00Z")
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = dt - now
        days = delta.days
        if days < 0:
            return "expired"
        if days == 0:
            hours = delta.seconds // 3600
            mins = (delta.seconds % 3600) // 60
            if hours > 0:
                return f"{hours}h {mins}m"
            return f"{mins}m"
        if days < 7:
            return dt.strftime("%a %I:%M%p")
        if dt.year != now.year:
            return dt.strftime("%b %d, %Y")
        return dt.strftime("%b %d")
    except Exception:
        return str(raw)[:16]


def market_table(markets: list, title: str = "Markets", show_expiry: bool = False) -> Table:
    """Build a rich table for a list of markets.

    Auto-detects whether markets have subtitles (event outcomes like player/team names).
    If they do: shows Outcome + Title columns.
    If they don't: shows Title + Yes/No/Volume columns.
    """
    has_subtitles = any(
        m.get("yes_sub_title") or m.get("no_sub_title") for m in markets
    )

    t = Table(title=title, box=box.ROUNDED)

    if has_subtitles:
        t.add_column("Outcome", max_width=30)
    t.add_column("Title", max_width=45)
    if show_expiry:
        t.add_column("Expires", style="yellow")
    t.add_column("Yes", justify="right", style="green")
    t.add_column("No", justify="right", style="red")
    t.add_column("Volume", justify="right", style="dim")

    for m in markets:
        row_title = m.get("title", "")
        expiry = fmt_expiry(m.get("expiration_time") or m.get("close_time")) if show_expiry else None
        yes = fmt_price(m.get("yes_bid_dollars"))
        no = fmt_price(m.get("no_bid_dollars"))
        vol = fmt_dollars(m.get("volume_fp", 0))

        if has_subtitles:
            outcome = m.get("yes_sub_title", "") or m.get("no_sub_title", "") or ""
            row = [outcome, row_title]
        else:
            row = [row_title]
        if show_expiry:
            row.append(expiry)
        row.extend([yes, no, vol])
        t.add_row(*row)
    return t


def display_event(event_data: dict, limit: int = 10, min_odds: float = 0.5, expiring: bool = False):
    """Display an event and its markets with optional min odds filter"""
    e = event_data.get("event", {})
    all_markets = event_data.get("markets", [])
    filtered = filter_by_min_odds(all_markets, min_odds)
    if expiring:
        filtered = sort_by_expiry(filtered)

    console.print()
    console.print(f"[bold]{e.get('title', 'Event')}[/bold]")
    if len(filtered) < len(all_markets):
        console.print(f"  {len(filtered)} of {len(all_markets)} markets (min odds: {min_odds}%)")
    else:
        console.print(f"  {len(filtered)} markets")
    console.print()
    console.print(market_table(filtered[:limit], title="Event Markets", show_expiry=expiring))


# ── Series Discovery ────────────────────────────────────────

# Aliases: user term → additional search terms to broaden matching
# These supplement tag/title matching for common shorthand
SEARCH_ALIASES = {
    "nfl": ["football"],
    "epl": ["premier league"],
    "ucl": ["champions league"],
    "nba": ["basketball"],
    "mlb": ["baseball"],
    "nhl": ["hockey"],
    "f1": ["formula 1", "motorsport"],
    "mma": ["ufc"],
    "sb": ["super bowl"],
    "btc": ["bitcoin"],
    "eth": ["ethereum"],
}

# How many series before we show a list instead of fetching all markets
SERIES_DRILL_DOWN_THRESHOLD = 5

_series_cache = None
_active_series_cache = None


def get_all_series() -> list:
    """Fetch all series from the API (cached per session)"""
    global _series_cache
    if _series_cache is not None:
        return _series_cache
    data = api("GET", "series")
    _series_cache = data.get("series", [])
    return _series_cache


def get_active_series_tickers() -> dict:
    """Fetch series tickers that have open events, with earliest market close time.
    Returns dict: {series_ticker: earliest_close_time_iso_string}

    Uses events endpoint with nested markets to get both series_ticker
    and market close_time in a single paginated pass.
    """
    global _active_series_cache
    if _active_series_cache is not None:
        return _active_series_cache

    active = {}  # series_ticker -> earliest close_time

    with console.status("[dim]Loading active series...[/dim]"):
        cursor = ""
        for _ in range(20):  # paginate through all open events
            url = "events?status=open&limit=200&with_nested_markets=true"
            if cursor:
                url += f"&cursor={cursor}"
            data = api("GET", url)
            events = data.get("events", [])
            for e in events:
                st = e.get("series_ticker", "")
                if not st:
                    continue
                # Check nested markets for close times
                nested = e.get("markets") or []
                for m in nested:
                    exp = m.get("close_time") or m.get("expiration_time") or ""
                    if exp and (not active.get(st) or exp < active[st]):
                        active[st] = exp
                # Ensure series appears even if no nested markets returned
                if st not in active:
                    active[st] = ""
            cursor = data.get("cursor", "")
            if not cursor or not events:
                break

    _active_series_cache = active
    return active


def find_matching_series(query: str, active_only: bool = True) -> list:
    """Find series matching query against title, ticker, category, and tags.
    If active_only=True, only returns series that have open events.
    """
    query_lower = query.lower()

    # Expand with aliases
    search_terms = [query_lower]
    if query_lower in SEARCH_ALIASES:
        search_terms.extend(SEARCH_ALIASES[query_lower])
    # Also check if query is an alias value (e.g., "premier league" → add parent)
    for alias_key, alias_values in SEARCH_ALIASES.items():
        if query_lower in [v.lower() for v in alias_values]:
            search_terms.append(alias_key)

    all_series = get_all_series()

    if active_only:
        active_map = get_active_series_tickers()
    else:
        active_map = None

    matches = []
    seen = set()
    for s in all_series:
        ticker = s.get("ticker", "")
        if ticker in seen:
            continue
        if active_map is not None and ticker not in active_map:
            continue
        title = s.get("title", "").lower()
        ticker_lower = ticker.lower()
        category = s.get("category", "").lower()
        tags = " ".join(t.lower() for t in (s.get("tags") or []))
        searchable = f"{title} {ticker_lower} {category} {tags}"
        for term in search_terms:
            if term in searchable:
                matches.append(s)
                seen.add(ticker)
                break
    return matches


def display_series_list(
    series_list: list,
    query: str,
    expiring: bool = False,
    limit: int = 10,
    min_odds: float = 0.5,
):
    """Show an interactive numbered list of series. User can pick one to drill into."""
    active_map = get_active_series_tickers() if expiring else {}

    t = Table(title=f"'{query}' — {len(series_list)} series", box=box.ROUNDED)
    t.add_column("#", style="bold", justify="right", width=3)
    t.add_column("Ticker", style="cyan")
    t.add_column("Title", max_width=50)
    if expiring:
        t.add_column("Soonest", style="yellow")
    t.add_column("Tags", style="dim", max_width=30)

    for i, s in enumerate(series_list, 1):
        tags = ", ".join(s.get("tags") or [])
        row = [str(i), s.get("ticker", ""), s.get("title", "")]
        if expiring:
            row.append(fmt_expiry(active_map.get(s.get("ticker", ""), "")))
        row.append(tags)
        t.add_row(*row)
    console.print(t)

    # Interactive prompt
    while True:
        console.print()
        choice = console.input("[dim]Enter # to drill down (or q to quit):[/dim] ").strip()
        if not choice or choice.lower() == "q":
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(series_list):
                selected = series_list[idx]
                ticker = selected.get("ticker", "")
                console.print(f"\n[bold]Loading {selected.get('title', ticker)}...[/bold]")
                try:
                    data = api("GET", f"markets?series_ticker={ticker}&status=open&limit=200")
                    ms = filter_by_min_odds(data.get("markets", []), min_odds)
                    if expiring:
                        ms = sort_by_expiry(ms)
                    if ms:
                        console.print(market_table(ms[:limit], title=selected.get("title", ticker), show_expiry=expiring))
                    else:
                        console.print("[dim]No open markets in this series[/dim]")
                except ApiError as e:
                    console.print(f"[red]Error:[/red] {e}")
            else:
                console.print(f"[red]Pick 1-{len(series_list)}[/red]")
        except ValueError:
            console.print(f"[red]Enter a number or 'q'[/red]")


def display_series_markets(series_list: list, query: str, limit: int = 10, min_odds: float = 0.5, expiring: bool = False):
    """Fetch and display open markets across a small number of series"""
    all_markets = []
    series_names = []

    for s in series_list:
        ticker = s.get("ticker", "")
        if not ticker:
            continue
        series_names.append(s.get("title", ticker))
        try:
            data = api("GET", f"markets?series_ticker={ticker}&status=open&limit=100")
            for m in data.get("markets", []):
                all_markets.append(m)
        except ApiError:
            pass

    if not all_markets:
        return False

    filtered = filter_by_min_odds(all_markets, min_odds)
    if expiring:
        filtered = sort_by_expiry(filtered)

    console.print()
    console.print(f"[bold]{query.title()}[/bold] — {len(series_list)} series, {len(filtered)} markets")
    if len(filtered) < len(all_markets):
        console.print(f"  [dim](filtered from {len(all_markets)}, min odds: {min_odds}%)[/dim]")
    for name in series_names[:5]:
        console.print(f"  [dim]{name}[/dim]")
    if len(series_names) > 5:
        console.print(f"  [dim]...and {len(series_names) - 5} more[/dim]")
    console.print()
    console.print(market_table(filtered[:limit], title=f"{query.title()} Markets", show_expiry=expiring))
    return True


# ── Commands ────────────────────────────────────────────────


def handle_api_error(e: ApiError):
    """Print an API error and exit"""
    console.print(f"[red]API Error {e.status_code}:[/red] {e.message}")
    raise typer.Exit(1)


@app.command()
def markets(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of markets to show"),
    status: str = typer.Option("open", "--status", "-s", help="Filter by status (open, closed, settled)"),
):
    """List markets on Kalshi"""
    try:
        data = api("GET", f"markets?limit={limit}&status={status}")
    except ApiError as e:
        handle_api_error(e)
    ms = data.get("markets", [])
    if not ms:
        console.print("[dim]No markets found[/dim]")
        raise typer.Exit()
    console.print(market_table(ms, title=f"Kalshi Markets ({status}, top {len(ms)})"))


@app.command()
def search(
    query: str = typer.Argument(help="Search query, ticker, or category (e.g. 'soccer', 'KXWO-GOLD-26', 'oviedo')"),
    limit: int = typer.Option(10, "--limit", "-l", help="Max results"),
    min_odds: float = typer.Option(0.5, "--min-odds", "-m", help="Hide markets where either side is below this % (default 0.5)"),
    expiring: bool = typer.Option(False, "--expiring", "-e", help="Sort by expiration (soonest first)"),
):
    """Search markets by keyword, ticker, or category"""
    query_lower = query.lower()

    def apply_filters(ms: list) -> list:
        """Apply min-odds filter and optional expiry sort"""
        ms = filter_by_min_odds(ms, min_odds)
        if expiring:
            ms = sort_by_expiry(ms)
        return ms

    # ── Strategy 1: Direct ticker lookup (KX...) ──
    if query.upper().startswith("KX"):
        # Try as market ticker
        if "-" in query:
            try:
                data = api("GET", "markets/" + query.upper())
                m = data.get("market")
                if m:
                    console.print(market_table(apply_filters([m]), title=f"Market: {query.upper()}", show_expiry=expiring))
                    return
            except ApiError:
                pass

        # Try as event ticker
        try:
            ed = api("GET", "events/" + query.upper() + "?with_nested_markets=true")
            if ed.get("event") and ed.get("markets"):
                display_event(ed, limit=limit, min_odds=min_odds, expiring=expiring)
                return
        except ApiError:
            pass

        # Try as series ticker — fetch markets directly
        try:
            data = api("GET", f"markets?series_ticker={query.upper()}&status=open&limit=200")
            ms = apply_filters(data.get("markets", []))
            if ms:
                console.print(market_table(ms[:limit], title=f"Series: {query.upper()}", show_expiry=expiring))
                return
        except ApiError:
            pass

    # ── Strategy 2: Search series by keyword/tag/category ──
    try:
        matching_series = find_matching_series(query)
        if matching_series:
            if len(matching_series) > SERIES_DRILL_DOWN_THRESHOLD:
                # Sort and filter before slicing when -e is used
                if expiring:
                    active_map = get_active_series_tickers()
                    now_iso = __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat()
                    # Drop series whose soonest market already expired
                    matching_series = [
                        s for s in matching_series
                        if not active_map.get(s.get("ticker", ""))
                        or active_map[s["ticker"]] >= now_iso
                    ]
                    # Sort by soonest expiry before slicing
                    matching_series.sort(
                        key=lambda s: active_map.get(s.get("ticker", ""), "") or "9999"
                    )
                display_series_list(matching_series[:50], query, expiring=expiring, limit=limit, min_odds=min_odds)
                return
            else:
                # Few enough to fetch markets from all of them
                if display_series_markets(matching_series, query, limit=limit, min_odds=min_odds, expiring=expiring):
                    return
    except ApiError:
        pass

    # ── Strategy 3: Search open markets by title/ticker ──
    try:
        data = api("GET", "markets?limit=1000&status=open")
        ms = data.get("markets", [])
        matching = [
            m for m in ms
            if query_lower in m.get("title", "").lower()
            or query_lower in m.get("ticker", "").lower()
        ]
        matching = apply_filters(matching)
        if matching:
            console.print(market_table(matching[:limit], title=f"Search: {query}", show_expiry=expiring))
            return
    except ApiError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[dim]No markets found for '{query}'[/dim]")


@app.command()
def series(
    query: str = typer.Argument(None, help="Optional keyword to filter series"),
    all_series_flag: bool = typer.Option(False, "--all", "-a", help="Include series with no active markets"),
    expiring: bool = typer.Option(False, "--expiring", "-e", help="Sort by soonest expiry, show expiry column"),
):
    """List available series (market categories). Only shows active series by default."""
    try:
        all_series = get_all_series()
    except ApiError as e:
        handle_api_error(e)

    active_only = not all_series_flag
    if query:
        matching = find_matching_series(query, active_only=active_only)
    else:
        if active_only:
            active_map = get_active_series_tickers()
            matching = [s for s in all_series if s.get("ticker", "") in active_map]
        else:
            matching = all_series

    if not matching:
        console.print(f"[dim]No series found{' for ' + repr(query) if query else ''}[/dim]")
        raise typer.Exit()

    active_map = get_active_series_tickers() if expiring else {}
    if expiring:
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        # Drop expired series
        matching = [
            s for s in matching
            if not active_map.get(s.get("ticker", ""))
            or active_map[s["ticker"]] >= now_iso
        ]
        # Sort by soonest expiry
        matching.sort(
            key=lambda s: active_map.get(s.get("ticker", ""), "") or "9999"
        )

    display = matching[:50]

    t = Table(title=f"Series ({len(matching)})", box=box.ROUNDED)
    t.add_column("#", style="bold", justify="right", width=3)
    t.add_column("Ticker", style="cyan")
    t.add_column("Title", max_width=50)
    if expiring:
        t.add_column("Soonest", style="yellow")
    t.add_column("Category", style="dim")
    t.add_column("Tags", style="dim", max_width=30)

    for i, s in enumerate(display, 1):
        tags = ", ".join(s.get("tags") or [])
        row = [str(i), s.get("ticker", ""), s.get("title", "")]
        if expiring:
            row.append(fmt_expiry(active_map.get(s.get("ticker", ""), "")))
        row.extend([s.get("category", ""), tags])
        t.add_row(*row)
    console.print(t)

    # Interactive drill-down
    while True:
        console.print()
        choice = console.input("[dim]Enter # to drill down (or q to quit):[/dim] ").strip()
        if not choice or choice.lower() == "q":
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(display):
                selected = display[idx]
                ticker = selected.get("ticker", "")
                console.print(f"\n[bold]Loading {selected.get('title', ticker)}...[/bold]")
                try:
                    data = api("GET", f"markets?series_ticker={ticker}&status=open&limit=200")
                    ms = data.get("markets", [])
                    if expiring:
                        ms = sort_by_expiry(ms)
                    if ms:
                        console.print(market_table(ms[:20], title=selected.get("title", ticker), show_expiry=expiring))
                    else:
                        console.print("[dim]No open markets in this series[/dim]")
                except ApiError as e:
                    console.print(f"[red]Error:[/red] {e}")
            else:
                console.print(f"[red]Pick 1-{len(display)}[/red]")
        except ValueError:
            console.print(f"[red]Enter a number or 'q'[/red]")


@app.command()
def detail(ticker: str = typer.Argument(help="Market ticker (e.g. KXWO-GOLD-26-NOR)")):
    """Show detailed info for a single market"""
    try:
        data = api("GET", "markets/" + ticker.upper())
    except ApiError as e:
        handle_api_error(e)
    m = data.get("market", data)

    console.print()
    console.print(Panel(
        f"[bold]{m.get('title', 'N/A')}[/bold]\n\n"
        f"  Ticker:        [cyan]{m.get('ticker', '')}[/cyan]\n"
        f"  Status:        {m.get('status', 'N/A')}\n"
        f"  Yes Bid:       [green]{fmt_price(m.get('yes_bid_dollars'))}[/green]\n"
        f"  Yes Ask:       [green]{fmt_price(m.get('yes_ask_dollars'))}[/green]\n"
        f"  No Bid:        [red]{fmt_price(m.get('no_bid_dollars'))}[/red]\n"
        f"  No Ask:        [red]{fmt_price(m.get('no_ask_dollars'))}[/red]\n"
        f"  Last Price:    {fmt_price(m.get('last_price_dollars'))}\n"
        f"  Volume:        {fmt_dollars(m.get('volume_fp', 0))}\n"
        f"  Open Interest: {m.get('open_interest', 'N/A')}\n"
        f"  Expiration:    {m.get('expiration_time', 'N/A')}\n"
        f"  Category:      {m.get('category', 'N/A')}\n"
        f"  Subtitle:      {m.get('yes_sub_title', '') or m.get('no_sub_title', '') or '—'}",
        title="Market Detail",
        border_style="blue",
    ))


@app.command()
def orderbook(ticker: str = typer.Argument(help="Market ticker")):
    """Show the orderbook for a market"""
    try:
        data = api("GET", f"markets/{ticker.upper()}/orderbook")
    except ApiError as e:
        handle_api_error(e)
    ob = data.get("orderbook", data)

    console.print()

    # Yes side
    yes_table = Table(title=f"{ticker.upper()} — YES", box=box.SIMPLE, style="green")
    yes_table.add_column("Price", justify="right")
    yes_table.add_column("Quantity", justify="right")
    for level in ob.get("yes", []):
        # API may return [price, qty] arrays or {price, quantity} dicts
        if isinstance(level, (list, tuple)):
            price, qty = level[0], level[1]
        else:
            price = level.get("price", level.get("yes_price", 0))
            qty = level.get("quantity", 0)
        if qty > 0:
            yes_table.add_row(f"{price}¢", str(qty))

    # No side
    no_table = Table(title=f"{ticker.upper()} — NO", box=box.SIMPLE, style="red")
    no_table.add_column("Price", justify="right")
    no_table.add_column("Quantity", justify="right")
    for level in ob.get("no", []):
        if isinstance(level, (list, tuple)):
            price, qty = level[0], level[1]
        else:
            price = level.get("price", level.get("no_price", 0))
            qty = level.get("quantity", 0)
        if qty > 0:
            no_table.add_row(f"{price}¢", str(qty))

    console.print(yes_table)
    console.print(no_table)


@app.command()
def balance():
    """Show account balance"""
    try:
        data = api("GET", "portfolio/balance")
    except ApiError as e:
        handle_api_error(e)
    b = data.get("balance", 0) / 100
    console.print(f"\n[bold]Balance:[/bold] ${b:.2f}\n")


@app.command()
def positions():
    """Show current positions"""
    try:
        data = api("GET", "portfolio/positions")
    except ApiError as e:
        handle_api_error(e)
    ps = data.get("market_positions", [])
    if not ps:
        console.print("[dim]No open positions[/dim]")
        raise typer.Exit()

    t = Table(title="Current Positions", box=box.ROUNDED)
    t.add_column("Ticker", style="cyan", max_width=28)
    t.add_column("Side", justify="center")
    t.add_column("Qty", justify="right")
    t.add_column("Exposure", justify="right", style="yellow")

    for p in ps:
        pos = p.get("position", 0)
        side = "[green]Yes[/green]" if pos > 0 else "[red]No[/red]"
        t.add_row(
            p.get("ticker", ""),
            side,
            str(abs(pos)),
            fmt_dollars(p.get("market_exposure_dollars", 0)),
        )
    console.print(t)


@app.command()
def orders():
    """Show open (resting) orders"""
    try:
        data = api("GET", "portfolio/orders?status=resting")
    except ApiError as e:
        handle_api_error(e)
    order_list = data.get("orders", [])
    if not order_list:
        console.print("[dim]No open orders[/dim]")
        raise typer.Exit()

    t = Table(title="Open Orders", box=box.ROUNDED)
    t.add_column("Order ID", style="dim", max_width=14)
    t.add_column("Ticker", style="cyan")
    t.add_column("Action")
    t.add_column("Side", justify="center")
    t.add_column("Price", justify="right")
    t.add_column("Qty", justify="right")
    t.add_column("Remaining", justify="right")

    for o in order_list:
        side_str = "[green]Yes[/green]" if o.get("side") == "yes" else "[red]No[/red]"
        t.add_row(
            o.get("order_id", "")[:14],
            o.get("ticker", ""),
            o.get("action", ""),
            side_str,
            f"{o.get('yes_price', 0)}¢",
            str(o.get("count", 0)),
            str(o.get("remaining_count", 0)),
        )
    console.print(t)


@app.command()
def buy(
    ticker: str = typer.Argument(help="Market ticker"),
    count: int = typer.Argument(help="Number of contracts"),
    price: int = typer.Argument(help="Price in cents (e.g. 68 for $0.68)"),
    side: str = typer.Option("yes", "--side", "-s", help="Contract side: 'yes' or 'no'"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
):
    """Place a limit buy order"""
    side = side.lower()
    if side not in ("yes", "no"):
        console.print("[red]Error:[/red] --side must be 'yes' or 'no'")
        raise typer.Exit(1)

    cost = count * price
    console.print(
        f"\n  Buy [bold]{count}x {side.upper()}[/bold] on "
        f"[cyan]{ticker.upper()}[/cyan] @ {price}¢ each"
    )
    console.print(f"  Max cost: [yellow]${cost / 100:.2f}[/yellow]\n")

    if not force:
        if not typer.confirm("Confirm order?"):
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit()

    api_body = {
        "action": "buy",
        "count": count,
        "side": side,
        "ticker": ticker.upper(),
        "client_order_id": str(int(time.time())),
        "type": "limit",
        "yes_price": price if side == "yes" else 100 - price,
    }
    try:
        res = api("POST", "portfolio/orders", body=api_body)
    except ApiError as e:
        handle_api_error(e)
    order_id = res.get("order", {}).get("order_id", "unknown")
    console.print(f"[bold green]Order placed![/bold green] ID: {order_id}")


@app.command()
def sell(
    ticker: str = typer.Argument(help="Market ticker"),
    count: int = typer.Argument(help="Number of contracts"),
    price: int = typer.Argument(help="Price in cents"),
    side: str = typer.Option("yes", "--side", "-s", help="Contract side: 'yes' or 'no'"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
):
    """Place a limit sell order"""
    side = side.lower()
    if side not in ("yes", "no"):
        console.print("[red]Error:[/red] --side must be 'yes' or 'no'")
        raise typer.Exit(1)

    proceeds = count * price
    console.print(
        f"\n  Sell [bold]{count}x {side.upper()}[/bold] on "
        f"[cyan]{ticker.upper()}[/cyan] @ {price}¢ each"
    )
    console.print(f"  Expected proceeds: [yellow]${proceeds / 100:.2f}[/yellow]\n")

    if not force:
        if not typer.confirm("Confirm order?"):
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit()

    api_body = {
        "action": "sell",
        "count": count,
        "side": side,
        "ticker": ticker.upper(),
        "client_order_id": str(int(time.time())),
        "type": "limit",
        "yes_price": price if side == "yes" else 100 - price,
    }
    try:
        res = api("POST", "portfolio/orders", body=api_body)
    except ApiError as e:
        handle_api_error(e)
    order_id = res.get("order", {}).get("order_id", "unknown")
    console.print(f"[bold green]Sell order placed![/bold green] ID: {order_id}")


@app.command()
def cancel(
    order_id: str = typer.Argument(help="Order ID to cancel"),
):
    """Cancel an open order"""
    try:
        api("DELETE", f"portfolio/orders/{order_id}")
    except ApiError as e:
        handle_api_error(e)
    console.print(f"[bold green]Order {order_id} cancelled[/bold green]")


@app.command(name="setup-shell")
def setup_shell():
    """Add KALSHI_ACCESS_KEY to your shell config"""
    load_env()
    key_id = os.getenv("KALSHI_ACCESS_KEY")
    if not key_id:
        console.print("[red]Error:[/red] No KALSHI_ACCESS_KEY found in ~/.kalshi/.env")
        raise typer.Exit(1)

    line = f'export KALSHI_ACCESS_KEY="{key_id}"\n'
    bashrc = os.path.expanduser("~/.bashrc")
    zshrc = os.path.expanduser("~/.zshrc")

    updated = False
    for rc in [bashrc, zshrc]:
        if os.path.exists(rc):
            with open(rc, "r") as f:
                content = f.read()
            if key_id not in content:
                with open(rc, "a") as f:
                    f.write(f"\n# Kalshi API Key\n{line}")
                console.print(f"[green]Added[/green] KALSHI_ACCESS_KEY to {rc}")
                updated = True
            else:
                console.print(f"[dim]KALSHI_ACCESS_KEY already in {rc}[/dim]")
                updated = True

    if updated:
        console.print("\nRestart your terminal or run [bold]source ~/.bashrc[/bold]")
    else:
        console.print("[dim]No shell config files found (~/.bashrc or ~/.zshrc)[/dim]")


if __name__ == "__main__":
    app()
