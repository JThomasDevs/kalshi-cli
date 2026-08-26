#!/usr/bin/env python3
"""Kalshi CLI - Prediction market trading tool"""

import base64
import json
import os
import sys
import time
import uuid

import requests
import typer
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Kalshi prediction market CLI")
BASE_URL = os.getenv("KALSHI_BASE_URL", "https://external-api.kalshi.com")
DEMO_BASE_URL = "https://external-api.demo.kalshi.co"
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
    with open(env_path) as f:
        for line in f:
            line = line.removeprefix("export ")
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


def api(method: str, endpoint: str, body: dict | None = None) -> dict:
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
    timeout = 10

    try:
        if method == "GET":
            r = requests.get(url, headers=headers, timeout=timeout)
        elif method == "POST":
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
        elif method == "DELETE":
            r = requests.delete(url, headers=headers, timeout=timeout)
        else:
            console.print(f"[red]Unsupported HTTP method:[/red] {method}")
            raise typer.Exit(1)
    except requests.exceptions.RequestException as e:
        raise ApiError(0, f"Network error: {e}") from e

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


def to_cents(val) -> int | None:
    """Convert a dollar amount (int/float/str) to integer cents. Returns None on bad input."""
    if val is None:
        return None
    try:
        return int(float(val) * 100)
    except (ValueError, TypeError):
        return None


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


def _parse_expiry_from_ticker(ticker: str):
    """Parse a 15M/5M-series ticker's expiry segment into an ISO timestamp in UTC.

    Kalshi's 15M/5M tickers use a custom 11-char segment (e.g. ``26AUG242345``) whose
    format is documented internally but not publicly stable enough to parse here
    without risk. The API reliably returns ``close_time`` and ``expected_expiration_time``
    for every market, so this function is dead code in practice and was removed
    in 1.2.74. Kept as a stub returning None to preserve import-time signature
    compatibility in case any external tooling references it.

    Returns: None.
    """
    return None


def _market_expiry_ts(m: dict) -> str:
    """Best expiry timestamp from API fields.

    15M/5M-series ticker-segment parsing was removed in 1.2.74 — it relied on an
    undocumented ticker format that didn't match real-world samples. The API
    reliably returns ``expected_expiration_time`` / ``close_time`` for every
    market, so we just use those.
    """
    return (
        m.get("expected_expiration_time")
        or m.get("close_time")
        or m.get("expiration_time")
        or m.get("latest_expiration_time")
        or ""
    )


def sort_by_expiry(markets: list) -> list:
    """Filter out expired markets, then sort by expiration ascending (soonest first).
    Markets without an expiration are pushed to the end.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    future = []
    for m in markets:
        exp = _market_expiry_ts(m)
        if exp and exp < now:
            continue  # skip expired
        future.append(m)

    def key(m):
        exp = _market_expiry_ts(m)
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


def is_parlay(m: dict) -> bool:
    """Detect parlay/combo markets by checking for multiple comma-separated legs."""
    for field in ("yes_sub_title", "no_sub_title", "title"):
        val = m.get(field, "") or ""
        if val.count(",") >= 2:
            return True
    return False


def _fmt_legs(text: str) -> str:
    """Format comma-separated parlay legs as one leg per line."""
    if not text:
        return text
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) <= 1:
        return text
    return "\n".join(parts)


def _is_binary_yes_no_market(title: str) -> bool:
    """True if the market is a single yes/no proposition (e.g. 'Will Gavin Newsom be...' or 'Team X wins by over 2.5 points') not a multi-outcome pick."""
    if not title:
        return False
    t = title.strip().lower()
    # "Will [X] be/have/win..." = one proposition → binary
    if t.startswith("will ") and (" be " in t or " win " in t or " have " in t or " the " in t):
        return True
    # Spread markets: "Team wins by over X points" = binary
    return bool("wins by over" in t and "points" in t)


def _position_side_label(yes_sub: str, no_sub: str, position: int, title: str = "") -> str:
    """Side column: multifaceted markets show the choice (e.g. Anduril, May 31 2026); yes/no markets show Yes or No."""
    if position < 0:
        return "No"
    if _is_binary_yes_no_market(title):
        return "Yes"
    # Holding Yes: show outcome name for multi-outcome markets, else "Yes"
    sub = yes_sub or ""
    if not sub:
        return "Yes"
    parts = [p.strip() for p in sub.split(",") if p.strip()]
    if len(parts) >= 2 and all(
        p.lower().startswith("yes ") or p.lower().startswith("no ") for p in parts
    ):
        return "Yes"
    if sub.strip().lower().startswith("yes "):
        return sub.strip()[4:].strip()
    if sub.strip().lower().startswith("no "):
        return sub.strip()[3:].strip()
    return sub.strip()


def _parse_leg(leg: str) -> tuple:
    """Split a leg into (outcome, title). Outcome is 'yes'/'no', title is the proposition."""
    leg = leg.strip()
    if leg.lower().startswith("yes "):
        return ("yes", leg[4:].strip())
    if leg.lower().startswith("no "):
        return ("no", leg[3:].strip())
    return ("yes", leg)


def _outcome_from_ticker(ticker: str) -> str:
    """Derive a short outcome label from market ticker (e.g. range/strike suffix). KXBTC-95-96 -> 95-96; KXBTC-26FEB0917-B74250 -> $97,425)."""
    if not ticker or "-" not in ticker:
        return ""
    parts = ticker.split("-")
    if len(parts) >= 3:
        suffix = "-".join(parts[-2:])  # e.g. 95-96 or 26FEB0917-B74250
        # Price-range style: last segment is strike in dollars (e.g. B74250 → $74,250 or above)
        last = parts[-1]
        if len(last) >= 4 and last[0].isalpha() and last[1:].isdigit():
            try:
                num = int(last[1:])
                # 1 letter + 4 digits: letter = leading digit (A=0..I=9), e.g. I7425 → 97425
                if len(last) == 5 and last[0].upper() in "ABCDEFGHIJ":
                    lead = "ABCDEFGHIJ".index(last[0].upper())
                    num = lead * 10000 + num
                if num >= 1000:
                    return f"${num:,} or above"
            except ValueError:
                pass
        if len(last) >= 2 and last.isdigit():
            try:
                num = int(last)
                if num >= 1000:
                    return f"${num:,} or above"
            except ValueError:
                pass
        return suffix
    return parts[-1] if parts else ""


def _is_generic_subtitle(sub: str) -> bool:
    """True if subtitle is just 'yes'/'no' with no actual outcome label (e.g. range markets)."""
    s = (sub or "").strip().lower()
    return s in ("yes", "no")


def _is_parlay_subtitle(sub: str) -> bool:
    """True if subtitle looks like parlay legs: 'Yes Team A wins, No Team B wins'."""
    parts = [p.strip() for p in sub.split(",") if p.strip()]
    if len(parts) < 2:
        return False
    return all(
        p.lower().startswith("yes ") or p.lower().startswith("no ")
        for p in parts
    )


def _outcome_column(m: dict) -> str:
    """Outcome column: for binary Yes/No markets, color YES green or NO red based on higher %.
    For multi-outcome markets: returns the outcome name.
    For parlays: returns yes/no per leg.
    """
    title = m.get("title", "") or ""

    # Binary Yes/No markets: show YES or NO based on higher percentage
    if _is_binary_yes_no_market(title):
        yes_price = float(m.get("yes_bid_dollars", 0) or 0)
        no_price = float(m.get("no_bid_dollars", 0) or 0)
        if yes_price > no_price:
            return "[green]YES[/green]"
        elif no_price > yes_price:
            return "[red]NO[/red]"
        else:
            return "YES"

    # Try subtitle (for multi-outcome markets like Olympics, player props)
    sub = m.get("yes_sub_title", "") or m.get("no_sub_title", "") or ""
    if sub and not _is_generic_subtitle(sub):
        if _is_parlay_subtitle(sub):
            # Parlay: show yes/no per leg
            parts = [p.strip() for p in sub.split(",") if p.strip()]
            return "\n".join(_parse_leg(p)[0] for p in parts)
        # Single outcome: strip "Yes "/"No " prefix if present
        return _parse_leg(sub.strip())[1]

    # Fall back to strike/range from API or ticker
    out = m.get("strike") or m.get("strike_price") or m.get("resolution_value")
    if out is not None and str(out).strip():
        return str(out).strip()
    ticker = m.get("ticker", "") or m.get("market_ticker", "") or ""
    out = _outcome_from_ticker(ticker)
    if out:
        return out
    return "Yes"


def _normalize_title(raw: str) -> str:
    """Single line: collapse newlines and extra spaces."""
    if not raw:
        return ""
    return " ".join(str(raw).split())


def _title_column(m: dict) -> str:
    """Title column: one proposition (title) per line for each leg (parlays only)."""
    raw = m.get("title", "") or m.get("yes_sub_title", "") or m.get("no_sub_title", "") or ""
    raw = _normalize_title(raw)
    if not raw:
        return ""
    if _is_parlay_subtitle(raw):
        # Parlay: show each leg's title on its own line
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return "\n".join(_parse_leg(p)[1] for p in parts)
    # Single market: show title as-is (strip "Yes "/"No " prefix if present)
    return _parse_leg(raw)[1] if raw else raw


def _is_up_down_market(markets: list) -> bool:
    """Detect up/down price markets (e.g. 'BTC price up in next 15 mins?')."""
    if not markets:
        return False
    return all(
        "up" in (m.get("title", "") or "").lower()
        and ("price" in (m.get("title", "") or "").lower() or "15 min" in (m.get("title", "") or "").lower())
        for m in markets
    )


def _clean_up_down_outcome(sub: str) -> str:
    """Clean up 'Price to beat: $70,783.57' → 'Target: $70,783.57'."""
    s = (sub or "").strip()
    if s.lower().startswith("price to beat:"):
        return "Target:" + s[len("price to beat:"):]
    return s


def market_table(markets: list, title: str = "Markets", show_expiry: bool = False, numbered: bool = False) -> Table:
    """Build a rich table for a list of markets.

    Auto-detects whether markets have subtitles (event outcomes like player/team names).
    If they do: shows Outcome + Title columns.
    If they don't: shows Title + Yes/No/Volume columns.
    If numbered=True, adds a # column for drill-down (1, 2, 3, ...).
    """
    # Detect whether markets need an Outcome column:
    # 1. Real subtitles (not generic yes/no) — multi-outcome markets like Olympics
    # 2. Ticker-derived outcomes — range markets like Bitcoin price
    has_real_subtitles = any(
        (m.get("yes_sub_title") or m.get("no_sub_title") or "")
        and not _is_generic_subtitle(m.get("yes_sub_title", "") or m.get("no_sub_title", ""))
        and not _is_binary_yes_no_market(m.get("title", ""))
        for m in markets
    )
    has_ticker_outcomes = (
        not has_real_subtitles
        and any(_outcome_from_ticker(m.get("ticker", "")) for m in markets)
    )
    has_binary_outcomes = any(_is_binary_yes_no_market(m.get("title", "")) for m in markets)
    has_subtitles = has_real_subtitles or has_ticker_outcomes or has_binary_outcomes

    # Up/down markets get directional column labels
    up_down = _is_up_down_market(markets)
    yes_label = "Up ↑" if up_down else "Yes"
    no_label = "Down ↓" if up_down else "No"

    t = Table(title=title, box=box.ROUNDED)

    if numbered:
        t.add_column("#", style="bold", justify="right", width=3)
    if has_subtitles:
        t.add_column("Prediction", max_width=30)
    t.add_column("Title", max_width=45)
    if show_expiry:
        t.add_column("Expires", style="yellow")
    t.add_column(yes_label, justify="right", style="green")
    t.add_column(no_label, justify="right", style="red")
    t.add_column("Volume", justify="right", style="dim")
    t.add_column("Ticker", style="cyan", overflow="fold")

    for i, m in enumerate(markets, 1):
        expiry = fmt_expiry(_market_expiry_ts(m)) if show_expiry else None
        yes = fmt_price(m.get("yes_bid_dollars"))
        no = fmt_price(m.get("no_bid_dollars"))
        vol = fmt_dollars(m.get("volume_fp", 0))

        row = [str(i)] if numbered else []
        if has_subtitles:
            outcome = _outcome_column(m)
            if up_down:
                outcome = _clean_up_down_outcome(outcome)
            row.extend([outcome, _title_column(m)])
        else:
            row.append(_title_column(m))
        if show_expiry:
            row.append(expiry)
        row.extend([yes, no, vol, m.get("ticker", "")])
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
                    exp = _market_expiry_ts(m)
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

    active_map = get_active_series_tickers() if active_only else None

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
    t.add_column("Ticker", style="cyan", overflow="fold")
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
                        console.print(market_table(ms[:limit], title=selected.get("title", ticker), show_expiry=expiring, numbered=True))
                        if not _prompt_market_drill_down(ms, limit):
                            return
                    else:
                        console.print("[dim]No open markets in this series[/dim]")
                except ApiError as e:
                    console.print(f"[red]Error:[/red] {e}")
            else:
                console.print(f"[red]Pick 1-{len(series_list)}[/red]")
        except ValueError:
            console.print("[red]Enter a number or 'q'[/red]")


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
            all_markets.extend(data.get("markets", []))
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
    """Print an API error and exit.

    In JSON/agent mode (KALSHI_OUTPUT=json), emits a stable error envelope to
    stdout with a non-zero exit code. In human mode, prints a colored error to
    stderr and exits.
    """
    if os.getenv("KALSHI_OUTPUT", "").lower() == "json":
        sys.stdout.write(json.dumps({
            "error": {
                "code": _classify_error(e.status_code, e.message),
                "status": e.status_code,
                "message": e.message,
            }
        }) + "\n")
        sys.stdout.flush()
        raise typer.Exit(_exit_code_for_status(e.status_code))
    console.print(f"[red]API Error {e.status_code}:[/red] {e.message}")
    raise typer.Exit(1)


def _classify_error(status_code: int, message: str) -> str:
    """Map HTTP status + body to a stable agent-parseable error code."""
    msg_lower = (message or "").lower()
    if status_code == 410:
        return "DEPRECATED_ENDPOINT"
    if status_code == 401:
        return "AUTH_FAILED"
    if status_code == 403:
        return "FORBIDDEN"
    if status_code == 404:
        return "NOT_FOUND"
    if status_code == 429:
        return "RATE_LIMITED"
    if status_code in (400, 422):
        if "insufficient" in msg_lower or "balance" in msg_lower:
            return "INSUFFICIENT_FUNDS"
        if "invalid" in msg_lower and "ticker" in msg_lower:
            return "INVALID_TICKER"
        if "price" in msg_lower and ("invalid" in msg_lower or "out" in msg_lower):
            return "INVALID_PRICE"
        if "quantity" in msg_lower or "count" in msg_lower or "size" in msg_lower:
            return "INVALID_QUANTITY"
        return "BAD_REQUEST"
    if status_code in (500, 502, 503, 504):
        return "SERVER_ERROR"
    return "API_ERROR"


def _exit_code_for_status(status_code: int) -> int:
    """Distinct process exit codes per error category so agents can branch on $?."""
    if status_code == 401:
        return 2
    if status_code == 403:
        return 3
    if status_code == 404:
        return 4
    if status_code == 429:
        return 5
    if status_code in (400, 422):
        return 6
    if status_code in (500, 502, 503, 504):
        return 7
    return 1


def output_json(data, exit_code: int = 0) -> None:
    """Print data as JSON to stdout and exit.

    Flushes stdout before raising typer.Exit so subprocess.Popen capture works
    correctly when downstream callers pipe the output.
    """
    sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")
    sys.stdout.flush()
    raise typer.Exit(exit_code)


def json_mode(*, flag: bool = False) -> bool:
    """Resolve whether the current invocation should produce JSON output.

    True if EITHER the --json flag was passed for this command OR the
    KALSHI_OUTPUT=json env var is set. The env var lets agents skip adding
    --json to every command.
    """
    if flag:
        return True
    return os.getenv("KALSHI_OUTPUT", "").lower() == "json"


@app.command()
def markets(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of markets to show"),
    status: str = typer.Option("open", "--status", "-s", help="Filter by status (open, closed, settled)"),
    no_parlays: bool = typer.Option(False, "--no-parlays", help="Exclude parlay/multi-leg markets"),
    all_markets: bool = typer.Option(False, "--all", "-a", help="Show all markets instead of just recently traded ones"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """List markets on Kalshi (defaults to most recently traded)"""
    ms = []

    if not all_markets:
        with console.status("[dim]Loading active markets from recent trades...[/dim]"):
            try:
                trades_data = api("GET", "markets/trades?limit=500")
                trades = trades_data.get("trades", [])
            except ApiError:
                trades = []

            # Get unique tickers from trades
            seen = set()
            for t in trades:
                ticker = t.get("ticker", "")
                if ticker:
                    seen.add(ticker)

            # Batch fetch all markets at once
            if seen:
                tickers_str = ",".join(list(seen)[:100])  # API limit
                try:
                    data = api("GET", f"markets?tickers={tickers_str}&status={status}")
                    ms = data.get("markets", [])
                except ApiError:
                    ms = []
    else:
        try:
            data = api("GET", f"markets?limit=200&status={status}")
            ms = data.get("markets", [])
        except ApiError as e:
            handle_api_error(e)

    if not ms:
        console.print("[dim]No markets found[/dim]")
        raise typer.Exit()

    # Filter out parlays if requested
    if no_parlays:
        ms = [m for m in ms if not is_parlay(m)]

    # Sort by volume descending
    def sort_key(m):
        vol = float(m.get("volume_fp", 0) or 0)
        return vol
    ms.sort(key=sort_key, reverse=True)
    ms = ms[:limit]

    # Output as JSON if global flag is set
    if json_output:
        output = [{
            "ticker": m.get("ticker", ""),
            "title": m.get("title", ""),
            "yes_bid": m.get("yes_bid_dollars"),
            "no_bid": m.get("no_bid_dollars"),
            "yes_ask": m.get("yes_ask_dollars"),
            "no_ask": m.get("no_ask_dollars"),
            "volume": m.get("volume_fp"),
            "status": m.get("status"),
            "expiration": m.get("expiration_time", "")[:10] if m.get("expiration_time") else ""
        } for m in ms]
        output_json(output)

    console.print(market_table(ms, title=f"Kalshi Markets ({status}, top {len(ms)})"))


@app.command()
def search(
    query: str = typer.Argument(help="Search query, ticker, or category (e.g. 'soccer', 'KXWO-GOLD-26', 'oviedo')"),
    limit: int = typer.Option(10, "--limit", "-l", help="Max results"),
    min_odds: float = typer.Option(0.5, "--min-odds", "-m", help="Hide markets where either side is below this % (default 0.5)"),
    expiring: bool = typer.Option(False, "--expiring", "-e", help="Sort by expiration (soonest first)"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON (or set KALSHI_OUTPUT=json)"),
):
    """Search markets by keyword, ticker, or category"""
    as_json = json_mode(flag=json_output)
    query_lower = query.lower()
    query_lower = query.lower()

    def apply_filters(ms: list) -> list:
        """Apply min-odds filter and optional expiry sort"""
        ms = filter_by_min_odds(ms, min_odds)
        if expiring:
            ms = sort_by_expiry(ms)
        return ms

    # ── Strategy 1: Direct ticker lookup (KX...) ──
    if query.upper().startswith("KX"):
        # Try as market ticker (full ticker has hyphens, e.g. KXBTC15M-26FEB161715-15)
        if "-" in query:
            try:
                data = api("GET", "markets/" + query.upper())
                m = data.get("market")
                if m:
                    if as_json:
                        output_json(data)
                    console.print(market_table(apply_filters([m]), title=f"Market: {query.upper()}", show_expiry=expiring))
                    return
            except ApiError:
                pass

        # Try as series ticker first when query has no hyphen (e.g. KXBTC15M)
        # so we get real market close times; event API often returns container event with wrong expiry
        try:
            data = api("GET", f"markets?series_ticker={query.upper()}&status=open&limit=200")
            ms = apply_filters(data.get("markets", []))
            if ms:
                if as_json:
                    output_json({"strategy": "series_ticker", "query": query, "markets": ms[:limit]})
                console.print(market_table(ms[:limit], title=f"Series: {query.upper()}", show_expiry=expiring))
                return
        except ApiError:
            pass

        # Try as event ticker (e.g. full event id)
        try:
            ed = api("GET", "events/" + query.upper() + "?with_nested_markets=true")
            if ed.get("event") and ed.get("markets"):
                display_event(ed, limit=limit, min_odds=min_odds, expiring=expiring)
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
                if as_json:
                    output_json({"strategy": "series_keyword", "query": query, "series": matching_series[:50]})
                display_series_list(matching_series[:50], query, expiring=expiring, limit=limit, min_odds=min_odds)
                return
            else:
                # Few enough to fetch markets from all of them
                if as_json:
                    output_json({"strategy": "series_markets", "query": query, "series": matching_series})
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
            if as_json:
                output_json({"strategy": "title_match", "query": query, "markets": matching[:limit]})
            console.print(market_table(matching[:limit], title=f"Search: {query}", show_expiry=expiring))
            return
    except ApiError as e:
        if as_json:
            output_json({"error": {"code": _classify_error(e.status_code, e.message), "status": e.status_code, "message": e.message}}, exit_code=_exit_code_for_status(e.status_code))
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if as_json:
        output_json({"strategy": "none", "query": query, "markets": []}, exit_code=4)
    console.print(f"[dim]No markets found for '{query}'[/dim]")


@app.command()
def series(
    query: str = typer.Argument(None, help="Optional keyword to filter series"),
    all_series_flag: bool = typer.Option(False, "--all", "-a", help="Include series with no active markets"),
    expiring: bool = typer.Option(False, "--expiring", "-e", help="Sort by soonest expiry, show expiry column"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON (or set KALSHI_OUTPUT=json)"),
):
    """List available series (market categories). Only shows active series by default."""
    as_json = json_mode(flag=json_output)
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
        if as_json:
            output_json({"query": query, "series": []}, exit_code=4)
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
    t.add_column("Ticker", style="cyan", overflow="fold")
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
    if as_json:
        soonest = {s.get("ticker", ""): active_map.get(s.get("ticker", ""), "") for s in display} if expiring else {}
        output_json({"query": query, "count": len(matching), "series": [
            {**s, "soonest_expiration": soonest.get(s.get("ticker", ""))} if expiring else s
            for s in display
        ]})
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
                        console.print(market_table(ms[:20], title=selected.get("title", ticker), show_expiry=expiring, numbered=True))
                        if not _prompt_market_drill_down(ms, 20):
                            return
                    else:
                        console.print("[dim]No open markets in this series[/dim]")
                except ApiError as e:
                    console.print(f"[red]Error:[/red] {e}")
            else:
                console.print(f"[red]Pick 1-{len(display)}[/red]")
        except ValueError:
            console.print("[red]Enter a number or 'q'[/red]")


def _show_market_detail_panel(m: dict):
    """Render the market detail Panel (shared by detail command and drill-down)."""
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
        f"  Expiration:    {_market_expiry_ts(m) or 'N/A'}\n"
        f"  Category:      {m.get('category', 'N/A')}\n"
        f"  Subtitle:      {m.get('yes_sub_title', '') or m.get('no_sub_title', '') or '—'}",
        title="Market Detail",
        border_style="blue",
    ))


def _prompt_market_drill_down(ms: list, limit: int) -> bool:
    """Let user pick a market # to see detail. Returns True for 'b' (back), False for 'q' (quit)."""
    n = min(len(ms), limit)
    while True:
        console.print()
        choice = console.input(
            f"[dim]Enter market # (1-{n}) for detail, b back to series list, q quit:[/dim] "
        ).strip().lower()
        if not choice or choice == "q":
            return False
        if choice == "b":
            return True
        try:
            idx = int(choice) - 1
            if 0 <= idx < n:
                m = ms[idx]
                ticker = m.get("ticker", "") or m.get("market_ticker", "")
                if not ticker:
                    console.print("[red]No ticker for this market[/red]")
                    continue
                try:
                    data = api("GET", "markets/" + ticker.upper())
                    _show_market_detail_panel(data.get("market", data))
                except ApiError as e:
                    console.print(f"[red]Error:[/red] {e}")
            else:
                console.print(f"[red]Pick 1-{n}[/red]")
        except ValueError:
            console.print("[red]Enter a number, b, or q[/red]")


@app.command()
def detail(
    ticker: str = typer.Argument(help="Market ticker (e.g. KXWO-GOLD-26-NOR)"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON (or set KALSHI_OUTPUT=json)"),
):
    """Show detailed info for a single market"""
    try:
        data = api("GET", "markets/" + ticker.upper())
    except ApiError as e:
        handle_api_error(e)
    if json_mode(flag=json_output):
        output_json(data)
    _show_market_detail_panel(data.get("market", data))


def _orderbook_best_bid_cents(levels: list, in_dollars: bool) -> int | None:
    """Return the best (highest) bid price in cents, or None if no levels. Levels are ascending so last is best."""
    if not levels:
        return None
    level = levels[-1]
    if not isinstance(level, (list, tuple)) or len(level) < 1:
        return None
    price_val = level[0]
    try:
        if in_dollars:
            return round(float(price_val) * 100)
        return int(price_val)
    except (TypeError, ValueError):
        return None


def _orderbook_levels_to_rows(levels: list, in_dollars: bool) -> list[tuple[str, str]]:
    """Convert API orderbook levels to (price_cent_str, quantity_str) rows.
    levels: from 'yes'/'no' (legacy [cents, count]) or 'yes_dollars'/'no_dollars' ([dollars_str, count_str]).
    """
    rows = []
    for level in levels or []:
        if isinstance(level, (list, tuple)) and len(level) >= 2:
            price_val, size_val = level[0], level[1]
            if in_dollars:
                try:
                    price_cents = round(float(price_val) * 100)
                except (TypeError, ValueError):
                    price_cents = 0
                size_str = str(size_val).rstrip("0").rstrip(".") if size_val else "0"
            else:
                price_cents = int(price_val) if price_val is not None else 0
                size_str = str(int(size_val)) if size_val else "0"
            if size_val and float(size_val) != 0:
                rows.append((f"{price_cents}¢", size_str))
        elif isinstance(level, dict):
            price = level.get("price") or level.get("yes_price") or level.get("no_price") or 0
            size = level.get("count") or level.get("remaining_count") or level.get("quantity") or 0
            if size and size != 0:
                rows.append((f"{price}¢", str(size)))
    return rows


def _orderbook_is_empty(data: dict) -> bool:
    """True if the orderbook response has no bid levels on either side."""
    fp = data.get("orderbook_fp") or {}
    legacy = data.get("orderbook") or {}
    yes_fp = fp.get("yes_dollars") or []
    no_fp = fp.get("no_dollars") or []
    yes_legacy = legacy.get("yes") or []
    no_legacy = legacy.get("no") or []
    return (not yes_fp and not no_fp and not yes_legacy and not no_legacy)


def _orderbook_series_error(series_ticker: str, example_tickers: list[str]) -> None:
    """Print an error when user passed a series ticker but orderbook requires a market ticker."""
    console.print("[red]Error:[/red] Orderbook requires a [bold]market ticker[/bold], not a series ticker.")
    console.print(f"  [cyan]{series_ticker}[/cyan] is a series; each market in the series has its own ticker.")
    if example_tickers:
        examples = ", ".join(example_tickers[:3])
        console.print(f"  Use a specific market ticker, e.g.: [bold]{examples}[/bold]")
        console.print("  Run [bold]kalshi search " + series_ticker + "[/bold] to list markets and their tickers.")
    else:
        console.print("  This series has no open markets, or the ticker was not found.")
    console.print()


def _get_open_markets_for_series(series_ticker: str, limit: int = 10) -> list[dict]:
    """Return list of open markets for the given series ticker (empty on error or not a series)."""
    try:
        r = api("GET", f"markets?series_ticker={series_ticker.upper()}&status=open&limit={limit}")
        return r.get("markets") or []
    except (ApiError, typer.Exit):
        return []


def _resolve_series_to_single_market(series_ticker: str) -> tuple[str | None, list[str]]:
    """If series has exactly one open market, return (that_ticker, []). Else return (None, example_tickers)."""
    markets = _get_open_markets_for_series(series_ticker, limit=5)
    tickers = [m["ticker"] for m in markets if m.get("ticker")]
    if len(tickers) == 1:
        return (tickers[0], [])
    return (None, tickers)


@app.command()
def orderbook(
    ticker: str = typer.Argument(help="Market ticker (use a specific market, not a series, when multiple exist)"),
    raw: bool = typer.Option(False, "--raw", help="Print raw API response JSON"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON (or set KALSHI_OUTPUT=json)"),
):
    """Show the orderbook for a market with best bid AND ask. With a series that has exactly one active market, shows that market's orderbook."""
    as_json = json_mode(flag=json_output)
    if as_json or raw:
        # Fast-path: JSON mode skips the extra markets/{ticker} enrichment call
        # that the human path uses for ask prices (not present in orderbook payload).
        # Agents that need the ask can call `kalshi detail <ticker>` separately.
        ticker = ticker.upper()
        try:
            data = api("GET", f"markets/{ticker}/orderbook")
        except ApiError as e:
            if e.status_code == 404:
                single, _ = _resolve_series_to_single_market(ticker)
                if single:
                    ticker = single
                    data = api("GET", f"markets/{ticker}/orderbook")
                else:
                    handle_api_error(e)
            else:
                handle_api_error(e)
        if raw and not as_json:
            sys.stdout.write(json.dumps(data, indent=2) + "\n")
            sys.stdout.flush()
            return
        output_json({"ticker": ticker, "orderbook": data})
    ticker = ticker.upper()
    try:
        data = api("GET", f"markets/{ticker}/orderbook")
    except ApiError as e:
        if e.status_code == 404:
            single, examples = _resolve_series_to_single_market(ticker)
            if single:
                ticker = single
                data = api("GET", f"markets/{ticker}/orderbook")
                console.print(f"[dim]Resolved series to market: [cyan]{ticker}[/cyan][/dim]\n")
            else:
                _orderbook_series_error(ticker, examples)
                raise typer.Exit(1)
        else:
            handle_api_error(e)

    if raw:
        print(json.dumps(data, indent=2))
        return

    # If orderbook came back empty, ticker might be a series
    if _orderbook_is_empty(data):
        single, examples = _resolve_series_to_single_market(ticker)
        if single:
            ticker = single
            data = api("GET", f"markets/{ticker}/orderbook")
            console.print(f"[dim]Resolved series to market: [cyan]{ticker}[/cyan][/dim]\n")
        elif examples or _get_open_markets_for_series(ticker, limit=1):
            _orderbook_series_error(ticker, examples or [])
            raise typer.Exit(1)

    # Get orderbook data (bids only)
    fp = data.get("orderbook_fp") or {}
    legacy = data.get("orderbook") or {}
    if fp.get("yes_dollars") is not None or fp.get("no_dollars") is not None:
        yes_levels = fp.get("yes_dollars") or []
        no_levels = fp.get("no_dollars") or []
        in_dollars = True
    else:
        yes_levels = legacy.get("yes") or []
        no_levels = legacy.get("no") or []
        in_dollars = False

    # Fetch market detail for best bid/ask prices
    market_data = api("GET", f"markets/{ticker}")
    m = market_data.get("market", market_data)

    # If orderbook is empty but we have market detail, still show what we can
    if not yes_levels and not no_levels:
        yes_bid = to_cents(m.get("yes_bid_dollars"))
        yes_ask = to_cents(m.get("yes_ask_dollars"))
        no_bid = to_cents(m.get("no_bid_dollars"))
        no_ask = to_cents(m.get("no_ask_dollars"))

        console.print()
        console.print(f"[bold]{ticker}[/bold] (no active orderbook)")
        console.print()

        if yes_bid is not None:
            console.print(f"  [green]YES[/green]: Bid ${yes_bid/100:.2f} | Ask ${yes_ask/100:.2f}" if yes_ask else f"  [green]YES[/green]: Bid ${yes_bid/100:.2f}")
        if no_bid is not None:
            console.print(f"  [red]NO[/red]:   Bid ${no_bid/100:.2f} | Ask ${no_ask/100:.2f}" if no_ask else f"  [red]NO[/red]:   Bid ${no_bid/100:.2f}")
        console.print()
        return

    # Parse best bid/ask from market detail (convert to cents)
    yes_bid = to_cents(m.get("yes_bid_dollars"))
    yes_ask = to_cents(m.get("yes_ask_dollars"))
    no_bid = to_cents(m.get("no_bid_dollars"))
    no_ask = to_cents(m.get("no_ask_dollars"))

    # Get volume at best bid level
    def get_vol_at_price(levels, target_cents):
        if not levels or target_cents is None:
            return None
        for level in levels:
            price = int(float(level[0]) * 100) if in_dollars else level[0]
            if price == target_cents:
                return level[1]
        return None

    yes_bid_vol = get_vol_at_price(yes_levels, yes_bid)
    yes_ask_vol = get_vol_at_price(yes_levels, yes_ask)
    no_bid_vol = get_vol_at_price(no_levels, no_bid)
    no_ask_vol = get_vol_at_price(no_levels, no_ask)

    console.print()

    # Format helpers
    def fmt_price(cents):
        if cents is None:
            return "—"
        return f"${cents/100:.2f}"

    def fmt_vol(vol):
        if vol is None:
            return "—"
        # Handle string volumes like '289672.00'
        try:
            v = float(str(vol).replace(',', ''))
            return f"{int(v):,}" if v == int(v) else str(vol)
        except (TypeError, ValueError):
            return str(vol)

    # YES table
    yes_title = f"{ticker} — YES"
    yes_table = Table(title=yes_title, box=box.SIMPLE, style="green")
    yes_table.add_column("Price", justify="right")
    yes_table.add_column("Quantity", justify="right")

    yes_rows = _orderbook_levels_to_rows(yes_levels, in_dollars)
    if not yes_rows:
        yes_table.add_row("—", "no bids")
    for price_str, qty_str in yes_rows:
        yes_table.add_row(price_str, qty_str)

    # NO table
    no_title = f"{ticker} — NO"
    no_table = Table(title=no_title, box=box.SIMPLE, style="red")
    no_table.add_column("Price", justify="right")
    no_table.add_column("Quantity", justify="right")

    no_rows = _orderbook_levels_to_rows(no_levels, in_dollars)
    if not no_rows:
        no_table.add_row("—", "no bids")
    for price_str, qty_str in no_rows:
        no_table.add_row(price_str, qty_str)

    # Print header with bid/ask info
    console.print(f"[bold]{ticker}[/bold]")
    console.print()

    if yes_bid is not None:
        console.print(f"  [green]YES[/green]: [bold]Bid {fmt_price(yes_bid)}[/bold] ({fmt_vol(yes_bid_vol)}) | [bold]Ask {fmt_price(yes_ask)}[/bold] ({fmt_vol(yes_ask_vol)})")

    if no_bid is not None:
        console.print(f"  [red]NO[/red]:   [bold]Bid {fmt_price(no_bid)}[/bold] ({fmt_vol(no_bid_vol)}) | [bold]Ask {fmt_price(no_ask)}[/bold] ({fmt_vol(no_ask_vol)})")

    console.print()
    console.print(Columns([yes_table, no_table], equal=True, expand=True))



@app.command()
def balance(json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON")):
    """Show account balance"""
    try:
        data = api("GET", "portfolio/balance")
    except ApiError as e:
        handle_api_error(e)
    if json_output:
        output_json({"balance": data.get("balance", 0) / 100})

    b = data.get("balance", 0) / 100
    console.print(f"\n[bold]Balance:[/bold] ${b:.2f}\n")


def _position_outcome_label(p: dict) -> str:
    """Derive the best outcome label for a position.

    Uses subtitle from the market API (fetched by _enrich_positions).
    - Binary markets (subtitle name appears in title): plain Yes / No
    - Multi-outcome markets (subtitle is a distinct pick): show the outcome name
    """
    pos = p.get("position", 0)
    sub = (p.get("yes_sub_title", "") or p.get("no_sub_title", "") or "").strip()
    title = (p.get("title", "") or "").strip()

    # If subtitle is generic or empty, or the subtitle is already in the title
    # (e.g. "Gavin Newsom" in "Will Gavin Newsom be..."), it's binary → Yes/No
    is_binary = (
        not sub
        or _is_generic_subtitle(sub)
        or sub.lower() in title.lower()
    )

    if pos < 0:
        return "[red]No[/red]"

    if is_binary:
        return "[green]Yes[/green]"

    # Multi-outcome: show the actual pick
    return f"[green]{sub}[/green]"


def _enrich_positions(ps: list) -> list:
    """Fetch market details for each position to get title/subtitle/expiry/price.
    Returns enriched position dicts with all fields needed for display.
    """
    enriched = []
    for p in ps:
        ticker = p.get("ticker", "")
        merged = dict(p)
        if ticker:
            try:
                data = api("GET", "markets/" + ticker)
                m = data.get("market", data)
                merged["title"] = m.get("title", "")
                merged["yes_sub_title"] = m.get("yes_sub_title", "")
                merged["no_sub_title"] = m.get("no_sub_title", "")
                merged["expiration_time"] = _market_expiry_ts(m)
                merged["yes_bid_dollars"] = m.get("yes_bid_dollars")
                merged["no_bid_dollars"] = m.get("no_bid_dollars")
            except ApiError:
                pass
        enriched.append(merged)
    return enriched


def _position_current_value(p: dict) -> float:
    """Current market value of position (qty × bid price for the side held)."""
    try:
        pos = p.get("position", 0)
        if pos > 0:
            price = float(p.get("yes_bid_dollars", 0) or 0)
        else:
            price = float(p.get("no_bid_dollars", 0) or 0)
        return abs(pos) * price
    except (ValueError, TypeError):
        return 0.0


def _position_return_pct(p: dict) -> str:
    """Calculate unrealized return % for a position.
    Return = (current_value - cost) / cost * 100
    """
    try:
        cost = float(p.get("total_traded_dollars", 0) or 0)
        if cost <= 0:
            return "—"
        current_value = _position_current_value(p)
        pnl = current_value - cost
        pct = (pnl / cost) * 100
        if pct >= 0:
            return f"[green]+{pct:.0f}%[/green]"
        return f"[red]{pct:.0f}%[/red]"
    except (ValueError, TypeError, ZeroDivisionError):
        return "—"


@app.command()
def positions(
    expiring: bool = typer.Option(False, "--expiring", "-e", help="Sort by soonest expiry"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Show current positions"""
    try:
        data = api("GET", "portfolio/positions")
    except ApiError as e:
        handle_api_error(e)
    # Kalshi returns positions in two shapes on /portfolio/positions:
    #   - market_positions (legacy): keyed by ticker, has 'position' field
    #   - event_positions (current): keyed by event_ticker, has 'total_cost_shares_fp'
    # Newer accounts only populate event_positions. Merge both so the CLI doesn't
    # silently drop live positions.
    ps = list(data.get("market_positions", []))
    for ep in data.get("event_positions", []):
        ticker = ep.get("event_ticker")
        if not ticker:
            continue
        shares = float(ep.get("total_cost_shares_fp", 0) or 0)
        if shares == 0:
            continue
        # Synthesize a market_positions-shaped entry so the rest of the pipeline works.
        ps.append({
            "ticker": ticker,
            "position": int(shares),
            "total_traded_dollars": ep.get("total_cost_dollars", 0),
            "fees_paid_dollars": ep.get("fees_paid_dollars", 0),
            "realized_pnl_dollars": ep.get("realized_pnl_dollars", 0),
            "event_ticker": ticker,
            "synthetic_from_event": True,
        })
    ps = [p for p in ps if p.get("position", 0) != 0]
    if not ps:
        if json_mode(flag=json_output):
            output_json([])
        console.print("[dim]No open positions[/dim]")
        raise typer.Exit()

    with console.status("[dim]Loading position details...[/dim]"):
        ps = _enrich_positions(ps)

    if expiring:
        ps.sort(key=lambda p: p.get("expiration_time", "") or "9999")

    if json_output:
        output = []
        for p in ps:
            title = _normalize_title(p.get("title", "") or "")
            qty = abs(p.get("position", 0))
            total_cost = float(p.get("total_traded_dollars", 0) or 0)
            cost_per_share = (total_cost / qty) if qty > 0 else 0

            if p.get("position", 0) > 0:
                val_per_share = float(p.get("yes_bid_dollars", 0) or 0)
            else:
                val_per_share = float(p.get("no_bid_dollars", 0) or 0)

            current_value = _position_current_value(p)

            output.append({
                "ticker": p.get("ticker", ""),
                "title": title,
                "side": "YES" if p.get("position", 0) > 0 else "NO",
                "quantity": qty,
                "cost_per_share": round(cost_per_share, 2),
                "current_value_per_share": round(val_per_share, 2),
                "total_value": round(current_value, 2),
                "return_pct": round((val_per_share - cost_per_share) / cost_per_share * 100 if cost_per_share > 0 else 0, 1),
                "expiration": p.get("expiration_time", "")[:10] if p.get("expiration_time") else ""
            })

        output_json(output)

    # Original table output
    t = Table(title="Current Positions", box=box.ROUNDED)
    t.add_column("Title", max_width=40)
    t.add_column("Side", max_width=25)
    t.add_column("Qty", justify="right")
    t.add_column("Cost/Sh", justify="right", style="yellow")
    t.add_column("Val/Sh", justify="right")
    t.add_column("Return", justify="right")
    t.add_column("Value", justify="right")
    t.add_column("Expires", style="dim")
    t.add_column("Ticker", style="cyan", overflow="fold")

    for p in ps:
        title = _normalize_title(p.get("title", "") or "")
        side_str = _position_outcome_label(p)
        qty = abs(p.get("position", 0))
        total_cost = float(p.get("total_traded_dollars", 0) or 0)
        cost_per_share = (total_cost / qty) if qty > 0 else 0

        if p.get("position", 0) > 0:
            val_per_share = float(p.get("yes_bid_dollars", 0) or 0)
        else:
            val_per_share = float(p.get("no_bid_dollars", 0) or 0)

        current_value = _position_current_value(p)

        t.add_row(
            title,
            side_str,
            str(qty),
            f"${cost_per_share:.2f}" if cost_per_share > 0 else "—",
            f"${val_per_share:.2f}" if val_per_share > 0 else "—",
            _position_return_pct(p),
            fmt_dollars(current_value),
            fmt_expiry(p.get("expiration_time", "")),
            p.get("ticker", ""),
        )
    console.print(t)


@app.command()
def orders(json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON")):
    """Show open (resting) orders"""
    try:
        data = api("GET", "portfolio/orders?status=resting")
    except ApiError as e:
        handle_api_error(e)
    order_list = data.get("orders", [])
    if not order_list:
        if json_mode(flag=json_output):
            output_json([])
        console.print("[dim]No open orders[/dim]")
        raise typer.Exit()

    t = Table(title="Open Orders", box=box.ROUNDED)
    t.add_column("Order ID", style="dim")
    t.add_column("Ticker", style="cyan", overflow="fold")
    t.add_column("Action")
    t.add_column("Side", justify="center")
    t.add_column("Price", justify="right")
    t.add_column("Qty", justify="right")
    t.add_column("Remaining", justify="right")

    # Output as JSON if global flag is set
    if json_output:
        output = [{
            "order_id": o.get("order_id", ""),
            "ticker": o.get("ticker", ""),
            "action": o.get("action", ""),
            "side": o.get("side", ""),
            "price": o.get("yes_price", 0) if o.get("side") == "yes" else o.get("no_price", 0),
            "quantity": o.get("count", 0),
            "remaining": o.get("remaining_count", 0),
            "status": o.get("status", "")
        } for o in order_list]
        output_json(output)

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


def _build_v2_order_body(
    *,
    ticker: str,
    side: str,
    count: int,
    price_cents: int,
    action: str,
    client_order_id: str,
) -> dict:
    """Build a V2 create-order request body from a (user-friendly) yes/no + cents API.

    Maps the legacy `action` + yes/no `side` into V2's book-side `bid`/`ask` quoted
    from the YES leg, per Kalshi docs:
        BookSide: "For event markets, this refers to the YES leg only:
                   `bid` means buy YES, `ask` means sell YES."

    Translation:
        action=buy,  side=yes → book_side=bid,  yes_dollars = price_cents / 100
        action=buy,  side=no  → book_side=ask, yes_dollars = (100 - price_cents) / 100
        action=sell, side=yes → book_side=ask, yes_dollars = price_cents / 100
        action=sell, side=no  → book_side=bid, yes_dollars = (100 - price_cents) / 100

    V2 also requires `time_in_force` and `self_trade_prevention_type`. We use
    `good_till_canceled` (rest until cancelled or filled) and `taker_at_cross`
    (cancel this order if it would self-trade against another order from us).
    """
    if action == "buy":
        book_side = "bid" if side == "yes" else "ask"
    else:  # sell
        book_side = "ask" if side == "yes" else "bid"

    yes_dollars = price_cents / 100.0 if side == "yes" else (100 - price_cents) / 100.0

    return {
        "ticker": ticker.upper(),
        "client_order_id": client_order_id,
        "side": book_side,
        "count": f"{count:.2f}",
        "price": f"{yes_dollars:.4f}",
        "time_in_force": "good_till_canceled",
        "self_trade_prevention_type": "taker_at_cross",
        "post_only": False,
        "cancel_order_on_pause": False,
        "reduce_only": False,
    }


@app.command()
def buy(
    ticker: str = typer.Argument(help="Market ticker"),
    count: int = typer.Argument(help="Number of contracts"),
    price: int = typer.Argument(help="Price in cents (e.g. 68 for $0.68)"),
    side: str = typer.Option("yes", "--side", "-s", help="Contract side: 'yes' or 'no'"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
    yes_flag: bool = typer.Option(False, "--yes", "-y", help="Alias for --force (agent-friendly)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be placed without calling the API"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON (or set KALSHI_OUTPUT=json)"),
):
    """Place a limit buy order"""
    as_json = json_mode(flag=json_output)
    if yes_flag:
        force = True
    side = side.lower()
    if side not in ("yes", "no"):
        if as_json:
            output_json({"error": {"code": "BAD_REQUEST", "message": "side must be 'yes' or 'no'"}}, exit_code=6)
        console.print("[red]Error:[/red] --side must be 'yes' or 'no'")
        raise typer.Exit(1)

    cost = count * price
    api_body = _build_v2_order_body(
        ticker=ticker,
        side=side,
        count=count,
        price_cents=price,
        action="buy",
        client_order_id=str(uuid.uuid4()),
    )

    if dry_run:
        if as_json:
            output_json({"dry_run": True, "request_body": api_body, "max_cost_cents": cost})
        console.print(
            f"\n  [dim][dry-run][/dim] Buy [bold]{count}x {side.upper()}[/bold] on "
            f"[cyan]{ticker.upper()}[/cyan] @ {price}¢ each"
        )
        console.print(f"  Max cost: [yellow]${cost / 100:.2f}[/yellow]")
        console.print(f"  Request body: [dim]{json.dumps(api_body)}[/dim]")
        return

    if not as_json:
        console.print(
            f"\n  Buy [bold]{count}x {side.upper()}[/bold] on "
            f"[cyan]{ticker.upper()}[/cyan] @ {price}¢ each"
        )
        console.print(f"  Max cost: [yellow]${cost / 100:.2f}[/yellow]\n")
    if not force and not typer.confirm("Confirm order?"):
        if as_json:
            output_json({"cancelled": True, "reason": "user_declined"})
        console.print("[dim]Cancelled[/dim]")
        raise typer.Exit()

    try:
        res = api("POST", "portfolio/events/orders", body=api_body)
    except ApiError as e:
        handle_api_error(e)
    if as_json:
        output_json(res)
    order_id = res.get("order_id", "unknown")
    fill_count = res.get("fill_count", "0.00")
    console.print(f"[bold green]Order placed![/bold green] ID: {order_id} (filled: {fill_count})")


@app.command()
def sell(
    ticker: str = typer.Argument(help="Market ticker"),
    count: int = typer.Argument(help="Number of contracts"),
    price: int = typer.Argument(help="Price in cents"),
    side: str = typer.Option("yes", "--side", "-s", help="Contract side: 'yes' or 'no'"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
    yes_flag: bool = typer.Option(False, "--yes", "-y", help="Alias for --force (agent-friendly)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be placed without calling the API"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON (or set KALSHI_OUTPUT=json)"),
):
    """Place a limit sell order"""
    as_json = json_mode(flag=json_output)
    if yes_flag:
        force = True
    side = side.lower()
    if side not in ("yes", "no"):
        if as_json:
            output_json({"error": {"code": "BAD_REQUEST", "message": "side must be 'yes' or 'no'"}}, exit_code=6)
        console.print("[red]Error:[/red] --side must be 'yes' or 'no'")
        raise typer.Exit(1)

    proceeds = count * price
    api_body = _build_v2_order_body(
        ticker=ticker,
        side=side,
        count=count,
        price_cents=price,
        action="sell",
        client_order_id=str(uuid.uuid4()),
    )

    if dry_run:
        if as_json:
            output_json({"dry_run": True, "request_body": api_body, "expected_proceeds_cents": proceeds})
        console.print(
            f"\n  [dim][dry-run][/dim] Sell [bold]{count}x {side.upper()}[/bold] on "
            f"[cyan]{ticker.upper()}[/cyan] @ {price}¢ each"
        )
        console.print(f"  Expected proceeds: [yellow]${proceeds / 100:.2f}[/yellow]")
        console.print(f"  Request body: [dim]{json.dumps(api_body)}[/dim]")
        return

    if not as_json:
        console.print(
            f"\n  Sell [bold]{count}x {side.upper()}[/bold] on "
            f"[cyan]{ticker.upper()}[/cyan] @ {price}¢ each"
        )
        console.print(f"  Expected proceeds: [yellow]${proceeds / 100:.2f}[/yellow]\n")
    if not force and not typer.confirm("Confirm order?"):
        if as_json:
            output_json({"cancelled": True, "reason": "user_declined"})
        console.print("[dim]Cancelled[/dim]")
        raise typer.Exit()

    try:
        res = api("POST", "portfolio/events/orders", body=api_body)
    except ApiError as e:
        handle_api_error(e)
    if as_json:
        output_json(res)
    order_id = res.get("order_id", "unknown")
    fill_count = res.get("fill_count", "0.00")
    console.print(f"[bold green]Sell order placed![/bold green] ID: {order_id} (filled: {fill_count})")


@app.command()
def cancel(
    order_id: str = typer.Argument(help="Order ID to cancel"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON (or set KALSHI_OUTPUT=json)"),
):
    """Cancel an open order. Always non-interactive (cancels are easily reversible by re-placing)."""
    as_json = json_mode(flag=json_output)
    try:
        res = api("DELETE", f"portfolio/events/orders/{order_id}")
    except ApiError as e:
        handle_api_error(e)
    if as_json:
        output_json({"cancelled": True, "order_id": order_id, "response": res})
    reduced_by = (res or {}).get("reduced_by") if isinstance(res, dict) else None
    extra = f" (reduced_by={reduced_by})" if reduced_by else ""
    console.print(f"[bold green]Order {order_id} cancelled[/bold green]{extra}")


@app.command(name="setup-shell")
def setup_shell():
    """Add KALSHI_ACCESS_KEY to your shell config and create RSA private key template"""
    load_env()
    key_id = os.getenv("KALSHI_ACCESS_KEY")

    # Create ~/.kalshi directory and files if they don't exist
    kalshi_dir = os.path.expanduser("~/.kalshi")
    os.makedirs(kalshi_dir, exist_ok=True)

    env_path = os.path.join(kalshi_dir, '.env')
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write(f'KALSHI_ACCESS_KEY={key_id or "your_access_key_here"}\n')
        console.print(f"[green]Created[/green] {env_path}")
    elif key_id:
        with open(env_path) as f:
            existing = f.read()
        if key_id not in existing:
            with open(env_path, "a") as f:
                f.write(f'KALSHI_ACCESS_KEY={key_id}\n')
        console.print(f"[green]Updated[/green] {env_path}")

    key_path = os.path.join(kalshi_dir, 'private_key.pem')
    if not os.path.exists(key_path):
        with open(key_path, "w") as f:
            f.write("# Place your RSA private key here\n# Get it from: https://kalshi.com/api\n-----BEGIN RSA PRIVATE KEY-----\nyour_private_key_here\n-----END RSA PRIVATE KEY-----\n")
        console.print(f"[yellow]Created[/yellow] {key_path} — paste your RSA private key here")

    if not key_id:
        console.print("[red]Error:[/red] No KALSHI_ACCESS_KEY found. Add it to ~/.kalshi/.env")
        raise typer.Exit(1)

    line = f'export KALSHI_ACCESS_KEY="{key_id}"\n'
    bashrc = os.path.expanduser("~/.bashrc")
    zshrc = os.path.expanduser("~/.zshrc")

    updated = False
    for rc in [bashrc, zshrc]:
        if os.path.exists(rc):
            with open(rc) as f:
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
        console.print("\n✅ Setup complete!")
        console.print("   - ~/.kalshi/.env has your API key")
        console.print("   - ~/.kalshi/private_key.pem needs your RSA key")
        console.print("\nRestart your terminal or run [bold]source ~/.bashrc[/bold]")
    else:
        console.print("[dim]No shell config files found (~/.bashrc or ~/.zshrc)[/dim]")


if __name__ == "__main__":
    app(prog_name="kalshi")
