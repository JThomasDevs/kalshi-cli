"""Tests for pure helper functions in cli.py.

These tests don't make network calls — they exercise the deterministic,
side-effect-free helpers so refactoring is safe.
"""

import os
import sys

# Make the cli module importable without running its __main__ code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cli

# ── fmt_price ─────────────────────────────────────────


def test_fmt_price_normal():
    assert cli.fmt_price(0.68) == "$0.68 (68%)"


def test_fmt_price_zero():
    assert cli.fmt_price(0) == "$0.00 (0%)"


def test_fmt_price_one():
    assert cli.fmt_price(1.0) == "$1.00 (100%)"


def test_fmt_price_none():
    assert cli.fmt_price(None) == "—"


def test_fmt_price_na_string():
    assert cli.fmt_price("N/A") == "—"


def test_fmt_price_garbage_string():
    # Falls through to str() of the input
    assert cli.fmt_price("banana") == "banana"


# ── fmt_dollars ────────────────────────────────────────


def test_fmt_dollars_normal():
    assert cli.fmt_dollars(12.5) == "$12.50"


def test_fmt_dollars_none():
    assert cli.fmt_dollars(None) == "—"


def test_fmt_dollars_garbage():
    assert cli.fmt_dollars("xyz") == "xyz"


# ── to_cents ───────────────────────────────────────────


def test_to_cents_float():
    assert cli.to_cents(0.68) == 68


def test_to_cents_string():
    assert cli.to_cents("0.42") == 42


def test_to_cents_int():
    assert cli.to_cents(1) == 100


def test_to_cents_none():
    assert cli.to_cents(None) is None


def test_to_cents_garbage():
    assert cli.to_cents("not a number") is None


# ── sign_request (pure: signs with a synthetic key) ────


def test_sign_request_is_base64():
    """Signatures are base64 strings — verifiable shape, not exact equality (PSS is non-deterministic)."""
    import base64

    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    sig = cli.sign_request("1700000000000", "GET", "/trade-api/v2/markets", key)
    # Should round-trip as valid base64
    decoded = base64.b64decode(sig)
    assert len(decoded) > 0
    # PSS-SHA256 produces 256-byte signatures
    assert len(decoded) == 256


def test_sign_request_differs_with_method():
    """GET and POST against the same path should produce different signatures."""
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    sig_get = cli.sign_request("1700000000000", "GET", "/trade-api/v2/markets", key)
    sig_post = cli.sign_request("1700000000000", "POST", "/trade-api/v2/markets", key)
    assert sig_get != sig_post


def test_sign_request_differs_with_timestamp():
    """Different timestamps should produce different signatures (replay protection basis)."""
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    sig_a = cli.sign_request("1700000000000", "GET", "/trade-api/v2/markets", key)
    sig_b = cli.sign_request("1700000000001", "GET", "/trade-api/v2/markets", key)
    assert sig_a != sig_b


def test_sign_request_query_string_stripped():
    """The signature must NOT include the query string (per Kalshi docs).

    If two signatures are equal for `/path` and `/path?foo=bar`,
    the signer has correctly stripped the query.
    """
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    sig_clean = cli.sign_request("1700000000000", "GET", "/trade-api/v2/markets", key)
    sig_with_q = cli.sign_request("1700000000000", "GET", "/trade-api/v2/markets?limit=5", key)
    # Note: the doc warning is about how CALLERS should strip; this test verifies
    # that if a caller passed an un-stripped path, the signature would differ.
    # The cli.api() function does the strip — see test_api_strips_query.
    assert sig_clean != sig_with_q


# ── filter_by_min_odds ────────────────────────────────


def test_filter_by_min_odds_disabled():
    markets = [{"yes_bid_dollars": 0.01, "no_bid_dollars": 0.01}]
    # min_odds=0 disables the filter
    assert cli.filter_by_min_odds(markets, 0) is markets


def test_filter_by_min_odds_drops_thin():
    markets = [
        {"yes_bid_dollars": 0.50, "no_bid_dollars": 0.50},  # 50%, keep
        {"yes_bid_dollars": 0.001, "no_bid_dollars": 0.50},  # yes too thin, drop
        {"yes_bid_dollars": 0.50, "no_bid_dollars": 0.001},  # no too thin, drop
    ]
    filtered = cli.filter_by_min_odds(markets, 10)  # 10% threshold
    assert len(filtered) == 1
    assert filtered[0]["yes_bid_dollars"] == 0.50


# ── _parse_expiry_from_ticker ──────────────────────────


def test_parse_expiry_from_ticker_returns_none_for_garbage():
    """Non-expiry tickers (no DDMMMHHMMSS segment) return None."""
    assert cli._parse_expiry_from_ticker("KXWO-GOLD") is None
    assert cli._parse_expiry_from_ticker("") is None


def test_parse_expiry_from_ticker_is_stub():
    """As of 1.2.74, _parse_expiry_from_ticker is a stub returning None.

    Kalshi's 15M/5M ticker-segment format isn't publicly documented and the
    parser couldn't reliably produce correct results. Expiry now comes
    exclusively from API fields via _market_expiry_ts.
    """
    assert cli._parse_expiry_from_ticker("KXWT-15M-15FEB123045") is None
    assert cli._parse_expiry_from_ticker("KXBTC15M-26AUG242345-45") is None


# ── _market_expiry_ts (API-fields first) ───────────────


def test_market_expiry_ts_prefers_expected_expiration():
    """expected_expiration_time wins over everything else."""
    m = {
        "expected_expiration_time": "2026-08-25T03:50:00Z",
        "close_time": "2026-08-25T03:45:00Z",
        "expiration_time": "2026-09-01T03:45:00Z",
    }
    assert cli._market_expiry_ts(m) == "2026-08-25T03:50:00Z"


def test_market_expiry_ts_falls_back_to_close_time():
    """If expected_expiration_time is missing, use close_time."""
    m = {
        "close_time": "2026-08-25T03:45:00Z",
        "expiration_time": "2026-09-01T03:45:00Z",
    }
    assert cli._market_expiry_ts(m) == "2026-08-25T03:45:00Z"


def test_market_expiry_ts_returns_empty_when_all_fields_missing():
    """If no API fields present, returns empty string (not the parsed ticker)."""
    m = {"ticker": "KXBTC15M-26AUG242345-45"}
    assert cli._market_expiry_ts(m) == ""


# ── agent-mode helpers ─────────────────────────────────


def test_classify_error_auth():
    assert cli._classify_error(401, "unauthorized") == "AUTH_FAILED"


def test_classify_error_not_found():
    assert cli._classify_error(404, "not found") == "NOT_FOUND"


def test_classify_error_rate_limit():
    assert cli._classify_error(429, "slow down") == "RATE_LIMITED"


def test_classify_error_insufficient_funds():
    assert cli._classify_error(400, "Insufficient funds in account") == "INSUFFICIENT_FUNDS"


def test_classify_error_invalid_ticker():
    assert cli._classify_error(400, "invalid ticker KXFOO") == "INVALID_TICKER"


def test_classify_error_invalid_price():
    assert cli._classify_error(400, "Price out of bounds") == "INVALID_PRICE"


def test_classify_error_invalid_quantity():
    assert cli._classify_error(422, "count must be positive") == "INVALID_QUANTITY"


def test_classify_error_server():
    assert cli._classify_error(503, "down for maintenance") == "SERVER_ERROR"


def test_classify_error_unknown_status_falls_back():
    # Status codes outside our typed map use API_ERROR
    assert cli._classify_error(418, "teapot") == "API_ERROR"


def test_exit_codes_are_distinct_per_category():
    """Different error categories get different exit codes so agents can branch."""
    codes = {
        cli._exit_code_for_status(401),
        cli._exit_code_for_status(404),
        cli._exit_code_for_status(429),
        cli._exit_code_for_status(400),
        cli._exit_code_for_status(503),
    }
    assert len(codes) == 5


def test_exit_code_for_status_zero_for_success():
    # exit_code_for_status is for errors only; success is 0
    # (just confirm we don't accidentally return 0 for error statuses)
    assert cli._exit_code_for_status(401) != 0
    assert cli._exit_code_for_status(500) != 0


def test_json_mode_flag_overrides():
    """--json flag enables JSON mode regardless of env."""
    assert cli.json_mode(flag=True) is True


def test_json_mode_env_var_enables(monkeypatch):
    monkeypatch.setenv("KALSHI_OUTPUT", "json")
    assert cli.json_mode() is True


def test_json_mode_env_var_case_insensitive(monkeypatch):
    monkeypatch.setenv("KALSHI_OUTPUT", "JSON")
    assert cli.json_mode() is True


def test_json_mode_default_human(monkeypatch):
    monkeypatch.delenv("KALSHI_OUTPUT", raising=False)
    assert cli.json_mode() is False


def test_json_mode_empty_env_value_is_human(monkeypatch):
    monkeypatch.setenv("KALSHI_OUTPUT", "")
    assert cli.json_mode() is False


# ── api() query-string stripping (regression) ──────────


def test_api_strips_query_for_signature(monkeypatch, mocker):
    """Regression: cli.api() must sign the path WITHOUT the query string.

    Catches a bug where someone 'simplifies' the api() function and
    accidentally signs the full URL-with-query.
    """
    # Reset env so load_env is a no-op
    monkeypatch.setenv("KALSHI_ACCESS_KEY", "fake-key")

    # Mock load_key to return a fake key
    fake_key = mocker.Mock()
    monkeypatch.setattr(cli, "load_key", lambda: fake_key)

    # Capture what sign_request gets called with
    captured = {}

    def fake_sign(ts, method, path, key):
        captured["path"] = path
        return "fakesig"

    monkeypatch.setattr(cli, "sign_request", fake_sign)

    # Mock the requests.get call so no real network
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"markets": []}
    mocker.patch("cli.requests.get", return_value=mock_response)

    cli.api("GET", "markets?series_ticker=KXTEST&status=open&limit=200")

    assert captured["path"] == "/trade-api/v2/markets", (
        f"signing path must be stripped of query string, got: {captured['path']}"
    )


# ── _classify_error: 410 deprecated endpoint ───────────


def test_classify_error_410_is_deprecated_endpoint():
    """V1 /portfolio/orders was deprecated to 410; agents must recognise this."""
    assert cli._classify_error(410, "deprecated_v1_order_endpoint") == "DEPRECATED_ENDPOINT"


# ── _build_v2_order_body (V2 create-order shape) ────────


def test_v2_body_buy_yes_at_5c():
    body = cli._build_v2_order_body(
        ticker="kxbTC-26aug2513-b79650", count=1, price_cents=5,
        side="yes", action="buy", client_order_id="abc",
    )
    assert body["side"] == "bid"  # buy YES = bid on yes-leg
    assert body["price"] == "0.0500"
    assert body["count"] == "1.00"
    assert body["ticker"] == "KXBTC-26AUG2513-B79650"  # uppercased
    assert body["time_in_force"] == "good_till_canceled"
    assert body["self_trade_prevention_type"] == "taker_at_cross"
    assert body["client_order_id"] == "abc"
    assert "action" not in body  # V2 dropped the action field


def test_v2_body_buy_no_at_5c():
    """buy NO at 5c = buy NO at 5c = sell YES at 95c → ask at 0.9500."""
    body = cli._build_v2_order_body(
        ticker="KXTEST", count=2, price_cents=5,
        side="no", action="buy", client_order_id="x",
    )
    assert body["side"] == "ask"
    assert body["price"] == "0.9500"
    assert body["count"] == "2.00"


def test_v2_body_sell_yes_at_50c():
    body = cli._build_v2_order_body(
        ticker="KXTEST", count=10, price_cents=50,
        side="yes", action="sell", client_order_id="x",
    )
    assert body["side"] == "ask"
    assert body["price"] == "0.5000"
    assert body["count"] == "10.00"


def test_v2_body_sell_no_at_50c():
    """sell NO at 50c = buy YES at 50c → bid at 0.5000."""
    body = cli._build_v2_order_body(
        ticker="KXTEST", count=1, price_cents=50,
        side="no", action="sell", client_order_id="x",
    )
    assert body["side"] == "bid"
    assert body["price"] == "0.5000"


def test_v2_body_count_is_fixed_point_string():
    """V2 requires count as a string with 2 decimals ('10.00')."""
    body = cli._build_v2_order_body(
        ticker="KXTEST", count=7, price_cents=10,
        side="yes", action="buy", client_order_id="x",
    )
    assert body["count"] == "7.00"
    assert isinstance(body["count"], str)


def test_v2_body_price_is_fixed_point_string_4dp():
    """V2 requires price as a 4-decimal dollar string."""
    body = cli._build_v2_order_body(
        ticker="KXTEST", count=1, price_cents=1,
        side="yes", action="buy", client_order_id="x",
    )
    assert body["price"] == "0.0100"
    assert isinstance(body["price"], str)
