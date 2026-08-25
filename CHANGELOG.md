# Changelog

## 1.2.74 — 2026-08-24

### Fixed

- **Orderbook crash on empty book** — `to_cents()` is now defined at module scope. Previously, it was a nested function inside one branch of `display_orderbook`, but was referenced from a different (earlier) branch, causing a `NameError` whenever the orderbook came back empty but market detail was available. Hoisted to top-level helper.
- **API base URL** — switched from legacy shared host `api.elections.kalshi.com` to the recommended `external-api.kalshi.com`. The legacy host still works but is not the recommended path for API traders per current Kalshi docs. Override with `KALSHI_BASE_URL` env var if needed.
- **Bare `except:` clauses** — replaced with `except (TypeError, ValueError)` in two places (`to_cents`, volume formatter).
- **File handle leak** — `setup-shell` no longer reads `~/.kalshi/.env` via `open(...).read()` (no context manager); now uses `with open(...)`.
- **Ticker-segment expiry parser** — the old `_parse_expiry_from_ticker` had wrong slice indices (assumed `DDMMMHHMMSS` but real Kalshi tickers like `26AUG242345` for KXBTC15M use a different encoding). The parser silently returned `None` for every real ticker. Replaced with a stub returning `None`; expiry now relies exclusively on the API fields (`expected_expiration_time` / `close_time`), which Kalshi reliably returns for every market. The bug was inert because API fields take precedence in `_market_expiry_ts`, but the dead branch was removed for clarity.

### Changed

- **Version drift** — `pyproject.toml` is now in sync with `package.json` (was stuck at 1.2.3 while npm was at 1.2.73).
- **BASE_URL is configurable** — set `KALSHI_BASE_URL` to override (e.g. for demo: `https://external-api.demo.kalshi.co`).
- **`_market_expiry_ts` simplified** — removed the unreachable ticker-parsing fallback; the function now just returns the first available API field.

### Added

- **`tests/`** — pytest suite (26 tests) covering formatters, signing, query-string stripping, expiry parsing, and min-odds filter. `test_api_strips_query_for_signature` is a regression test for the Kalshi docs requirement that signatures strip query params.
- **`ruff.toml`** — lint config + CI step.
- **`CHANGELOG.md`** — this file.

## Earlier

See git history. Versions 1.2.0–1.2.73 were npm-only releases (pyproject.toml drifted since 1.2.3).