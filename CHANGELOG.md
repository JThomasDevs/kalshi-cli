# Changelog

## 1.2.75 — 2026-08-24

### Added

- **Agent-friendly mode** — every command now supports `--json` / `-j` (or globally via `KALSHI_OUTPUT=json` env var). The following commands that previously lacked JSON output now have it: `search`, `series`, `detail`, `orderbook`, `buy`, `sell`, `cancel`. When JSON mode is active, output is plain JSON to stdout (no Rich tables, no color codes), suitable for piping into another tool.
- **`--yes` / `-y` flag on `buy` and `sell`** — agent-friendly alias for `--force`. Lets agents skip confirmation prompts without remembering the longer flag name.
- **`--dry-run` on `buy` and `sell`** — prints the request body and cost calculation without calling the API. Useful for backtests and validation.
- **`output_json(data, exit_code)` helper** — writes JSON to stdout, flushes, and exits with the given code. Used by every JSON-mode code path so output is consistent and pipe-safe.
- **`json_mode(flag)` helper** — resolves whether a command should produce JSON output. True if either `--json` was passed or `KALSHI_OUTPUT=json` is set.
- **Stable error envelope** — when `KALSHI_OUTPUT=json`, errors come back as `{"error": {"code": "...", "status": N, "message": "..."}}` instead of free text on stderr. Codes: `AUTH_FAILED` (401), `FORBIDDEN` (403), `NOT_FOUND` (404), `RATE_LIMITED` (429), `INSUFFICIENT_FUNDS` / `INVALID_TICKER` / `INVALID_PRICE` / `INVALID_QUANTITY` / `BAD_REQUEST` (4xx), `SERVER_ERROR` (5xx), `API_ERROR` (fallback).
- **Distinct exit codes per error category** — 0 success, 1 generic, 2 auth, 3 forbidden, 4 not_found, 5 rate_limited, 6 bad_request, 7 server. Agents can `if [ $? -eq 5 ]` to detect rate-limit backoff without parsing stderr.
- **42 tests** (was 26). Added coverage for `_classify_error`, `_exit_code_for_status`, `json_mode`, env-var flag override.

### Changed

- **`orderbook` JSON-mode fast-path** — skips the redundant `markets/{ticker}` enrichment call that the human-path uses for ask prices. The orderbook payload doesn't contain asks; agents that need them should call `kalshi detail <ticker> --json` separately. Saves ~50–150ms per orderbook fetch in agent mode.
- **Removed local `import json` shadowing** in 5 functions that now use the module-level import added in this release.

### Notes for existing users

- No breaking changes. Default behavior unchanged for human users.
- New `KALSHI_OUTPUT=json` env var is opt-in.
- `buy`/`sell` confirmation prompts still apply unless `--force` or `--yes` is passed.

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