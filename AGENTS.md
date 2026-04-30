# Repository Instructions

<!-- context7 -->
Use Context7 MCP to fetch current documentation whenever the user asks about a library, framework, SDK, API, CLI tool, or cloud service, even well-known ones like React, Next.js, Prisma, Express, Tailwind, Django, or Spring Boot. This includes API syntax, configuration, version migration, library-specific debugging, setup instructions, and CLI tool usage. Use it even when you think you know the answer because training data may not reflect recent changes. Prefer this over web search for library docs.

Do not use Context7 for refactoring, writing scripts from scratch, debugging business logic, code review, or general programming concepts.

Steps:

1. Always start with `resolve-library-id` using the library name and the user's question, unless the user provides an exact library ID in `/org/project` format.
2. Pick the best match by exact name match, description relevance, code snippet count, source reputation, and benchmark score. If results do not look right, try alternate names or queries. Use version-specific IDs when the user mentions a version.
3. Run `query-docs` with the selected library ID and the user's full question.
4. Answer using the fetched docs.
<!-- context7 -->

## Project Shape

- This is a Python/Kedro project managed with `uv`; use Python `3.13.12` from `.python-version`.
- Main Python package: `src/cartola`. Tests live in `src/tests`.
- Operational scripts live in `scripts/`. Generated reports and model outputs are written under `data/08_reporting/`.
- Do not commit secrets or local machine config from `conf/local`.

## Setup And Quality

- Install local dev dependencies with `uv sync --dev`.
- Reproduce the GitHub Actions quality gate with `uv sync --locked --dev` and `uv run --frozen scripts/pyrepo-check --all`.
- `scripts/pyrepo-check` supports targeted checks: `ruff`, `ty`, `bandit`, and `pytest`.
- Run tests directly with `uv run --frozen pytest` when a narrower pytest workflow is useful.
- Use `make clean` to remove Python caches, build artifacts, coverage fragments, `references/`, and `results/`.

## Backtesting And Audits

- Offline no-fixture backtest:
  `uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode none`
- Exploratory fixture backtest for reconstructed 2025 fixtures:
  `uv run --frozen python -m cartola.backtesting.cli --season 2025 --start-round 5 --budget 100 --fixture-mode exploratory`
- Season compatibility audit:
  `uv run --frozen python scripts/audit_backtest_compatibility.py --current-year 2026`
- FootyStats compatibility audit:
  `uv run --frozen python scripts/audit_footystats_compatibility.py --current-year 2026`
- FootyStats PPG is the current recommended no-fixture feature mode. Keep `ppg_xg` experimental unless a fresh ablation justifies changing the default.
- Backtests and recommendations use the `cartola_standard_2026_v1` scoring contract: all official formations are searched, a non-tecnico captain is selected with a `1.5x` multiplier, and report totals should use the captain-aware point fields.

## Live Recommendation Workflow

- Preferred one-command live workflow:
  `uv run --frozen python scripts/run_live_round.py --season 2026 --budget 100 --footystats-mode ppg --current-year 2026`
- `scripts/run_live_round.py` defaults to `--capture-policy fresh`; use `missing` to reuse a valid live capture when present, or `skip` to require one without fetching `atletas/mercado`.
- One-command live recommendation outputs are archived under `data/08_reporting/recommendations/{season}/round-{target_round}/live/runs/run_started_at=.../`.
- Capture the open market round before a live recommendation:
  `uv run --frozen python scripts/capture_market_round.py --season 2026 --auto --current-year 2026`
- Generate a live squad recommendation:
  `uv run --frozen python scripts/recommend_squad.py --season 2026 --target-round 14 --mode live --budget 100 --footystats-mode ppg --current-year 2026`
- Replay a completed current-season round:
  `uv run --frozen python scripts/recommend_squad.py --season 2026 --target-round 10 --mode replay --budget 100 --footystats-mode ppg --current-year 2026`
- Recommendation outputs are written under `data/08_reporting/recommendations/{season}/round-{target_round}/{mode}/`.
- `recommended_squad.csv` keeps per-player `predicted_points` raw; use `predicted_points_with_captain` and `actual_points_with_captain` for captain-adjusted totals when present.

## Strict Fixture Capture

- Capture strict pre-lock fixture evidence for the open market round:
  `uv run --frozen python scripts/capture_strict_round_fixture.py --season 2026 --auto --current-year 2026`
- The command writes raw evidence under `data/01_raw/fixtures_snapshots/{season}/` and canonical strict fixture CSV/manifests under `data/01_raw/fixtures_strict/{season}/`.
- This is provenance only for now; live recommendations still use `fixture_mode=none` unless strict fixture integration is explicitly added later.

## Cautions

- `--fixture-mode strict` requires pre-lock snapshot/manifests under `data/01_raw/fixtures_strict/{season}/`; do not claim strict no-leakage fixture evaluation without those files.
- CI updates raw data with `uv run --frozen --no-dev python src/cartola/download_data.py` and then `src/cartola/update_readme.py`.
- TODO: Verify the Docker workflow before relying on `make docker`; `Dockerfile` still references Poetry and Python 3.10 while the current project setup uses `uv` and Python 3.13.12.
