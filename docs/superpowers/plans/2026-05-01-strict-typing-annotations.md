# Strict Typing Annotations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repository pass `uv run --frozen scripts/pyrepo-check --all` with Ruff `ANN` annotation checks enabled.

**Architecture:** Keep the strict annotation policy in `pyproject.toml`. Fix missing annotations in production code, scripts, and tests using precise local types where practical. Use narrow `Any` only at true dynamic boundaries such as arbitrary JSON payloads, third-party objects, or plugin/Kedro entry points.

**Tech Stack:** Python 3.13.12, Ruff `ANN`, ty, pytest, bandit, uv.

---

### Task 1: Establish The Failing Baseline

**Files:**
- Read: `pyproject.toml`
- Read: `scripts/pyrepo-check`

- [ ] **Step 1: Run the full gate**

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected: fails at Ruff with `ANN` annotation violations.

- [ ] **Step 2: Capture a focused annotation report**

```bash
uv run --frozen python -m ruff check src/cartola src/tests scripts/pyrepo-check --select ANN --output-format concise
```

Expected: lists every remaining annotation violation.

### Task 2: Apply Mechanical Annotation Fixes

**Files:**
- Modify: files reported by Ruff under `src/cartola`, `src/tests`, and `scripts/pyrepo-check`

- [ ] **Step 1: Run Ruff's annotation fixer**

```bash
uv run --frozen python -m ruff check src/cartola src/tests scripts/pyrepo-check --select ANN --fix --unsafe-fixes
```

Expected: Ruff applies safe mechanical annotations such as `-> None` on tests and fixtures.

- [ ] **Step 2: Re-run the focused annotation report**

```bash
uv run --frozen python -m ruff check src/cartola src/tests scripts/pyrepo-check --select ANN --output-format concise
```

Expected: remaining violations are the non-mechanical cases requiring explicit argument or dynamic-boundary types.

### Task 3: Fix Remaining Production Annotations

**Files:**
- Modify: remaining production files reported under `src/cartola`

- [ ] **Step 1: Replace untyped parameters with concrete local types**

Examples:

```python
def _find_run_command(package_name: str) -> Callable[..., object] | None:
    ...
```

```python
def _optional_int(value: object) -> int | None:
    ...
```

- [ ] **Step 2: Keep dynamic boundaries explicit**

Use `object` for values only inspected/coerced locally. Use `Any` with `# noqa: ANN401` only when the function genuinely accepts or returns arbitrary third-party data and no stronger type is useful.

- [ ] **Step 3: Run production annotation check**

```bash
uv run --frozen python -m ruff check src/cartola --select ANN
```

Expected: no `ANN` errors in production code.

### Task 4: Fix Remaining Test And Script Annotations

**Files:**
- Modify: remaining test files under `src/tests`
- Modify: `scripts/pyrepo-check` if reported

- [ ] **Step 1: Type pytest fixtures and monkeypatch helpers**

Examples:

```python
def test_example(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ...
```

```python
def fake_run_model_experiment(**kwargs: object) -> object:
    ...
```

- [ ] **Step 2: Type nested fake functions precisely enough**

Use existing project dataclasses where obvious, for example `BacktestConfig`, `BacktestResult`, and `CaptureFixture[str]`.

- [ ] **Step 3: Run test annotation check**

```bash
uv run --frozen python -m ruff check src/tests scripts/pyrepo-check --select ANN
```

Expected: no `ANN` errors in tests or scripts.

### Task 5: Run Full Verification

**Files:**
- No planned file edits.

- [ ] **Step 1: Run the full quality gate**

```bash
uv run --frozen scripts/pyrepo-check --all
```

Expected:
- Ruff passes with `ANN` enabled.
- ty passes.
- bandit passes.
- pytest passes.

- [ ] **Step 2: Inspect final diff**

```bash
git status --short
git diff --stat
```

Expected: only annotation-related changes plus this plan file.
