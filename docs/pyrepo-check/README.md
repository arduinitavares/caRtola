# pyrepo-check

`pyrepo-check` is the repository's Python quality-gate command. It runs the
standard local checks used before merging Python changes.

Older revisions of this repository used an executable script at
`scripts/pyrepo-check`. In the current project shape, that script has been
removed and the Makefile calls the installed `pyrepo-check` command directly.

## Full Quality Gate

Run the full gate from the repository root:

```bash
pyrepo-check --all
```

Before reproducing the CI-style gate locally, install the locked development
environment:

```bash
uv sync --locked --dev
```

## What It Runs

`--all` runs these checks:

| Check | Purpose |
| --- | --- |
| `ruff` | Lint Python code and enforce configured Ruff rules |
| `ty` | Type-check the Python package |
| `bandit` | Scan production Python code for common security issues |
| `pytest` | Run the test suite |

The project-specific targets live in `pyproject.toml`:

```toml
[tool.pyrepo-check]
ruff_targets = ["src/cartola", "src/tests", "scripts"]
bandit_targets = ["src/cartola"]
```

## Type Annotation Enforcement

This repository enforces annotation presence through Ruff's `ANN` rules:

```toml
[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I"]
extend-select = ["ANN"]
```

That means `pyrepo-check ruff` is not only a style lint. It also fails when
the checked targets violate the annotation policy, including:

- missing regular argument annotations
- missing `*args` and `**kwargs` annotations
- missing return annotations on public, private, static, class, and special methods
- `Any` used as a function argument type where Ruff requires a more specific type

To run only the annotation gate directly:

```bash
uv run --frozen ruff check src/cartola src/tests scripts --select ANN
```

The original strict-typing workflow used both the script-driven gate and direct
Ruff annotation commands:

```bash
pyrepo-check --all
uv run --frozen ruff check src/cartola src/tests scripts --select ANN --output-format concise
uv run --frozen ruff check src/cartola src/tests scripts --select ANN --fix --unsafe-fixes
uv run --frozen ruff check src/cartola src/tests scripts --select ANN
pyrepo-check --all
```

`pyrepo-check --all` then runs `ty` after Ruff, so annotation presence is
followed by type-consistency checking.

Use precise local types where practical. Use `Any` only at true dynamic
boundaries, such as arbitrary JSON payloads, third-party objects, or framework
entry points.

## Targeted Checks

You can run one or more named checks instead of the full gate:

```bash
pyrepo-check ruff
pyrepo-check ty
pyrepo-check bandit
pyrepo-check pytest
pyrepo-check ruff pytest
```

You can also run the Makefile wrappers:

```bash
make ruff
make ty
make bandit
make quality
```

`make quality` runs `pyrepo-check --all`.

## Exit Behavior

- Checks run sequentially.
- The command stops at the first failing check.
- On failure, it exits with that check's return code.
- Unknown check names are rejected before checks run.

## Maintenance

When changing the quality gate, update both:

- `[tool.pyrepo-check]` in `pyproject.toml`
- this README's check list, targets, and annotation notes
