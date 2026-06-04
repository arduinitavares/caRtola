# Live Budget Parameter Contract

## Scope

This document defines the current live recommendation budget contract for
`scripts/run_live_round.py`. It supersedes older live-round design notes that
described a default `--budget 100.0` for this command.

The contract applies only to live pre-lock recommendation runs. Historical
backtests, model experiments, tuning commands, and completed-round replay paths
may keep their own budget defaults until a separate accepted requirement changes
them.

## CLI Argument Contract

`scripts/run_live_round.py` must expose `--budget` as a required operator input.

The accepted value is the operator's current available Cartola budget for the
live round, expressed in Cartola C$ / cartoletas.

The accepted type is a positive finite decimal number.

The command must reject all of the following before market capture, model
prediction, recommendation artifact creation, or squad generation starts:

- missing `--budget`;
- empty `--budget`;
- non-numeric `--budget`;
- zero `--budget`;
- negative `--budget`;
- non-finite `--budget`, including NaN and infinity.

## Error Handling

Missing `--budget` must fail with a non-zero exit code and a stderr message that
tells the operator to provide the available budget in Cartola C$ / cartoletas.

Invalid `--budget` values must fail with a non-zero exit code and a stderr
message that identifies the value as invalid and states that the value must be a
positive numeric Cartola C$ / cartoletas amount.

These failures must occur before any of these side effects:

- Cartola market capture;
- model loading or prediction;
- optimizer execution;
- recommendation directory creation;
- recommendation CSV or JSON artifact writing.

## CLI Entry Point Design

Budget validation belongs at the `scripts/run_live_round.py` argument parsing
boundary.

The parser must use a dedicated `--budget` type validator that accepts only
positive finite numeric values. It must reject non-numeric, empty, zero,
negative, NaN, and infinite values before constructing `LiveWorkflowConfig`.

The parser must then perform a required-value check for `--budget`. A missing
budget must call the parser error path with this operator-facing guidance:

`--budget is required; provide available budget in Cartola C$ / cartoletas`

Invalid provided values must use this operator-facing guidance:

`budget must be a positive numeric Cartola C$ / cartoletas amount`

The entry point must not attempt to repair, infer, or replace missing budget
input. There is no fallback from config, environment, private account balance,
Cartola API account responses, or previous run metadata. The only value that may
enter `LiveWorkflowConfig.budget` is the explicit parsed `--budget` argument.

## Forbidden Budget Sources

The live recommendation workflow must not use any implicit budget source.

Forbidden sources include:

- hard-coded default `100.0`;
- optional budget fallback;
- project configuration files;
- `.env` files;
- process environment variables;
- private Cartola account balance;
- Cartola API account or wallet responses;
- prior live recommendation runs;
- historical backtest state;
- cached recommendation metadata.

The live command may read the explicit parsed `--budget` value only after it has
passed validation.

## Downstream Contract

When `--budget` is valid, the live workflow must pass exactly that explicit
operator-provided value into squad selection.

`recommendation_summary.json` must record:

- `budget`: the explicit operator-provided value;
- `budget_used`: the total selected squad cost;
- `budget_used <= budget`.

`recommended_squad.csv` must contain selected player prices whose summed
`preco_pre_rodada` equals the recorded `budget_used`.

No downstream component may replace, infer, or widen the budget after CLI
validation.

## Test Expectations

Implementation tests must cover at least:

- missing `--budget` fails before any workflow side effect;
- non-numeric `--budget` fails before any workflow side effect;
- zero `--budget` fails before any workflow side effect;
- negative `--budget` fails before any workflow side effect;
- a valid positive numeric `--budget` is accepted and propagated into the live
  workflow config;
- environment variables, `.env`, config files, API/account state, and prior-run
  metadata cannot make a missing budget succeed.
