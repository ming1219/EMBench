**English** | [简体中文](specification.zh-CN.md)

# EMBench technical specification

> For developers and task maintainers. New users should start with the [README](../README.md).

## 1. Command reference (run from the repository root)

```bash
python -m embench list   [-t FILTER]                 # list tasks
python -m embench check  [-t FILTER] [-v]            # self-check: structure + oracle must pass + empty/truncated/negatives must fail + registry
python -m embench registry [--write]                 # verify / rebuild registry.json
python -m embench run -a oracle|nop|replay [...]     # run one grading job
python -m embench score  <job_dir>                   # regrade an existing job with the current task definitions
python -m embench report <job_dir> [<job_dir> ...]   # single-job summary / multi-job comparison
python -m unittest discover -s embench/tests -t .    # unit tests
```

`run` options:

| Option | Meaning |
|---|---|
| `-t` | Task id glob or family name: `ob-kj`, `"design-*/*"`, `name2smiles/tnt`; default = all |
| `-a` | `oracle` (reference answers) / `nop` (blank sheet) / `replay` (replay your own trajectories) |
| `--replay-src` | Trajectory directory for replay, laid out as `<family>/<instance>.json` |
| `-k` | Trials per task; with k > 1, pass@k and pass^k are reported |
| `-n` | Concurrency |
| `--job` | Job name; rerunning the same name resumes (trials that already have reward.json are skipped) |
| `--jobs-dir` | Output directory for jobs, default `embench/jobs/` |
| `--force` | Ignore existing results and rerun everything |

## 2. Layout

```
embench/
├── tasks/<family>/<instance>/
│   ├── task.toml                 metadata + declarative verifier
│   ├── instruction.md            instruction sent to the agent
│   └── solution/
│       ├── trajectory.json       oracle reference trajectory (must pass 100%)
│       └── negatives/*.json      negative trajectories (must fail)
├── registry.json                 task index (id / family / difficulty / content hash)
├── core/                         task · trajectory · normalize · checks · verifier · metrics · runner · report · checker
├── agents/                       oracle · nop · replay
├── tests/
└── jobs/<job>/                   run outputs (gitignored)
    ├── config.json               agent under test, git commit, environment versions, per-task hashes
    ├── trials/<task>/trial_<k>/{trajectory.json, reward.json}
    ├── result.json               aggregated metrics
    └── report.md
```

## 3. Trajectory format (trajectory.json)

```json
{
  "schema_version": "1.0",
  "task_id": "ob-kj/tnt",
  "agent": {"name": "my-agent", "model": "some-llm"},
  "steps": [
    {"tool": "calculateOxygenBalance", "args": {"smiles_list": ["..."]}, "result": "<string or JSON>"},
    {"tool": "calculateKJ", "args": {"smiles": "..."}, "result": "..."}
  ],
  "final_answer": "the final reply to the user",
  "error": null,
  "metrics": {"wall_time_s": 12.3, "input_tokens": 0, "output_tokens": 0}
}
```

- `steps` are in call order. Omitting `result` means no result was recorded; checks that depend on results are then skipped. A skipped check is not a failure, but the reward is flagged `incomplete`
- `schema_version`, `task_id`, `error` and `metrics` are optional; replay fills in `task_id`
- If `metrics` provides `wall_time_s` / `input_tokens` / `output_tokens`, they appear in the report's cost statistics

## 4. Task format (task.toml, schema_version = "1.0")

```toml
schema_version = "1.0"

[task]
id = "ob-kj/tnt"                 # must equal the directory's relative path
family = "ob-kj"
description = "氧平衡 + KJ 爆轰参数：TNT"
difficulty = "medium"            # easy | medium | hard
tags = ["chemprop", "builtin"]
legacy = { exp = "Exp6_ob_kj", key = "TNT" }   # optional provenance marker

[agent]
timeout_sec = 300

[[verifier.checks]]
id = "trajectory"
type = "tool_calls_match"
ordered = true
calls = [{ tool = "calculateOxygenBalance", args = { smiles_list = ["Cc1c(...)"] } },
         { tool = "calculateKJ",            args = { smiles = "Cc1c(...)" } }]

[[verifier.checks]]
id = "kj_values"
type = "tool_result_values"
tool = "calculateKJ"
expected = { DetoD = 6995, DetoP = 21.0 }
rel_tol = 0.01

[[verifier.checks]]
id = "budget"
type = "call_budget"             # soft metric: required = false by default
max_calls = 2
```

### 4.1 Fields common to all checks

| Field | Meaning |
|---|---|
| `id` | Unique identifier (default = type) |
| `required` | Whether it counts toward pass (default: false for `call_budget`, true otherwise) |
| `threshold` | Each dimension must be ≥ this value for the check to pass (default 1.0) |
| `gate` | Which dimensions decide pass/fail (default = all) |
| `description` | Human-readable note |

Unknown parameters, missing required parameters and unknown check types are all reported by `check`, so typos never fail silently.

### 4.2 Check types

| type | Dimensions | Purpose | Key parameters |
|---|---|---|---|
| `tool_calls_match` | l1_tool, l2_args | The expected call sequence appears in the trajectory | `calls`, `ordered`=true, `allow_extra`=true |
| `tools_used` | l1_tool | Only whether certain tools were called | `all_of`, `none_of` |
| `arg_constraints` | l2_args | Argument constraints (for generation tasks with random outputs) | `tool`, `constraints`, `which`=first/all/any/sum |
| `tool_result_values` | l3_value | Tool output fields match reference values | `tool`, `expected`, `rel_tol`, `abs_tol`, `match_smiles`, `which` |
| `generation_output` | l3_value | Number generated ≥ min_count, results include the required properties | `min_count`, `required_properties` |
| `candidate_coverage` | coverage, consistency | Every generated candidate goes through the downstream tools; downstream inputs come from the generated set | `downstream`, `source_tool` |
| `answer_contains` | answer | The final reply contains keywords | `any_of`, `all_of`, `min_length` |
| `answer_smiles` | answer | The final reply contains the correct molecule (canonical comparison) | `expected`, `mode` |
| `call_budget` | efficiency | Tool-call budget (soft metric) | `max_calls` |
| `python` | custom | Escape hatch: calls `verify(traj, task, params)` in the task's `tests/verify.py` | `path`, `function` |

`arg_constraints` operators: `eq ne le ge lt gt in set_eq subset_of present absent`, plus `default` (value assumed when the argument is missing) and `rule` (normalization rule).

### 4.3 Argument normalization

Grading follows the tolerance of the tools themselves. The rule is chosen by argument name and can be overridden in `calls` with `arg_rules = { name = "rule" }`:

| Argument | Rule | Behavior |
|---|---|---|
| `smiles` | smiles | RDKit canonical form; different SMILES for the same molecule are equal |
| `smiles_list` | smiles_set | Accepts a list, a JSON string, a comma-separated string or a single string; compared as a set |
| `name` | alias | Lowercased with `-`, `_` and spaces removed (CL20 = CL-20) |
| `properties` | str_set | Compared as a set, case-insensitive |
| `formula` | str | Exact (case-sensitive) |
| others | default | Strings lowercased; "true"/"false" → boolean; numeric strings → numbers; floats with relative tolerance 1e-3 |

In `tool_calls_match`, arguments not declared in the expected call are ignored; use `forbid_args` to forbid one.

## 5. Grading semantics

- **pass**: all `required` checks pass. Skipped checks are not failures, but the reward is flagged `incomplete: true`
- **reward** = float(pass), the binary headline metric
- **dimensions**: the mean of each dimension over all checks, for diagnosis

| Dimension | Meaning |
|---|---|
| l1_tool | Right tools, right order |
| l2_args | Correct / compliant arguments |
| l3_value | Correct numeric tool outputs |
| coverage | Every generated candidate was fully analyzed |
| consistency | Downstream inputs really come from generated results rather than being made up |
| answer | Quality of the final reply |
| efficiency | Tool-call efficiency (soft metric) |

## 6. Metrics

| Metric | Meaning |
|---|---|
| pass@1 | Mean per-task pass rate |
| pass@k | Solved at least once in k trials (unbiased estimator, Chen et al. 2021); reported when `-k` > 1 |
| pass^k | Solved in all k trials (consistency metric from τ-bench); measures reliability |
| Wilson 95% CI | Confidence interval for the pass rate, suited to small samples |
| Dimension means / avg. tool calls / redundancy / time / tokens | Diagnostics and cost |

## 7. Maintenance

**Add a task:** copy a task directory of the same family → edit `task.toml`, `instruction.md`, `solution/trajectory.json`
→ ideally add a negative example under `solution/negatives/*.json` → `python -m embench check -t <id>` → `python -m embench registry --write`.

**Add a check type:** write a function in `embench/core/checks.py`, register it with `@register(name, params=..., dimensions=...)`,
and add test cases to `embench/tests/test_core.py`. Once `params` is declared, the checker validates task.toml parameters automatically.

**Change grading rules:** edit the checks → `check` (oracle and negative self-check) → `score <job_dir>` to regrade past results.
Each job's `config.json` records the hash of every task at run time, and `score` reports which tasks changed since the run.

**Must pass before committing:**

```bash
python -m embench check
python -m unittest discover -s embench/tests -t .
```

## 8. What the self-check verifies

| Check | Prevents |
|---|---|
| Required files present, valid task.toml fields, valid check types and parameters | Broken tasks; typos silently disabling checks |
| The oracle trajectory passes every check | Unsolvable tasks, or wrong checks |
| An empty trajectory fails | Scoring points for doing nothing |
| The oracle with its last step removed fails | Overly lenient checks; a last step no check constrains |
| Every negative in `solution/negatives/` fails | Known typical mistakes slipping through |
| registry.json matches the directories and its hashes are current | A stale task index |
