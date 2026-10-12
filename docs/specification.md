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
python -m embench export-harbor -o <dir> [-t FILTER] [--pip-index URL]   # export as Harbor tasks (Docker)
python -m unittest discover -s embench/tests -t .    # unit tests
```

`run` options:

| Option | Meaning |
|---|---|
| `-t` | Task id glob or family name: `ob-kj`, `"design-*/*"`, `name2smiles/tnt`; default = all |
| `-a` | `oracle` (reference answers) / `nop` (blank sheet) / `replay` (replay your own trajectories) |
| `--replay-src` | Trajectory directory for replay: `<family>/<instance>.json` (one trial), or `<family>/<instance>/trial_<k>.json` (one file per trial) |
| `-k` | Trials per task; with k > 1, pass@k and pass^k are reported. Deterministic agents (`oracle`, `nop`) refuse k > 1 |
| `-n` | Concurrency |
| `--job` | Job name; rerunning the same name resumes (trials that already have reward.json are skipped). Resuming is only allowed for the **same agent** and unchanged task definitions; otherwise the run is refused |
| `--jobs-dir` | Output directory for jobs, default `embench/jobs/` |
| `--force` | Wipe the job's existing results and start over |

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

- `steps` are in call order. Omitting `result` means no result was recorded. A required check that needs it cannot verify the task, which then **fails** (missing evidence), unless the task opts in with `allow_incomplete` (section 4)
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

[verifier]
# allow_incomplete = true   # optional: skip (instead of fail) required checks that lack evidence

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
| `generation_output` | l3_value | At least `min_count` **valid** molecules were generated (`null`, empty or unparseable entries don't count), and at least `min_count` of them have a record whose required properties are all finite numbers. Faking either with `null` fails and is reported under `invalid_evidence` | `min_count`, `required_properties` |
| `candidate_coverage` | coverage, consistency | Every generated candidate goes through the downstream tools; downstream inputs come from the generated set | `downstream`, `source_tool` |
| `answer_contains` | answer | The final reply contains keywords and none of the forbidden phrases. Substring matching: "不推荐" contains "推荐", so list refusals in `none_of` | `any_of`, `all_of`, `none_of`, `min_length` |
| `answer_mentions_candidate` | answer | The final reply names (by SMILES) at least `min_count` candidates that were actually generated in this run; it only checks that a generated molecule is **mentioned**; to check **what is recommended**, use `answer_recommends_candidate` | `source_tool`, `source_field`, `min_count` |
| `answer_recommends_candidate` | answer | The final reply recommends at least `min_count` molecules (by SMILES), and **every** recommended molecule was actually generated in this run. A molecule counts as recommended only when it appears in a clause that contains a recommendation keyword and no negation/deferral phrase ("不推荐", "暂缓", "证据不足", …). Listing, comparing or rejecting a molecule does not count, so "candidates include A; I suggest holding off" and "candidates include A; I recommend B (not generated)" both fail | `source_tool`, `source_field`, `min_count`, `keywords`, `negations` |
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

- **pass**: every `required` check passes, **and none of them lacks evidence**. A required check that cannot decide (e.g. the trajectory has no tool results) makes the task fail; the reward lists it in `missing_evidence` and sets `incomplete: true`. A task may opt out with `[verifier] allow_incomplete = true`
- **invalid evidence**: tool results that exist but are `null`, non-numeric or unparseable are not evidence. The checks reading them fail outright (never skipped, so `allow_incomplete` can't excuse them); the reward lists them in `invalid_evidence` and sets `incomplete: true`
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
| Wilson 95% CI | Confidence interval for the pass rate. **The sample unit is the task**: each task contributes its pass rate, so repeating trials never narrows the interval |
| Dimension means / avg. tool calls / redundancy / time / tokens | Diagnostics and cost |
| Duplicate / unverified trials | Trials whose trajectory is identical to an earlier trial of the same task, and trials without a recorded agent identity (shown as warnings) |

**Job integrity.** Every `reward.json` records the agent that produced it and the version (`task_hash`) of the task it was graded against. Resuming a job is refused when the agent differs, or when **any** task in the job with existing results has changed since it was graded (not just the tasks selected this time); run `score` first. A resume only updates the recorded versions of the tasks it actually grades. Aggregation rejects any trial produced by another agent or graded against another task version.

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
| The oracle with all tool results removed fails, whenever a required check then lacks evidence | Passing without evidence |
| The oracle with every tool-result value replaced by `null` fails (structure and field names kept) | Passing with fake evidence |
| Every negative in `solution/negatives/` fails | Known typical mistakes slipping through |
| registry.json matches the directories and its hashes are current | A stale task index |

These checks prove that the reference passes and that the listed kinds of mistakes are caught. They **do not** prove that every wrong answer is caught: a task with loosely written checks still passes the self-check. Add counterexamples under `solution/negatives/` for the mistakes you care about.

## 9. Harbor integration (Docker)

`python -m embench export-harbor -o <dir>` turns each EMBench task into a self-contained [Harbor](https://pypi.org/project/harbor/) task (tested with Harbor 0.23.0), so any Harbor agent can take it in Docker.

```
<dir>/<family>__<instance>/
├── task.toml                 Harbor config: MCP server URL, timeouts, grading-only token in [verifier.env]
├── instruction.md            the original instruction + "write your final answer to /app/answer.md"
├── environment/
│   ├── Dockerfile            agent container (python:3.12-slim pinned by digest, mcp 1.27.2, rdkit 2026.3.3)
│   ├── docker-compose.yaml   agent container + tool-server container
│   └── tool-server/          fixture MCP server: tool_server.py, tools.json, fixtures.json, embench/ (grading code copy)
├── tests/
│   ├── test.sh               runs grade.py
│   ├── grade.py              server-side call log + /app/answer.md → EMBench checks → reward
│   ├── embench/              grading code copy
│   └── embench_tasks/<id>/   the EMBench task definition (no solution/)
└── solution/
    ├── solve.sh, solve.py    reference solution: calls the tools over MCP, writes the reference answer
    └── trajectory.json
```

**Tool server.** Serves the tools described in `embench/harbor/tools.json` over MCP (streamable HTTP, `http://tool-server:8000/mcp`). Each call is answered by replaying a recorded result (`fixtures.json`): fixtures from the task's own reference trajectory first, then from all tasks' reference trajectories, preferring the fixture that declares the most arguments. A call matches a fixture when every argument the fixture declares is equal under EMBench's normalization (equivalent SMILES, lists sent as JSON strings, etc.); extra arguments are ignored. Unmatched calls return `{"error_type": "fixture_missing", ...}`. Optional parameters are exposed with a `null` default so that only the arguments the agent actually sent are recorded.

**Evidence.** The tool server records every call (tool, arguments, result) on its own side. The agent runs in a different container and can neither read nor change this log. The grader fetches it from `GET /embench/calls` with a token that is injected only in the verification phase (`[verifier.env] EMBENCH_LOG_TOKEN`). A fresh random token is generated on every export, so generated tasks must not be published. If the log cannot be fetched, the task scores 0 (`missing_evidence: ["tool_call_log"]`).

**Reward.** `grade.py` writes `/logs/verifier/reward.txt` (0 or 1, Harbor's headline reward) and `/logs/verifier/embench_reward.json` (the full EMBench reward with dimensions and per-check details).

**Known limits.**
- Fixture tools return recorded results only; they test tool use, not real chemistry predictions.
- The grader runs in the agent's container (Harbor's shared verifier mode). The evidence is protected, but an agent with root access could in principle tamper with the container's Python packages. Harbor's separate verifier container (`environment_mode = "separate"` with `collect`) is the planned hardening.
