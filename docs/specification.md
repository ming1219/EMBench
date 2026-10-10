# EMBench 技术规范

> 面向开发者与题目维护者。第一次使用请先看仓库根目录的 [README](../README.md)。

## 1. 命令速查（仓库根目录下）

```bash
python -m embench list   [-t 筛选]                 # 列出任务
python -m embench check  [-t 筛选] [-v]            # 题目自检：结构 + oracle 必过 + 空/截断/反例必挂 + registry
python -m embench registry [--write]               # 校验 / 重建 registry.json
python -m embench run -a oracle|nop|replay [...]   # 跑一次阅卷（job）
python -m embench score  <job目录>                 # 用当前题目定义重判已有 job
python -m embench report <job目录> [<job目录> ...] # 单 job 摘要 / 多 job 对比
python -m unittest discover -s embench/tests -t .  # 单元测试
```

`run` 的参数：

| 参数 | 说明 |
|---|---|
| `-t` | 任务 id 的 glob 或 family 名：`ob-kj`、`"design-*/*"`、`name2smiles/tnt`；缺省 = 全部 |
| `-a` | `oracle`（参考答案）/ `nop`（空白卷）/ `replay`（回放自备轨迹） |
| `--replay-src` | replay 的轨迹目录，结构 `<family>/<instance>.json` |
| `-k` | 每题重复次数，>1 时报告 pass@k 与 pass^k |
| `-n` | 并发数 |
| `--job` | job 名；同名再跑 = 续跑（已有 reward.json 的 trial 跳过） |
| `--jobs-dir` | job 输出目录，缺省 `embench/jobs/` |
| `--force` | 忽略已有结果，全部重跑 |

## 2. 目录结构

```
embench/
├── tasks/<family>/<instance>/
│   ├── task.toml                 元数据 + 声明式 verifier
│   ├── instruction.md            给智能体的指令
│   └── solution/
│       ├── trajectory.json       oracle 参考轨迹（必须 100% 通过）
│       └── negatives/*.json      反例轨迹（必须失败）
├── registry.json                 题库索引（id / family / 难度 / 内容 hash）
├── core/                         task · trajectory · normalize · checks · verifier · metrics · runner · report · checker
├── agents/                       oracle · nop · replay
├── tests/
└── jobs/<job>/                   运行产物（gitignore）
    ├── config.json               被测 agent、git commit、环境版本、每题 hash
    ├── trials/<task>/trial_<k>/{trajectory.json, reward.json}
    ├── result.json               聚合指标
    └── report.md
```

## 3. 轨迹格式（trajectory.json）

```json
{
  "schema_version": "1.0",
  "task_id": "ob-kj/tnt",
  "agent": {"name": "my-agent", "model": "some-llm"},
  "steps": [
    {"tool": "calculateOxygenBalance", "args": {"smiles_list": ["..."]}, "result": "<字符串或 JSON>"},
    {"tool": "calculateKJ", "args": {"smiles": "..."}, "result": "..."}
  ],
  "final_answer": "给用户的最终回答",
  "error": null,
  "metrics": {"wall_time_s": 12.3, "input_tokens": 0, "output_tokens": 0}
}
```

- `steps` 按调用顺序排列；`result` 省略表示没有记录结果，此时依赖结果的检查会被跳过（不算失败，但 reward 标 `incomplete`）
- `schema_version`、`task_id`、`error`、`metrics` 可省略；replay 会自动补上 `task_id`
- `metrics` 中的 `wall_time_s` / `input_tokens` / `output_tokens` 若提供，会出现在报告的成本统计里

## 4. 任务格式（task.toml，schema_version = "1.0"）

```toml
schema_version = "1.0"

[task]
id = "ob-kj/tnt"                 # 必须等于目录相对路径
family = "ob-kj"
description = "氧平衡 + KJ 爆轰参数：TNT"
difficulty = "medium"            # easy | medium | hard
tags = ["chemprop", "builtin"]
legacy = { exp = "Exp6_ob_kj", key = "TNT" }   # 可选：题目来源标记

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
type = "call_budget"             # 软指标：缺省 required = false
max_calls = 2
```

### 4.1 所有检查共有的字段

| 字段 | 含义 |
|---|---|
| `id` | 唯一标识（缺省 = type） |
| `required` | 是否计入 pass（缺省：`call_budget` 为 false，其余为 true） |
| `threshold` | 各维度 ≥ 该值才算通过（缺省 1.0） |
| `gate` | 哪些维度参与通过判定（缺省 = 全部） |
| `description` | 人读说明 |

未知参数、缺必填参数、未知检查类型都会被 `check` 报错，拼写错误不会静默失效。

### 4.2 检查类型

| type | 维度 | 用途 | 关键参数 |
|---|---|---|---|
| `tool_calls_match` | l1_tool, l2_args | 期望的调用序列出现在轨迹中 | `calls`, `ordered`=true, `allow_extra`=true |
| `tools_used` | l1_tool | 只看是否调用过某些工具 | `all_of`, `none_of` |
| `arg_constraints` | l2_args | 参数约束（适合输出随机的生成类任务） | `tool`, `constraints`, `which`=first/all/any/sum |
| `tool_result_values` | l3_value | 工具输出字段与参考值一致 | `tool`, `expected`, `rel_tol`, `abs_tol`, `match_smiles`, `which` |
| `generation_output` | l3_value | 生成数 ≥ min_count，结果含所需性质 | `min_count`, `required_properties` |
| `candidate_coverage` | coverage, consistency | 每个生成候选是否都走完下游；下游输入是否来自生成结果 | `downstream`, `source_tool` |
| `answer_contains` | answer | 最终回答包含关键词 | `any_of`, `all_of`, `min_length` |
| `answer_smiles` | answer | 最终回答包含正确分子（canonical 比较） | `expected`, `mode` |
| `call_budget` | efficiency | 调用数预算（软指标） | `max_calls` |
| `python` | 自定义 | 逃生口：调用题目目录下 `tests/verify.py` 的 `verify(traj, task, params)` | `path`, `function` |

`arg_constraints` 支持的算子：`eq ne le ge lt gt in set_eq subset_of present absent`，另有 `default`（缺参时的默认值）和 `rule`（归一规则）。

### 4.3 参数归一规则

评测跟随工具自身的容错。规则按参数名自动选择，可在 calls 中用 `arg_rules = { 参数名 = "规则" }` 覆盖：

| 参数名 | 规则 | 说明 |
|---|---|---|
| `smiles` | smiles | RDKit canonical，同一分子的不同写法视为相等 |
| `smiles_list` | smiles_set | 兼容 list、JSON 字符串、逗号分隔串、单个字符串，按集合比较 |
| `name` | alias | 去掉 `-`、`_`、空格后小写（CL20 = CL-20） |
| `properties` | str_set | 按集合比较，大小写无关 |
| `formula` | str | 精确比较（区分大小写） |
| 其他 | 默认 | 字符串小写；"true"/"false" 转为布尔；数字串转为数值；浮点相对容差 1e-3 |

`tool_calls_match` 中没有在 expected 里声明的参数一律不比较；需要禁止某个参数时用 `forbid_args`。

## 5. 判分语义

- **pass**：所有 `required` 检查都通过。被跳过的检查不算失败，但 reward 会标记 `incomplete: true`
- **reward** = float(pass)，二值主指标
- **dimensions**：各维度在所有检查上的均值，用于诊断

| 维度 | 含义 |
|---|---|
| l1_tool | 选对工具、顺序正确 |
| l2_args | 参数正确 / 合规 |
| l3_value | 工具输出的数值正确 |
| coverage | 生成的每个候选都走完了后续分析 |
| consistency | 后续分析的输入确实来自生成结果，而不是编造的 |
| answer | 最终回答的质量 |
| efficiency | 调用效率（软指标） |

## 6. 统计指标

| 指标 | 说明 |
|---|---|
| pass@1 | 每题通过率的均值 |
| pass@k | k 次中至少一次通过（Chen et al. 2021 无偏估计），`-k` > 1 时输出 |
| pass^k | k 次全部通过（τ-bench 一致性指标），衡量稳定性 |
| Wilson 95% CI | 小样本下的通过率置信区间 |
| 维度均值 / 平均调用数 / 冗余 / 耗时 / token | 诊断与成本 |

## 7. 维护流程

**加一道题**：复制同 family 的任务目录 → 修改 `task.toml`、`instruction.md`、`solution/trajectory.json`
→ 建议加一个 `solution/negatives/*.json` 反例 → `python -m embench check -t <id>` → `python -m embench registry --write`。

**加一种检查**：在 `embench/core/checks.py` 中写函数并用 `@register(name, params=..., dimensions=...)` 注册，
然后在 `embench/tests/test_core.py` 加测试用例。声明了 `params` 后，checker 会自动校验 task.toml 中的参数。

**修改判分口径**：改检查项 → `check`（oracle 与反例自检）→ `score <job目录>` 重判历史结果。
每次 job 的 `config.json` 记录了运行时每道题的 hash，`score` 会提示哪些题在运行之后被改过。

**提交前必须通过**：

```bash
python -m embench check
python -m unittest discover -s embench/tests -t .
```

## 8. 自检具体检查什么

| 检查项 | 防止的问题 |
|---|---|
| 必需文件齐全，task.toml 字段合法，检查类型与参数合法 | 题目写错、拼写错误导致检查静默失效 |
| oracle 参考轨迹通过全部检查 | 题目无解，或检查写错 |
| 空轨迹必须失败 | 什么都不做也能得分 |
| 去掉最后一步的 oracle 必须失败 | 检查过松，最后一步没有被任何检查约束 |
| `solution/negatives/` 中每个反例必须失败 | 已知的典型错误被放过 |
| registry.json 与目录一一对应且 hash 最新 | 题库索引过期 |
