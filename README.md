**English** | [简体中文](README.zh-CN.md)

# EMBench: a framework for writing and grading exams for AI assistants

**EMBench** is a framework for checking whether an AI assistant that uses tools really did a task right. Its focus is the **workflow**, not any particular question set: how to write a task so that it can be verified, how to grade it automatically, and how to make sure the grading itself can be trusted.

Today's AI assistants do more than chat. They call "tools" on their own: look up a structure, predict a density, compute a detonation velocity, plan a synthesis route.
EMBench grades exactly that process:

> The task says "first compute the oxygen balance, then the detonation parameters". Did the AI pick the right tools? In the right order?
> Did it pass in the right molecule? Are the computed numbers right? Did it clearly tell you the conclusion at the end?

The repository ships a **seed set of chemistry tasks**, migrated from earlier experiments, to demonstrate the workflow end to end ([section 3](#3-the-seed-task-set)). They are examples, not a polished benchmark. The value is in the framework: anyone can write new tasks, check that they are sound, and grade any agent against them.

---

## Contents

1. [Why EMBench](#1-why-embench)
2. [An analogy: how EMBench works](#2-an-analogy-how-embench-works)
3. [The seed task set](#3-the-seed-task-set)
4. [Getting started (about 10 minutes)](#4-getting-started-about-10-minutes)
5. [Reading the report card](#5-reading-the-report-card)
6. [Grading your own AI](#6-grading-your-own-ai)
7. [Grading rules: what counts as correct](#7-grading-rules-what-counts-as-correct)
8. [Tools your AI needs to provide](#8-tools-your-ai-needs-to-provide)
9. [Writing a new task](#9-writing-a-new-task)
10. [FAQ](#10-faq)
11. [License](#11-license)

---

## 1. Why EMBench

Large language models are increasingly used as **agents** that call domain tools step by step. In scientific domains it is hard to tell whether an agent truly did the job right. Final answers can look plausible while the tool calls behind them are wrong, skipped or made up.
EMBench turns this into an engineering problem that can be measured, reproduced and improved:

1. **Typical tasks with clear specifications.** Real chemistry workflows (name-to-structure lookup, structure validation, property prediction, oxygen balance and detonation calculations, molecule generation, multi-step design and screening) are distilled into tasks. Each task has an explicit instruction, an expected sequence of steps, and verification criteria.
2. **A reproducible, automatically verifiable environment.** Every task is a self-contained folder containing the instruction, the declarative grading rules and a reference solution. Grading runs offline in seconds, needs no model calls, and gives the same result on any machine.
3. **Trustworthy trajectory data.** Agent runs are recorded as standardized tool-call trajectories: the tools called, their arguments, their results and the final answer. Each trajectory is checked against domain standards: canonical molecular structures, numeric tolerances, and consistency between steps.
4. **Quality-controlled task data.** The suite validates itself. Every reference solution must pass, while empty, truncated and known-wrong trajectories must fail. This catches broken or overly lenient tasks before they distort results, and keeps the task set correct and diverse.
5. **Evaluation mechanisms for model evaluation and reinforcement learning.** Pass/fail decisions are backed by multi-dimensional scores: tool selection, arguments, numeric results, coverage, consistency, answer quality and efficiency. Repeated-trial metrics (pass@k, pass^k) are also reported. The same verifiers can serve as reward signals for RL environments.
6. **Failure analysis that drives iteration.** Every failed task reports exactly which check failed and why. This makes it easy to analyze failure cases, refine tasks and environments, compare models and agent architectures, and verify that changes really improve capability.

## 2. An analogy: how EMBench works

Think of it as an exam:

| In an exam | In EMBench | Where |
|---|---|---|
| A question | A folder | `embench/tasks/<category>/<task>/` |
| The question text | An instruction sent to the AI | `instruction.md` in each task folder |
| The marking scheme | Which checks this task applies | `task.toml` in each task folder |
| The model answer | A "perfect answer sheet" | `solution/trajectory.json` in each task folder |
| The student's answer sheet | A record of what the AI did: which tools it called, what it passed in, what came back, what it finally said | You provide this (section 6) |
| The examiner | The EMBench program | the `embench/` folder |
| The report card | A table: how many tasks passed per category, and what went wrong | `report.md`, generated after every grading run |

**The exam itself is checked too.** Every task's model answer must get full marks. A blank sheet, a sheet with the last step missing, a sheet without the tool results the grading needs, and a sheet with a typical mistake (each task ships with such examples) must all fail.
This catches broken tasks and these known kinds of mistakes. It **cannot prove that every wrong answer will be caught**: if a task's checks are written loosely, the self-check won't notice. That is why each task should ship counterexamples of its own.

> Note: the seed task instructions are written in Chinese, because they target Chinese-language assistants. Command output is currently in Chinese as well.

## 3. The seed task set

These tasks were migrated from earlier experiments to demonstrate the framework, and their quality varies. Treat them as worked examples of how to write tasks; to write your own, see [section 9](#9-writing-a-new-task).

| Category (folder) | Tasks | Difficulty | What it tests |
|---|---:|---|---|
| `name2smiles` | 20 | easy | Given a chemical name (e.g. TNT, CL-20), look up its structure as SMILES (a one-line text notation for molecular structure) |
| `smiles-validate-draw` | 10 | easy | Check that a SMILES string is valid, then draw the molecule |
| `smiles-to-3d` | 10 | easy | Convert a SMILES string into 3D atomic coordinates |
| `property-predict` | 10 | medium | Validate a structure, then predict properties such as density |
| `ob-kj` | 6 | medium | Compute the oxygen balance, then the detonation parameters with the KJ equations |
| `generate-predict` | 10 | medium | Design a number of new molecules and predict the requested properties |
| `magus-generate` | 2 | medium | Check that the crystal-structure module is available, then generate crystal structures for a formula |
| `design-pipeline` | 3 | hard | A full design workflow: generate candidates → predict properties → compute detonation parameters → assess synthesizability → recommend the best candidate |
| `design-multi` | 3 | hard | Multi-objective design with screening conditions, e.g. "among candidates with decomposition temperature above 200 °C, recommend the one with the highest detonation velocity" |

To see exactly what a task asks, open its `instruction.md`, for example `embench/tasks/ob-kj/tnt/instruction.md`.

## 4. Getting started (about 10 minutes)

The steps below use Windows; macOS and Linux are almost identical. **You only need to copy and paste commands. No programming is required.**

### Step 1: Install Python

EMBench needs **Python 3.11 or newer**.

1. Go to <https://www.python.org/downloads/>, download the latest version and run the installer
2. At the bottom of the installer, **make sure "Add python.exe to PATH" is ticked**, then click "Install Now"

> If Python is already installed, skip this step. You can check in step 3.

### Step 2: Download EMBench

1. Near the top of this page, click the green **"Code"** button → **"Download ZIP"**
2. Unzip it anywhere, e.g. `D:\EMBench`

### Step 3: Open a command window

1. Open the unzipped folder in File Explorer (the level where you can see `README.md` and `embench`)
2. Click the **address bar** at the top of the window, replace the path with `cmd`, and press Enter

A black window appears. This is the command line, already "standing" inside the EMBench folder.
Type the following and press Enter to confirm Python is installed:

```
python --version
```

You should see something like `Python 3.12.x`. If it says the command is not recognized, see the [FAQ](#10-faq).

### Step 4: Install the dependency

EMBench needs one extra chemistry toolkit, RDKit, to recognize "different ways of writing the same molecule":

```
pip install -r requirements.txt
```

### Step 5: Check the exam itself

```
python -m embench check
```

After a few seconds the last line reads:

```
74/74 题通过自检
```

("74/74 tasks passed the self-check".) Every task in the seed set passed: its model answer gets full marks, and the known kinds of wrong answers (blank, truncated, missing evidence, and the task's own counterexamples) are caught.

### Step 6: Run your first grading

Invite two "demo students": one hands in the model answers, the other hands in a blank sheet.

```
python -m embench run -a oracle --job model-answers
python -m embench run -a nop --job blank-sheet
```

The first should pass 100%, the second 0%. The report card is printed in the window and also saved as
`embench\jobs\model-answers\report.md`, which you can open in Notepad or any Markdown viewer.

## 5. Reading the report card

Here is the report card produced from the sample answer sheets that ship with the repository (used again in section 6):

```
**pass@1 = 33.3%**（Wilson 95% CI 6.1% – 79.2%，按 3 道题计）

| family      | 题数 | pass@1 | l1_tool | l2_args | l3_value | answer | efficiency | 调用数 | 冗余 |
| name2smiles | 2    | 50.0%  | 100.0%  | 100.0%  | —        | 50.0%  | 100.0%     | 1.0    | 0.0  |
| ob-kj       | 1    | 0.0%   | 0.0%    | 0.0%    | 50.0%    | —      | 100.0%     | 1.0    | 1.0  |

## 未通过（2）
- `name2smiles/rdx#0` — answer: 回答中未找到 1 个期望分子
- `ob-kj/tnt#0` — trajectory: calculateOxygenBalance: 未调用; ob_values: calculateOxygenBalance 未调用
```

What each term means:

| Term | Meaning |
|---|---|
| **pass@1** | **Pass rate.** A task passes only if all of its required checks pass. 33.3% means 1 of 3 tasks passed |
| **Wilson 95% CI** | **Plausible range.** With few tasks, a score has some luck in it. "6.1% – 79.2%" means that on another set of similar tasks, the true level would very likely fall in this range. It is computed over **tasks** ("按 3 道题计" = "based on 3 tasks"): more tasks give a narrower range, but repeating the same tasks does not |
| family | Task category |
| 题数 | Number of tasks |
| l1_tool | **Tool choice:** were the required tools used, in the right order? |
| l2_args | **Arguments:** were the right molecule, numbers and options passed to the tools? |
| l3_value | **Results:** are the numbers the tools returned correct? |
| coverage | **Completeness:** in design tasks, was every candidate molecule fully analyzed? |
| consistency | **No fabrication:** were the analyzed molecules really the ones generated earlier, rather than made up? |
| answer | **Communication:** did the final reply give the user the correct conclusion? |
| efficiency | **Conciseness:** were unnecessary tool calls avoided? (for reference only; does not affect pass/fail) |
| 调用数 / 冗余 | Average number of tool calls per task / how many of them were unnecessary |
| — | Not applicable to this category |
| 未通过 | Failed tasks, with the exact check that failed. "未调用" = "not called"; "回答中未找到 1 个期望分子" = "expected molecule not found in the answer" |

The two failures above are two typical mistakes:
- **`name2smiles/rdx`**: the tool was called correctly, but the final reply only said "I've looked up the structure of RDX" and **never told the user the structure**
- **`ob-kj/tnt`**: the task required the oxygen balance first, then the detonation parameters; the AI **skipped the oxygen balance**

**Comparing several AIs?** Pass several report folders to `report` for a side-by-side table:

```
python -m embench report embench\jobs\ai-a embench\jobs\ai-b
```

**Checking how stable an AI is?** Let it do every task 3 times (`-k 3`), and the report adds two numbers:

| Metric | Meaning |
|---|---|
| pass@3 | Share of tasks solved **at least once** in 3 tries: *can* it do it? |
| pass^3 | Share of tasks solved **every time** in 3 tries: is it *reliable*? |

## 6. Grading your own AI

Three steps: **let the AI do the task → record what it did as an answer sheet → hand it in for grading.**

### 6.1 Let the AI do the task

Send the content of a task's `instruction.md` to your AI unchanged, and let it use its tools as usual.
(The tools it needs are listed in [section 8](#8-tools-your-ai-needs-to-provide).)

### 6.2 Write the answer sheet

An answer sheet is a `.json` text file (editable in Notepad) recording what the AI did. This is the sample
`examples/my_agent_outputs/name2smiles/tnt.json`:

```json
{
  "agent": {"name": "my-agent", "model": "my-llm"},
  "steps": [
    {
      "tool": "nameToSmiles",
      "args": {"name": "TNT"},
      "result": "{\"smiles\": \"Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]\", \"source\": \"builtin\"}"
    }
  ],
  "final_answer": "TNT（三硝基甲苯）的 SMILES 为 Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]"
}
```

| Field | What to put |
|---|---|
| `agent` | A name for your AI (shown on the report card); `model` can be the underlying LLM |
| `steps` | Every tool call the AI made, **in order**, one entry per call |
| └ `tool` | Tool name; must match a name in [section 8](#8-tools-your-ai-needs-to-provide) |
| └ `args` | What was passed to the tool |
| └ `result` | What the tool returned (copy as is). **Don't leave it out:** if a check needs it to verify the numbers, a missing result means the task fails ("缺少证据" = "missing evidence") |
| `final_answer` | The AI's full final reply to the user |

> Note: in this example the SMILES in the reply is written differently from the one the tool returned, but it is the same molecule, so **it is not marked wrong**.

If your AI is built in code, have the program write these files automatically while it works. No copying by hand needed.

### 6.3 Hand it in

Answer sheets are arranged like the task folders: `<answer-sheet folder>/<category>/<task>.json`. For example:

```
examples/my_agent_outputs/
├── name2smiles/
│   ├── tnt.json      ← for task embench/tasks/name2smiles/tnt/
│   └── rdx.json      ← for task embench/tasks/name2smiles/rdx/
└── ob-kj/
    └── tnt.json      ← for task embench/tasks/ob-kj/tnt/
```

If your AI did a task several times (see `-k` below), give **each attempt its own file**: `<category>/<task>/trial_0.json`, `trial_1.json`, … A single `<task>.json` counts as one attempt only. It is never reused as several.

Then run the command below. `-t` lists which tasks to grade; without `-t` every task is graded, and tasks without an answer sheet count as blank:

```
python -m embench run -a replay --replay-src examples/my_agent_outputs -t name2smiles/tnt name2smiles/rdx ob-kj/tnt --job example
```

This gives the report card shown in [section 5](#5-reading-the-report-card). Use your own folder and name:

```
python -m embench run -a replay --replay-src D:\my-answer-sheets --job my-ai-first-try
```

**Common ways to pick tasks:**

| Option | Meaning |
|---|---|
| `-t ob-kj` | Only the ob-kj category |
| `-t ob-kj name2smiles` | These two categories |
| `-t "design-*"` | All categories starting with `design-` |
| `-t name2smiles/tnt` | A single task |

To list all tasks: `python -m embench list`

## 7. Grading rules: what counts as correct

Each task's `task.toml` lists its checks. **A task passes only if every required check passes.**
Reference-only checks (currently just "efficiency") produce a score but do not affect pass/fail.

To avoid marking correct work as wrong, grading is lenient about:

| Situation | Treatment |
|---|---|
| The same molecule written as different SMILES | Treated as equal (recognized with RDKit) |
| Name variants such as CL20 / CL-20 / cl_20 | Treated as equal |
| Different list order or letter case | Treated as equal |
| Small numeric differences | Allowed (usually within 1%) |
| A few extra, unrelated tool calls | Not an error; only counted as redundancy |
| Optional tool parameters (e.g. image size) | Not checked |

Grading does **not** forgive:

- Skipping a required step, or doing steps in the wrong order
- Passing the wrong molecule, quantity or option to a tool
- In design tasks, leaving a candidate unanalyzed, or analyzing made-up molecules
- Not telling the user the conclusion (the structure, or the recommended candidate). In design tasks the reply must **name a candidate that was actually generated**: refusing ("I don't recommend any candidate"), a vague "the first one", or a made-up molecule all fail
- Missing evidence: when a check needs a tool result that the answer sheet doesn't contain, the task fails instead of being skipped

## 8. Tools your AI needs to provide

To take this exam, the AI must be able to call these tools (names must match exactly):

| Tool | Purpose | Main parameters | Used in |
|---|---|---|---|
| `nameToSmiles` | Chemical name → SMILES | `name` | name2smiles |
| `validateSmiles` | Check whether a SMILES string is valid | `smiles` | smiles-validate-draw, property-predict |
| `drawMolecule` | Draw a molecule | `smiles` | smiles-validate-draw |
| `smilesTo3D` | SMILES → 3D coordinates | `smiles` | smiles-to-3d |
| `batchPredictProperties` | Predict properties for a batch of molecules | `smiles_list`, `properties` | property-predict, design tasks |
| `calculateOxygenBalance` | Compute the oxygen balance | `smiles_list` | ob-kj |
| `calculateKJ` | Detonation parameters via the KJ equations | `smiles` | ob-kj, design tasks |
| `generateAndPredict` | Generate new molecules (optionally predicting properties) | `num`, `use_em`, `properties`, `skip_prediction` | generate-predict, design tasks |
| `predictRetrosynthesis` | Retrosynthesis (can it be synthesized?) | `smiles`, `max_routes`, `time_limit` | design tasks |
| `magusStatus` | Check whether the crystal-structure module is available | none | magus-generate |
| `magusGenerate` | Generate crystal structures | `formula`, `number` | magus-generate |

Property codes: `Density`, `DetoD` detonation velocity, `DetoP` detonation pressure, `DetoQ` heat of detonation, `DetoT` detonation temperature, `DetoV` detonation volume, `HOF_S` solid-state heat of formation, `Td` decomposition temperature, `BDE` bond dissociation energy.

> The tool implementations are **not** included in this repository. You can use your own, as long as the names and parameters match.
> To see what a tool is expected to return, look at the `result` fields in a task's `solution/trajectory.json`.

## 9. Writing a new task

1. Copy a task folder of the same category under a new name, e.g. copy `embench/tasks/ob-kj/tnt` to `embench/tasks/ob-kj/my-molecule`
2. Edit the three files inside:
   - `instruction.md`: the new task text
   - `task.toml`: change `id` to `ob-kj/my-molecule`, and update the molecules and values to check
   - `solution/trajectory.json`: the model answer sheet for the new task
3. Check it: `python -m embench check -t ob-kj/my-molecule`
4. Register it in the task index: `python -m embench registry --write`

If the check fails, it explains what is wrong. For example, it reports when the model answer doesn't get full marks, or when a blank sheet would pass (meaning the checks are too lenient).

For every available check and how to configure it, see [docs/specification.md](docs/specification.md).

## 10. FAQ

**Typing `python` says the command is not recognized?**
"Add python.exe to PATH" wasn't ticked during installation. Run the installer again, choose "Modify" and tick it, or try `py` instead of `python`.

**`pip install` is slow or fails?**
Make sure Python is 3.11 or newer. In mainland China, a mirror helps: add `-i https://pypi.tuna.tsinghua.edu.cn/simple` to the command.

**Why is RDKit required?**
The same molecule can be written as many different SMILES strings. Without RDKit, the program can't tell whether two strings are the same molecule, and correct answers could be marked wrong.

**Garbled characters in the report card?**
Open `report.md` in an editor that supports UTF-8 (e.g. VS Code, Typora), or run `chcp 65001` in the command window first.

**The same AI gives different results each time?**
That's normal: language models are random to some degree. Let your AI do each task 3 times, save every attempt as its own answer sheet (`trial_0.json` … `trial_2.json`), grade with `-k 3`, and look at pass^3 (the share solved every time). The built-in demo students (`oracle`, `nop`) always give the same answer, so they refuse `-k` above 1.

**If the grading rules change, does the AI have to retake the exam?**
No. Run `python -m embench score embench\jobs\<some-run>` to regrade with the new rules. The answer sheets stay the same.

**Grading was interrupted?**
Run the same command again with the same `--job` name. Tasks already graded are skipped automatically. A job belongs to one AI: if you grade a **different** AI under an existing job name, EMBench refuses rather than mixing up the two AIs' results. Use a new `--job` name, or add `--force` to wipe that job and start over.

## 11. License

Released under the [MIT License](LICENSE): free to use, modify and redistribute, as long as the copyright notice is kept.

Some molecules in the tasks come from the public [PubChem](https://pubchem.ncbi.nlm.nih.gov/) database.
