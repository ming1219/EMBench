"""
EMBench 题目 → Harbor 题目
=========================
每道 EMBench 题目生成一个自包含的 Harbor 题目目录：

    <out>/<family>__<instance>/
        task.toml                Harbor 配置：工具服务地址、超时、判分阶段才注入的密钥
        instruction.md           原题 + 「把最终回答写入 /app/answer.md」
        environment/
            Dockerfile           agent 容器
            docker-compose.yaml  agent 容器 + tool-server 容器
            tool-server/         模拟工具服务：tool_server.py、tools.json、fixtures.json、embench/（判分代码副本）
        tests/
            test.sh              运行 grade.py
            grade.py             取服务端调用记录 + 读回答 → EMBench 判分 → reward.txt
            embench/             判分代码副本
            embench_tasks/<id>/  原 EMBench 题目（判分规则来源）
        solution/
            solve.sh, solve.py   参考解：通过 MCP 真实调用工具并写出参考回答
            trajectory.json      参考轨迹

用法：python -m embench export-harbor -o harbor_tasks [-t 筛选]
"""
from __future__ import annotations

import json
import secrets
import shutil
from pathlib import Path

from ..core.task import PACKAGE_DIR, Task
from ..core.trajectory import Trajectory
from .fixtures import build_fixtures

HARBOR_DIR = Path(__file__).resolve().parent
RUNTIME = HARBOR_DIR / "runtime"
TOOLS_JSON = HARBOR_DIR / "tools.json"
DEFAULT_PIP_INDEX = "https://pypi.org/simple"
ANSWER_NOTE = ("\n\n---\n\n完成后，请把你给用户的完整最终回答写入文件 `/app/answer.md`。"
               "所需工具由 MCP 服务 `embench-tools` 提供。\n\n"
               "When you are done, write your complete final answer to the user into `/app/answer.md`. "
               "The tools are provided by the `embench-tools` MCP server.\n")
# 判分代码副本只需要这些（不含 agents / CLI / 题库）
VENDORED = ["__init__.py", "core", "harbor/__init__.py", "harbor/fixtures.py", "tools/__init__.py", "tools/open_tools.py"]


def global_fixture_steps(tasks: list[Task]) -> list[dict]:
    """所有题目参考轨迹里的工具调用，作为全局 fixture。

    生成类题目（tag 含 generation）的参考结果是为造题合成的示意值（例如候选分子的 KJ 结果），
    排在最后：同一工具同一参数有多条记录时，优先命中真实计算得到的结果。
    """
    ordered = sorted(tasks, key=lambda t: "generation" in t.meta.get("tags", []))
    return [s.to_dict() for t in ordered for s in Trajectory.load(t.solution_path).steps]


def harbor_name(task: Task) -> str:
    return task.id.replace("/", "__")


def _vendor_embench(dest: Path) -> None:
    for rel in VENDORED:
        src = PACKAGE_DIR / rel
        target = dest / "embench" / rel
        if src.is_dir():
            shutil.copytree(src, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)


def _render(name: str, **values) -> str:
    text = (RUNTIME / name).read_text(encoding="utf-8")
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def _write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")  # 容器里是 Linux：统一 LF
    if executable:
        path.chmod(0o755)


def _toml_str(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def _task_toml(task: Task, token: str) -> str:
    keywords = ["embench", task.family] + list(task.meta.get("tags", []))
    return "\n".join([
        'schema_version = "1.4"',
        "",
        "[task]",
        f"name = {_toml_str('embench/' + harbor_name(task).replace('__', '-'))}",
        'version = "1.0.0"',
        f"description = {_toml_str(task.meta.get('description', ''))}",
        f"keywords = {json.dumps(keywords, ensure_ascii=False)}",
        "",
        "[metadata]",
        f"embench_task_id = {_toml_str(task.id)}",
        f"embench_task_hash = {_toml_str(task.content_hash())}",
        f"difficulty = {_toml_str(task.meta.get('difficulty', ''))}",
        "",
        "[verifier]",
        "timeout_sec = 300.0",
        "",
        "[verifier.env]",
        "# 只在判分阶段注入：agent 运行时拿不到它，因此读不到、也伪造不了服务端调用记录",
        f"EMBENCH_LOG_TOKEN = {_toml_str(token)}",
        "",
        "[agent]",
        f"timeout_sec = {float(task.timeout_sec)}",
        "",
        "[environment]",
        "build_timeout_sec = 1200.0",
        "",
        "[[environment.mcp_servers]]",
        'name = "embench-tools"',
        'transport = "streamable-http"',
        'url = "http://tool-server:8000/mcp"',
        "",
    ])


def export_task(task: Task, out_dir: Path, global_steps: list[dict], token: str,
                pip_index: str = DEFAULT_PIP_INDEX, tools: str = "fixture") -> Path:
    d = out_dir / harbor_name(task)
    if d.exists():
        shutil.rmtree(d)
    oracle = Trajectory.load(task.solution_path)
    fixtures = build_fixtures([s.to_dict() for s in oracle.steps], global_steps)

    _write(d / "task.toml", _task_toml(task, token))
    _write(d / "instruction.md", task.instruction.rstrip() + ANSWER_NOTE)

    env = d / "environment"
    main_df = _render("Dockerfile.main").replace(f"PIP_INDEX_URL={DEFAULT_PIP_INDEX}", f"PIP_INDEX_URL={pip_index}")
    _write(env / "Dockerfile", main_df)
    _write(env / "docker-compose.yaml", _render("docker-compose.yaml", TOKEN=token, PIP_INDEX_URL=pip_index, TOOLS=tools))
    ts = env / "tool-server"
    _write(ts / "Dockerfile", _render("Dockerfile.tools"))
    shutil.copy2(RUNTIME / "tool_server.py", ts / "tool_server.py")
    shutil.copy2(TOOLS_JSON, ts / "tools.json")
    _write(ts / "fixtures.json", json.dumps(fixtures, ensure_ascii=False))
    _vendor_embench(ts)

    tests = d / "tests"
    _write(tests / "test.sh", _render("test.sh", TASK_ID=task.id), executable=True)
    shutil.copy2(RUNTIME / "grade.py", tests / "grade.py")
    _vendor_embench(tests)
    # 判分只需要题目定义；参考答案和反例不随判分代码分发
    shutil.copytree(task.dir, tests / "embench_tasks" / task.id, ignore=shutil.ignore_patterns("__pycache__", "solution"))

    sol = d / "solution"
    _write(sol / "solve.sh", _render("solve.sh"), executable=True)
    shutil.copy2(RUNTIME / "solve.py", sol / "solve.py")
    shutil.copy2(task.solution_path, sol / "trajectory.json")
    return d


def export_all(tasks: list[Task], all_tasks: list[Task], out_dir: Path, token: str | None = None,
               pip_index: str = DEFAULT_PIP_INDEX, tools: str = "fixture") -> list[Path]:
    """导出 tasks；fixture 全局表取自 all_tasks 的全部参考轨迹（agent 走别的合理路径时也能拿到结果）。

    每次导出生成一个新的随机密钥，所有题目共用；生成的题目目录不应提交到公开仓库。
    """
    token = token or secrets.token_hex(16)
    global_steps = global_fixture_steps(all_tasks)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [export_task(t, out_dir, global_steps, token, pip_index, tools) for t in tasks]
    manifest = {"embench_tasks": [t.id for t in tasks], "harbor_tasks": [p.name for p in paths],
                "pip_index": pip_index, "tools": tools}
    _write(out_dir / "embench_harbor_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return paths
