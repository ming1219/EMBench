"""
任务加载
========
一个任务 = 一个目录：

    tasks/<family>/<instance>/
        task.toml
        instruction.md
        solution/trajectory.json
        solution/negatives/*.json     （可选）
        tests/verify.py               （可选：type="python" 的自定义检查）

task.toml 结构见 docs/specification.md；load_task() 只做解析与最基本的字段校验，
深度校验（检查项类型合法、oracle 可通过等）在 checker.py。
"""
from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import toml as _toml

    class tomllib:  # noqa: N801  最小兼容层
        @staticmethod
        def loads(s):
            return _toml.loads(s)

PACKAGE_DIR = Path(__file__).resolve().parents[1]
TASKS_DIR = PACKAGE_DIR / "tasks"

REQUIRED_FILES = ("task.toml", "instruction.md", "solution/trajectory.json")


@dataclass
class Task:
    id: str
    dir: Path
    config: dict
    instruction: str

    @property
    def meta(self) -> dict:
        return self.config.get("task", {})

    @property
    def family(self) -> str:
        return self.meta.get("family", self.id.split("/")[0])

    @property
    def checks(self) -> list[dict]:
        return self.config.get("verifier", {}).get("checks", [])

    @property
    def agent_config(self) -> dict:
        return self.config.get("agent", {})

    @property
    def timeout_sec(self) -> float:
        return float(self.agent_config.get("timeout_sec", 600))

    @property
    def solution_path(self) -> Path:
        return self.dir / "solution" / "trajectory.json"

    @property
    def negative_paths(self) -> list[Path]:
        d = self.dir / "solution" / "negatives"
        return sorted(d.glob("*.json")) if d.is_dir() else []

    @property
    def root(self) -> Path:
        """题库根目录（id = 相对于它的路径）。"""
        return self.dir.parents[len(self.id.split("/")) - 1]

    def content_hash(self) -> str:
        """任务版本指纹：task.toml + instruction.md + tests/ 的 sha256 前 12 位。

        写进每次 job 的 manifest，结果可追溯到「用的是哪一版题」。
        solution/ 不参与（改参考解不改变题目本身）。
        """
        h = hashlib.sha256()
        files = [self.dir / "task.toml", self.dir / "instruction.md"]
        tests = self.dir / "tests"
        if tests.is_dir():
            files += sorted(p for p in tests.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        for p in files:
            h.update(p.relative_to(self.dir).as_posix().encode())
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        return h.hexdigest()[:12]


def load_task(task_dir: Path, tasks_root: Path = TASKS_DIR) -> Task:
    task_dir = Path(task_dir)
    task_id = task_dir.relative_to(tasks_root).as_posix()
    config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
    declared = config.get("task", {}).get("id")
    if declared and declared != task_id:
        raise ValueError(f"{task_dir}: task.id={declared!r} 与目录路径 {task_id!r} 不一致")
    return Task(id=task_id, dir=task_dir, config=config, instruction=instruction)


def discover_task_dirs(tasks_root: Path = TASKS_DIR) -> list[Path]:
    return sorted(p.parent for p in Path(tasks_root).rglob("task.toml"))


def load_tasks(patterns: list[str] | None = None, tasks_root: Path = TASKS_DIR) -> list[Task]:
    """按 glob 过滤任务 id（如 "ob-kj/*"、"design-*/*"）；None/空 = 全部。"""
    tasks = [load_task(d, tasks_root) for d in discover_task_dirs(tasks_root)]
    if not patterns:
        return tasks
    return [t for t in tasks if any(fnmatch.fnmatch(t.id, p) or t.family == p for p in patterns)]
