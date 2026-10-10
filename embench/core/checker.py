"""
任务自检（题库结构校验 + oracle 跑通）
=========================================================
`python -m embench check` 对每道题做：

结构校验
  - 必需文件存在：task.toml / instruction.md / solution/trajectory.json
  - task.toml 字段：schema_version、task.{id,family,description,difficulty}、agent.timeout_sec
  - 每个检查项：type 已注册、必填参数齐全、无未知参数（防拼写错误静默失效）、id 唯一
  - 至少一个 required 检查
  - registry.json 与 tasks/ 目录一一对应且 hash 最新

verifier 正确性（不需要 LLM，秒级完成）
  - oracle 轨迹：pass=True、所有检查（含软指标）都通过、无 skipped
  - nop 空轨迹：必须 pass=False（防「什么都不做也得分」）
  - 截断 oracle（去掉最后一步）：必须 pass=False（防检查项过松）
  - 去掉全部工具结果的 oracle：只要有 required 检查因此缺少证据，就必须 pass=False
    （防「没有证据也算通过」；题目显式 allow_incomplete = true 时除外）
  - 工具结果的值全部换成 null 的 oracle（结构和字段名保留）：同样必须 pass=False
    （防「交一堆 null 冒充证据」）
  - solution/negatives/*.json：手写反例，必须 pass=False

注意：这些自检只证明「参考答案能通过」以及「上述几类已知错法会被抓出」，
不能证明任意错误答案都会被判错——检查项本身写得弱，自检也发现不了。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .checks import CHECKS, COMMON_PARAMS
from .normalize import parse_json
from .task import REQUIRED_FILES, TASKS_DIR, Task, discover_task_dirs, load_task
from .trajectory import Trajectory
from .verifier import verify

DIFFICULTIES = {"easy", "medium", "hard"}
REGISTRY_FILE = TASKS_DIR.parent / "registry.json"


@dataclass
class TaskReport:
    task_id: str
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_config(task: Task) -> list[str]:
    errs = []
    cfg = task.config
    if cfg.get("schema_version") != "1.0":
        errs.append(f"schema_version 应为 \"1.0\"，实际 {cfg.get('schema_version')!r}")
    meta = cfg.get("task", {})
    for key in ("id", "family", "description", "difficulty"):
        if not meta.get(key):
            errs.append(f"[task].{key} 缺失")
    if meta.get("difficulty") and meta["difficulty"] not in DIFFICULTIES:
        errs.append(f"[task].difficulty 应为 {sorted(DIFFICULTIES)}")
    if meta.get("family") and not task.id.startswith(meta["family"] + "/"):
        errs.append(f"[task].family={meta['family']!r} 与目录不符")
    if not isinstance(task.agent_config.get("timeout_sec"), (int, float)):
        errs.append("[agent].timeout_sec 缺失或非数值")

    verifier_cfg = cfg.get("verifier", {})
    unknown_v = set(verifier_cfg) - {"checks", "allow_incomplete"}
    if unknown_v:
        errs.append(f"[verifier] 未知字段 {sorted(unknown_v)}")
    if "allow_incomplete" in verifier_cfg and not isinstance(verifier_cfg["allow_incomplete"], bool):
        errs.append("[verifier].allow_incomplete 应为 true / false")

    checks = task.checks
    if not checks:
        errs.append("[verifier].checks 为空")
    seen = set()
    for i, spec in enumerate(checks):
        where = f"checks[{i}]"
        ctype = spec.get("type")
        if ctype not in CHECKS:
            errs.append(f"{where}: 未知检查类型 {ctype!r}（可用: {sorted(CHECKS)}）")
            continue
        cs = CHECKS[ctype]
        for p, required in cs.params.items():
            if required and p not in spec:
                errs.append(f"{where} ({ctype}): 缺必填参数 {p}")
        unknown = set(spec) - set(cs.params) - COMMON_PARAMS
        if unknown:
            errs.append(f"{where} ({ctype}): 未知参数 {sorted(unknown)}")
        cid = spec.get("id", ctype)
        if cid in seen:
            errs.append(f"{where}: 检查 id {cid!r} 重复")
        seen.add(cid)
        if ctype == "python" and not (task.dir / spec.get("path", "")).is_file():
            errs.append(f"{where}: python 检查脚本不存在 {spec.get('path')}")
    def _required(s):
        cs = CHECKS.get(s.get("type"))
        return s.get("required", cs.required_default if cs else True)

    if checks and not any(_required(s) for s in checks):
        errs.append("没有任何 required 检查，pass 永远为 False")
    return errs


def validate_verifier(task: Task) -> list[str]:
    errs = []
    oracle = Trajectory.load(task.solution_path)
    r = verify(task, oracle)
    if not r["pass"]:
        bad = [c for c in r["checks"] if c["passed"] is False]
        errs.append(f"oracle 未通过: {[(c['id'], c['detail']) for c in bad]}")
    for c in r["checks"]:
        if c["passed"] is None:
            errs.append(f"oracle 下检查 {c['id']} 被跳过（参考轨迹缺工具结果？）")
        elif c["passed"] is False and not c["required"]:
            errs.append(f"oracle 未通过软检查 {c['id']}: {c['detail']}")

    if verify(task, Trajectory(task_id=task.id))["pass"]:
        errs.append("nop（空轨迹）竟然通过 —— 检查项过松")

    if oracle.steps:
        truncated = Trajectory.from_dict(oracle.to_dict())
        truncated.steps = truncated.steps[:-1]
        if verify(task, truncated)["pass"]:
            errs.append("截断 oracle（去掉最后一步）仍通过 —— 最后一步没被任何检查约束")

    if oracle.has_any_result:
        stripped = Trajectory.from_dict(oracle.to_dict())
        for st in stripped.steps:
            st.result, st.has_result = None, False
        rs = verify(task, stripped)
        if rs["incomplete"] and rs["pass"]:
            errs.append(f"去掉全部工具结果后仍通过（缺证据的检查: {rs['missing_evidence']}）—— 没有证据也算通过")
        if rs["incomplete"]:  # 有检查依赖工具结果 → 用 null 冒充的结果也必须判失败
            nulled = Trajectory.from_dict(oracle.to_dict())
            for st in nulled.steps:
                st.result = _null_out(st.result)
            if verify(task, nulled)["pass"]:
                errs.append("工具结果的值全部换成 null 后仍通过 —— null 被当成了证据")

    for neg in task.negative_paths:
        if verify(task, Trajectory.load(neg))["pass"]:
            errs.append(f"反例 {neg.name} 竟然通过")
    return errs


def _null_out(result):
    """保留工具结果的结构和字段名，把所有值换成 null（列表长度不变）。"""
    def walk(v):
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return None
    obj = parse_json(result)
    return json.dumps(walk(obj)) if isinstance(obj, (dict, list)) else None


def check_task(task_dir: Path, tasks_root: Path = TASKS_DIR) -> TaskReport:
    tid = Path(task_dir).relative_to(tasks_root).as_posix()
    rep = TaskReport(tid)
    missing = [f for f in REQUIRED_FILES if not (Path(task_dir) / f).is_file()]
    if missing:
        rep.errors.append(f"缺文件 {missing}")
        return rep
    try:
        task = load_task(task_dir, tasks_root)
    except Exception as e:
        rep.errors.append(f"加载失败: {e}")
        return rep
    if not task.instruction:
        rep.errors.append("instruction.md 为空")
    rep.errors += validate_config(task)
    if not rep.errors:  # 配置错时跑 verifier 只会产生噪声
        try:
            rep.errors += validate_verifier(task)
        except Exception as e:
            rep.errors.append(f"verifier 自检异常: {type(e).__name__}: {e}")
    return rep


# ----------------------------------------------------------------
#  registry.json
# ----------------------------------------------------------------

def build_registry(tasks_root: Path = TASKS_DIR) -> dict:
    entries = []
    for d in discover_task_dirs(tasks_root):
        t = load_task(d, tasks_root)
        entries.append({
            "id": t.id,
            "family": t.family,
            "difficulty": t.meta.get("difficulty"),
            "description": t.meta.get("description"),
            "hash": t.content_hash(),
        })
    families = sorted({e["family"] for e in entries})
    return {"name": "embench", "version": _version(), "n_tasks": len(entries), "families": families, "tasks": entries}


def _version() -> str:
    from .. import __version__
    return __version__


def write_registry(path: Path = REGISTRY_FILE, tasks_root: Path = TASKS_DIR) -> dict:
    reg = build_registry(tasks_root)
    Path(path).write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return reg


def check_registry(path: Path = REGISTRY_FILE, tasks_root: Path = TASKS_DIR) -> list[str]:
    if not Path(path).is_file():
        return ["registry.json 不存在，先运行 `embench registry --write`"]
    on_disk = json.loads(Path(path).read_text(encoding="utf-8"))
    fresh = build_registry(tasks_root)
    errs = []
    a = {e["id"]: e for e in on_disk.get("tasks", [])}
    b = {e["id"]: e for e in fresh["tasks"]}
    errs += [f"registry 有但目录不存在: {i}" for i in sorted(set(a) - set(b))]
    errs += [f"目录有但 registry 未登记: {i}" for i in sorted(set(b) - set(a))]
    errs += [f"任务内容已改但 registry hash 未更新: {i}" for i in sorted(set(a) & set(b)) if a[i]["hash"] != b[i]["hash"]]
    return errs
