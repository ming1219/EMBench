"""
Job 运行器
=========
一次 job = 一个 agent 配置 × 一组任务 × k 次试验，落盘结构：

    jobs/<job>/
        config.json                  manifest：agent/模型/模式、git commit、环境版本、每题 hash
        trials/<task__id>/trial_<k>/
            trajectory.json          统一轨迹
            reward.json              verifier 输出（含产生该 trial 的 agent 身份）
        result.json                  聚合指标（overall + 按 family）+ 每题明细
        report.md                    人读摘要

- 可续跑：已有 reward.json 的 trial 直接跳过。续跑前核对身份：agent 与 job 记录的不一致、
  或已有结果的题目定义被改过，都直接报错（防止旧成绩被贴上新 agent 的标签）
- --force：清空该 job 的全部结果，从头重跑
- 可重判：`score` 只对已有 trajectory 重新跑 verifier，不再调用 agent
- 汇总时逐个核对 trial 的 agent 身份，与 job 不一致即报错
"""
from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .metrics import aggregate
from .task import PACKAGE_DIR, Task, load_tasks
from .trajectory import Trajectory
from .verifier import verify

JOBS_DIR = PACKAGE_DIR / "jobs"


class JobError(Exception):
    """job 状态与本次运行冲突（换了 agent、题目被改过、对确定性 agent 重复试验等）。"""


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=PACKAGE_DIR, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def environment_info() -> dict:
    try:
        import rdkit
        rdkit_v = rdkit.__version__
    except Exception:
        rdkit_v = None
    from .. import __version__
    commit = _git("rev-parse", "--short", "HEAD")
    return {
        "embench_version": __version__,
        "git_commit": commit,
        "git_dirty": None if commit == "unknown" else bool(_git("status", "--porcelain", "--", ".")),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "rdkit": rdkit_v,
    }


def trial_dir(job_dir: Path, task_id: str, k: int) -> Path:
    return job_dir / "trials" / task_id.replace("/", "__") / f"trial_{k}"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _label(info: dict) -> str:
    return " / ".join(str(v) for v in info.values() if v not in (None, "", False))


def _reset_job(job_dir: Path) -> None:
    shutil.rmtree(job_dir / "trials", ignore_errors=True)
    for name in ("config.json", "result.json", "report.md"):
        (job_dir / name).unlink(missing_ok=True)


def _check_resume(job_dir: Path, agent_info: dict, tasks: list[Task]) -> dict | None:
    """续跑前核对：同一个 job 只能属于同一个 agent，已有结果的题目定义不能被改过。"""
    path = job_dir / "config.json"
    if not path.is_file():
        if (job_dir / "trials").is_dir() and any((job_dir / "trials").iterdir()):
            raise JobError(f"job {job_dir.name} 有成绩文件但缺少 config.json，无法确认是谁交的卷。"
                           "请换一个 --job 名，或加 --force 清空后重跑")
        return None
    old = _read_json(path)
    if old.get("agent") != agent_info:
        raise JobError(f"job {job_dir.name} 已由另一个 agent 运行过（{_label(old.get('agent', {}))}），"
                       f"本次是（{_label(agent_info)}）。请换一个 --job 名，或加 --force 清空该 job 从头重跑")
    changed = [t.id for t in tasks
               if t.id in old.get("tasks", {}) and old["tasks"][t.id] != t.content_hash()
               and (job_dir / "trials" / t.id.replace("/", "__")).is_dir()]
    if changed:
        raise JobError(f"{len(changed)} 道题在上次运行后被修改过（如 {changed[0]}），已有成绩按旧定义判出。"
                       "请先用 `score` 按新定义重判，或加 --force 从头重跑")
    return old


async def run_job(agent, tasks: list[Task], job_dir: Path, n_trials: int = 1,
                  n_concurrent: int = 1, force: bool = False, log=print) -> dict:
    if n_trials > 1 and getattr(agent, "deterministic", False):
        raise JobError(f"agent {agent.name} 每次给出的结果都一样，重复 {n_trials} 次不会带来新信息，"
                       "只会虚增样本量。请用 -k 1")
    job_dir.mkdir(parents=True, exist_ok=True)
    if force:
        _reset_job(job_dir)

    await agent.setup()
    try:
        agent_info = agent.info()
        old = _check_resume(job_dir, agent_info, tasks)
        all_ids = dict.fromkeys(list((old or {}).get("tasks", {})) + [t.id for t in tasks])
        all_tasks = {t.id: t for t in load_tasks(list(all_ids))} | {t.id: t for t in tasks}
        manifest = {
            "job": job_dir.name,
            "created": (old or {}).get("created") or datetime.datetime.now().isoformat(timespec="seconds"),
            "agent": agent_info,
            "n_trials": max(n_trials, (old or {}).get("n_trials", 0)),
            "environment": environment_info(),
            "tasks": {tid: t.content_hash() for tid, t in all_tasks.items()},
        }
        _write_json(job_dir / "config.json", manifest)

        todo = [(t, k) for t in tasks for k in range(n_trials)
                if not (trial_dir(job_dir, t.id, k) / "reward.json").is_file()]
        log(f"job {job_dir.name}: {len(tasks)} 题 × {n_trials} 次，待跑 {len(todo)} 个 trial")

        sem = asyncio.Semaphore(max(1, n_concurrent))
        done = 0

        async def one(task: Task, k: int):
            nonlocal done
            async with sem:
                traj = await agent.run(task, k)
            traj.task_id = task.id
            d = trial_dir(job_dir, task.id, k)
            traj.save(d / "trajectory.json")
            reward = verify(task, traj) | {"agent": agent_info}
            _write_json(d / "reward.json", reward)
            done += 1
            flag = "PASS" if reward["pass"] else ("ERR " if traj.error else "FAIL")
            log(f"[{done}/{len(todo)}] {flag} {task.id} #{k}  calls={len(traj.steps)} "
                f"{' '.join(f'{d}={v:.2f}' for d, v in reward['dimensions'].items())}")

        await asyncio.gather(*(one(t, k) for t, k in todo))
    finally:
        await agent.teardown()
    return finalize(job_dir, list(all_tasks.values()), manifest["n_trials"])


def _fingerprint(traj: dict) -> str:
    """轨迹内容指纹（工具调用 + 最终回答），用于发现「同一份答卷复制成多次试验」。"""
    body = json.dumps([traj.get("steps", []), traj.get("final_answer", "")], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def collect_trials(job_dir: Path, tasks: list[Task], n_trials: int) -> list[dict]:
    manifest = _read_json(job_dir / "config.json")
    trials = []
    for t in tasks:
        seen = set()
        for k in range(n_trials):
            d = trial_dir(job_dir, t.id, k)
            if not (d / "reward.json").is_file():
                continue
            reward = _read_json(d / "reward.json")
            if "agent" in reward and reward["agent"] != manifest["agent"]:
                raise JobError(f"{t.id} 第 {k} 次试验由另一个 agent 产生（{_label(reward['agent'])}），"
                               f"与 job 记录的（{_label(manifest['agent'])}）不一致，拒绝汇总")
            traj = _read_json(d / "trajectory.json")
            fp = _fingerprint(traj)
            trials.append({"task_id": t.id, "family": t.family, "trial": k, "reward": reward,
                           "metrics": traj.get("metrics", {}),
                           "verified_agent": "agent" in reward, "duplicate": fp in seen})
            seen.add(fp)
    return trials


def finalize(job_dir: Path, tasks: list[Task], n_trials: int) -> dict:
    trials = collect_trials(job_dir, tasks, n_trials)
    result = aggregate(trials)
    result["job"] = job_dir.name
    result["per_task"] = {
        t["task_id"] + f"#{t['trial']}": {
            "pass": t["reward"]["pass"],
            "dimensions": t["reward"]["dimensions"],
            "failed_checks": [c["id"] + ": " + c["detail"] for c in t["reward"]["checks"] if c["passed"] is False]
            + [c["id"] + ": 缺少证据（" + c["detail"] + "）" for c in t["reward"]["checks"]
               if c["passed"] is None and c["required"]],
        }
        for t in trials
    }
    _write_json(job_dir / "result.json", result)
    from .report import job_markdown
    (job_dir / "report.md").write_text(job_markdown(job_dir), encoding="utf-8")
    return result


def rescore_job(job_dir: Path, log=print) -> dict:
    """用当前任务定义重跑 verifier（不调用 agent）；保留每个 trial 原有的 agent 身份。"""
    manifest = _read_json(job_dir / "config.json")
    tasks = load_tasks(list(manifest["tasks"]))
    by_id = {t.id: t for t in tasks}
    missing = sorted(set(manifest["tasks"]) - set(by_id))
    if missing:
        log(f"⚠ 以下任务已不存在，跳过: {missing}")
    changed = [tid for tid, h in manifest["tasks"].items() if tid in by_id and by_id[tid].content_hash() != h]
    if changed:
        log(f"ℹ {len(changed)} 道题的定义自 job 运行后有变更（结果按新定义重判）: {changed[:5]}{' ...' if len(changed) > 5 else ''}")
    n = 0
    for t in tasks:
        for k in range(manifest["n_trials"]):
            d = trial_dir(job_dir, t.id, k)
            if (d / "trajectory.json").is_file():
                old = _read_json(d / "reward.json") if (d / "reward.json").is_file() else {}
                reward = verify(t, Trajectory.load(d / "trajectory.json"))
                if "agent" in old:
                    reward["agent"] = old["agent"]
                _write_json(d / "reward.json", reward)
                n += 1
    manifest["tasks"] = {t.id: t.content_hash() for t in tasks}
    manifest["rescored"] = {"at": datetime.datetime.now().isoformat(timespec="seconds"),
                            "environment": environment_info()}
    _write_json(job_dir / "config.json", manifest)
    log(f"已重判 {n} 个 trial")
    return finalize(job_dir, tasks, manifest["n_trials"])
