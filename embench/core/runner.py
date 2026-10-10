"""
Job 运行器
=========
一次 job = 一个 agent 配置 × 一组任务 × k 次试验，落盘结构：

    jobs/<job>/
        config.json                  manifest：agent/模型/模式、git commit、环境版本、每题 hash
        trials/<task__id>/trial_<k>/
            trajectory.json          统一轨迹
            reward.json              verifier 输出
        result.json                  聚合指标（overall + 按 family）+ 每题明细
        report.md                    人读摘要

- 可续跑：已有 reward.json 的 trial 直接跳过（长跑中断后重跑同一命令即可）
- 可重判：`score` 只对已有 trajectory 重新跑 verifier，不再调 LLM
  （改了检查项 / 修了判分 bug 后，历史结果一键刷新，agent 与 verifier 彻底解耦）
"""
from __future__ import annotations

import asyncio
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

from .metrics import aggregate
from .task import PACKAGE_DIR, Task, load_tasks
from .trajectory import Trajectory
from .verifier import verify

JOBS_DIR = PACKAGE_DIR / "jobs"


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


async def run_job(agent, tasks: list[Task], job_dir: Path, n_trials: int = 1,
                  n_concurrent: int = 1, force: bool = False, log=print) -> dict:
    job_dir.mkdir(parents=True, exist_ok=True)
    todo = [(t, k) for t in tasks for k in range(n_trials)
            if force or not (trial_dir(job_dir, t.id, k) / "reward.json").is_file()]
    log(f"job {job_dir.name}: {len(tasks)} 题 × {n_trials} 次，待跑 {len(todo)} 个 trial")

    await agent.setup()
    manifest = {
        "job": job_dir.name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "agent": agent.info(),
        "n_trials": n_trials,
        "environment": environment_info(),
        "tasks": {t.id: t.content_hash() for t in tasks},
    }
    _write_json(job_dir / "config.json", manifest)

    sem = asyncio.Semaphore(max(1, n_concurrent))
    done = 0

    async def one(task: Task, k: int):
        nonlocal done
        async with sem:
            traj = await agent.run(task, k)
        traj.task_id = task.id
        d = trial_dir(job_dir, task.id, k)
        traj.save(d / "trajectory.json")
        reward = verify(task, traj)
        _write_json(d / "reward.json", reward)
        done += 1
        flag = "PASS" if reward["pass"] else ("ERR " if traj.error else "FAIL")
        log(f"[{done}/{len(todo)}] {flag} {task.id} #{k}  calls={len(traj.steps)} "
            f"{' '.join(f'{d}={v:.2f}' for d, v in reward['dimensions'].items())}")

    try:
        await asyncio.gather(*(one(t, k) for t, k in todo))
    finally:
        await agent.teardown()
    return finalize(job_dir, tasks, n_trials)


def collect_trials(job_dir: Path, tasks: list[Task], n_trials: int) -> list[dict]:
    trials = []
    for t in tasks:
        for k in range(n_trials):
            d = trial_dir(job_dir, t.id, k)
            if not (d / "reward.json").is_file():
                continue
            reward = json.loads((d / "reward.json").read_text(encoding="utf-8"))
            traj = json.loads((d / "trajectory.json").read_text(encoding="utf-8"))
            trials.append({"task_id": t.id, "family": t.family, "trial": k,
                           "reward": reward, "metrics": traj.get("metrics", {})})
    return trials


def finalize(job_dir: Path, tasks: list[Task], n_trials: int) -> dict:
    trials = collect_trials(job_dir, tasks, n_trials)
    result = aggregate(trials)
    result["job"] = job_dir.name
    result["per_task"] = {
        t["task_id"] + f"#{t['trial']}": {
            "pass": t["reward"]["pass"],
            "dimensions": t["reward"]["dimensions"],
            "failed_checks": [c["id"] + ": " + c["detail"] for c in t["reward"]["checks"] if c["passed"] is False],
        }
        for t in trials
    }
    _write_json(job_dir / "result.json", result)
    from .report import job_markdown
    (job_dir / "report.md").write_text(job_markdown(job_dir), encoding="utf-8")
    return result


def rescore_job(job_dir: Path, log=print) -> dict:
    """用当前任务定义重跑 verifier（不调用 agent）。"""
    manifest = json.loads((job_dir / "config.json").read_text(encoding="utf-8"))
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
                _write_json(d / "reward.json", verify(t, Trajectory.load(d / "trajectory.json")))
                n += 1
    manifest["rescored"] = {"at": datetime.datetime.now().isoformat(timespec="seconds"),
                            "environment": environment_info(),
                            "tasks": {t.id: t.content_hash() for t in tasks}}
    _write_json(job_dir / "config.json", manifest)
    log(f"已重判 {n} 个 trial")
    return finalize(job_dir, tasks, manifest["n_trials"])
