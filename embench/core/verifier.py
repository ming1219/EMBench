"""
Verifier：跑一个任务的全部检查项，产出 reward（即 reward.json）。

    {
      "task_id": "ob-kj/tnt",
      "pass": true,            # 全部 required 检查通过
      "reward": 1.0,           # = float(pass)，二值主指标
      "dimensions": {"l1_tool": 1.0, "l2_args": 1.0, "l3_value": 1.0, "efficiency": 1.0},
      "incomplete": false,     # 有 required 检查因数据不足被跳过
      "checks": [...],
      "stats": {"n_tool_calls": 2, "redundancy": 0}
    }
"""
from __future__ import annotations

from .checks import run_check
from .trajectory import Trajectory


def verify(task, traj: Trajectory) -> dict:
    results = [run_check(traj, spec, task) for spec in task.checks]

    dims: dict[str, list[float]] = {}
    for r in results:
        for d, s in r.scores.items():
            if s is not None:
                dims.setdefault(d, []).append(s)

    required = [r for r in results if r.required]
    skipped = [r for r in required if r.passed is None]
    passed = bool(required) and all(r.passed is not False for r in required) and len(skipped) < len(required)

    redundancy = next((r.data.get("redundancy") for r in results if "redundancy" in r.data), None)
    return {
        "task_id": task.id,
        "task_hash": task.content_hash(),
        "pass": passed,
        "reward": 1.0 if passed else 0.0,
        "dimensions": {d: sum(v) / len(v) for d, v in dims.items()},
        "incomplete": bool(skipped),
        "checks": [r.to_dict() for r in results],
        "stats": {
            "n_tool_calls": len(traj.steps),
            "redundancy": redundancy,
            "error": traj.error,
        },
    }
