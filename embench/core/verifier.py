"""
Verifier：跑一个任务的全部检查项，产出 reward（即 reward.json）。

    {
      "task_id": "ob-kj/tnt",
      "pass": true,            # 全部 required 检查通过，且没有缺证据的 required 检查
      "reward": 1.0,           # = float(pass)，二值主指标
      "dimensions": {"l1_tool": 1.0, "l2_args": 1.0, "l3_value": 1.0, "efficiency": 1.0},
      "incomplete": false,     # 有 required 检查因缺少证据（如轨迹没记录工具结果）而无法判定
      "missing_evidence": [],  # 这些检查的 id
      "invalid_evidence": [],  # 工具结果用 null / 无法解析的内容冒充证据的检查 id（这些检查直接判失败）
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

    # 缺少证据 ≠ 通过：required 检查无法判定时整题不通过。
    # 只有题目在 [verifier] 里显式写 allow_incomplete = true 才放宽为「跳过」。
    allow_incomplete = bool(task.config.get("verifier", {}).get("allow_incomplete", False))
    required = [r for r in results if r.required]
    failed = [r for r in required if r.passed is False]
    skipped = [r for r in required if r.passed is None]
    # null / 无法解析的结果不是证据：这类检查已直接判失败，这里再标记出来，计入「证据不完整」
    invalid = [r.id for r in results if r.data.get("invalid_evidence")]
    passed = bool(required) and not failed and (
        not skipped or (allow_incomplete and len(skipped) < len(required)))

    redundancy = next((r.data.get("redundancy") for r in results if "redundancy" in r.data), None)
    return {
        "task_id": task.id,
        "task_hash": task.content_hash(),
        "pass": passed,
        "reward": 1.0 if passed else 0.0,
        "dimensions": {d: sum(v) / len(v) for d, v in dims.items()},
        "incomplete": bool(skipped) or bool(invalid),
        "missing_evidence": [r.id for r in skipped],
        "invalid_evidence": invalid,
        "checks": [r.to_dict() for r in results],
        "stats": {
            "n_tool_calls": len(traj.steps),
            "redundancy": redundancy,
            "error": traj.error,
        },
    }
