"""
Job 级指标聚合
=============
输入：一组 trial 的 (task_id, family, reward dict, trajectory metrics)。
输出：
- pass@1          每题通过率的均值（k 次试验的无偏估计）
- pass@k          k 次里至少一次通过（Chen et al. 2021 无偏估计）
- pass^k          k 次全部通过（τ-bench 一致性指标，衡量稳定性）
- wilson_95       全部 trial 通过率的 Wilson 95% 置信区间（小样本下比 ±σ 更靠谱）
- dimensions      各维度均值
- cost            平均工具调用数 / 墙钟 / token
以及按 family 拆分的同样指标。
"""
from __future__ import annotations

import math
from collections import defaultdict
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    return comb(c, k) / comb(n, k) if n >= k else 0.0


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _summarize(trials: list[dict]) -> dict:
    by_task = defaultdict(list)
    for t in trials:
        by_task[t["task_id"]].append(t)

    k = min(len(v) for v in by_task.values()) if by_task else 0
    p1, pk, phk = [], [], []
    for runs in by_task.values():
        n, c = len(runs), sum(r["reward"]["pass"] for r in runs)
        p1.append(c / n)
        pk.append(pass_at_k(n, c, k))
        phk.append(pass_hat_k(n, c, k))

    n_trials = len(trials)
    n_pass = sum(t["reward"]["pass"] for t in trials)
    dims = defaultdict(list)
    for t in trials:
        for d, s in t["reward"]["dimensions"].items():
            dims[d].append(s)

    lo, hi = wilson(n_pass, n_trials)
    out = {
        "n_tasks": len(by_task),
        "n_trials": n_trials,
        "k": k,
        "pass@1": _mean(p1),
    }
    if k > 1:  # k=1 时 f"pass@{k}" 就是 "pass@1"，不能覆盖
        out[f"pass@{k}"] = _mean(pk)
        out[f"pass^{k}"] = _mean(phk)
    return out | {
        "wilson_95": [lo, hi],
        "incomplete_trials": sum(t["reward"].get("incomplete", False) for t in trials),
        "error_trials": sum(bool(t["reward"]["stats"].get("error")) for t in trials),
        "dimensions": {d: _mean(v) for d, v in sorted(dims.items())},
        "cost": {
            "tool_calls": _mean([t["reward"]["stats"]["n_tool_calls"] for t in trials]),
            "redundancy": _mean([t["reward"]["stats"].get("redundancy") for t in trials]),
            "wall_time_s": _mean([t.get("metrics", {}).get("wall_time_s") for t in trials]),
            "input_tokens": _mean([t.get("metrics", {}).get("input_tokens") for t in trials]),
            "output_tokens": _mean([t.get("metrics", {}).get("output_tokens") for t in trials]),
        },
    }


def aggregate(trials: list[dict]) -> dict:
    by_family = defaultdict(list)
    for t in trials:
        by_family[t["family"]].append(t)
    return {
        "overall": _summarize(trials),
        "families": {f: _summarize(v) for f, v in sorted(by_family.items())},
    }
