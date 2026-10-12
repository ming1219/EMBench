"""
判分入口：由 tests/test.sh 在判分阶段调用。

1. 用判分阶段才注入的密钥（EMBENCH_LOG_TOKEN），向 tool-server 取服务端调用记录
2. 读取 agent 写下的最终回答（/app/answer.md）
3. 两者还原成轨迹，用题目自带的 EMBench 声明式检查判分
4. 写 /logs/verifier/reward.txt（0/1，Harbor 的主指标）和 /logs/verifier/embench_reward.json（多维明细）

取不到调用记录时整题判 0：没有证据不能算通过。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embench.core.task import load_task  # noqa: E402
from embench.core.verifier import verify  # noqa: E402
from embench.harbor.fixtures import trajectory_from_calls  # noqa: E402


def fetch_calls(url: str, token: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"X-Embench-Token": token})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def grade(task, calls: list[dict] | None, answer: str, fetch_error: str | None = None) -> dict:
    if calls is None:
        return {"task_id": task.id, "pass": False, "reward": 0.0, "dimensions": {}, "incomplete": True,
                "missing_evidence": ["tool_call_log"], "invalid_evidence": [], "checks": [],
                "stats": {"n_tool_calls": 0, "error": f"无法获取服务端调用记录: {fetch_error}"},
                "task_hash": task.content_hash(), "evidence": "server-side tool log"}
    traj = trajectory_from_calls(calls, answer, task.id)
    reward = verify(task, traj)
    reward["evidence"] = "server-side tool log"
    return reward


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task-root", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--answer", default="/app/answer.md")
    p.add_argument("--calls-url", default="http://tool-server:8000/embench/calls")
    p.add_argument("--out", default="/logs/verifier")
    a = p.parse_args(argv)

    root = Path(a.task_root)
    task = load_task(root / a.task_id, root)
    answer_path = Path(a.answer)
    answer = answer_path.read_text(encoding="utf-8") if answer_path.is_file() else ""
    try:
        calls, err = fetch_calls(a.calls_url, os.environ.get("EMBENCH_LOG_TOKEN", "")), None
    except Exception as e:  # 取不到记录 = 没有证据
        calls, err = None, f"{type(e).__name__}: {e}"

    reward = grade(task, calls, answer, err)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "embench_reward.json").write_text(json.dumps(reward, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "reward.txt").write_text("1\n" if reward["pass"] else "0\n", encoding="utf-8")
    failed = [c["id"] + ": " + c["detail"] for c in reward.get("checks", []) if c.get("passed") is False]
    print(f"EMBench {task.id}: {'PASS' if reward['pass'] else 'FAIL'}  calls={reward['stats']['n_tool_calls']}")
    for line in failed + ([reward["stats"]["error"]] if reward["stats"].get("error") else []):
        print("  -", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
