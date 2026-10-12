"""
模拟工具（fixture）的匹配规则与调用记录
====================================
Harbor 题目里的工具由 fixture server 提供：它不运行真实模型，而是按参数回放参考轨迹里记录的结果。

- fixtures：{"local": [...], "global": [...]}，每条 {"tool", "args", "result"}
  local  = 本题参考轨迹里的调用；global = 所有题目参考轨迹里的调用（agent 走了别的合理路径时也能拿到结果）
- 匹配：先 local 后 global；工具名相同，且 fixture 声明的每个参数都按 EMBench 的归一规则相等
  （SMILES 写法不同、列表传成 JSON 字符串等都视为相同；agent 多传的参数不影响匹配）
- 找不到：返回结构化错误 {"error_type": "fixture_missing", ...}，同样计入调用记录

调用记录（calls log）由工具服务在服务端写入，agent 无法修改；判分时据此还原轨迹。
"""
from __future__ import annotations

import json
from typing import Any

from ..core.checks import args_match
from ..core.trajectory import Step, Trajectory

FIXTURE_MISSING = "fixture_missing"


def build_fixtures(local_steps: list[dict], global_steps: list[dict]) -> dict:
    """去重后的 fixture 表。同一工具 + 同一参数（按 JSON 序列化）只保留第一条。"""
    def dedupe(steps):
        seen, out = set(), []
        for s in steps:
            if "result" not in s:
                continue
            key = (s["tool"], json.dumps(s.get("args", {}), sort_keys=True, ensure_ascii=False))
            if key not in seen:
                seen.add(key)
                out.append({"tool": s["tool"], "args": s.get("args", {}), "result": s["result"]})
        return out
    return {"local": dedupe(local_steps), "global": dedupe(global_steps)}


def _specificity(fixture: dict) -> int:
    return len(fixture.get("args") or {})


def match_fixture(fixtures: dict, tool: str, args: dict) -> dict | None:
    """返回匹配的 fixture；声明参数越多（越具体）的越优先，local 优先于 global。"""
    for pool in ("local", "global"):
        hits = [f for f in fixtures.get(pool, []) if f["tool"] == tool and args_match(args, f.get("args") or {})[0]]
        if hits:
            return max(hits, key=_specificity)
    return None


def answer_call(fixtures: dict, tool: str, args: dict) -> Any:
    hit = match_fixture(fixtures, tool, args or {})
    if hit is not None:
        return hit["result"]
    return json.dumps({
        "error_type": FIXTURE_MISSING,
        "error": f"This benchmark environment has no recorded result for {tool} with these arguments.",
        "args": args,
    }, ensure_ascii=False)


def trajectory_from_calls(calls: list[dict], final_answer: str, task_id: str = "",
                          agent: dict | None = None) -> Trajectory:
    """服务端调用记录 + agent 写下的最终回答 → EMBench 轨迹。"""
    steps = [Step(c["tool"], c.get("args") or {}, c.get("result"), True)
             for c in sorted(calls, key=lambda c: c.get("seq", 0))]
    return Trajectory(task_id=task_id, steps=steps, final_answer=final_answer or "",
                      agent=agent or {"name": "harbor", "evidence": "server-side tool log"})
