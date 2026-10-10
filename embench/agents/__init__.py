"""
Agent 适配器
===========
所有被测对象统一为：async run(task, trial) -> Trajectory。

    oracle         回放 solution/trajectory.json（应 100% 通过；验证题目本身可解）
    nop            什么都不做（应 0% 通过；验证 verifier 不会白给分）
    replay         回放自备的轨迹文件（<dir>/<family>/<instance>.json），任何 agent 的通用接入方式

接入新 agent：继承 base.Agent 实现 async run(task, trial) -> Trajectory，并在 build_agent 中注册。
"""
from __future__ import annotations

import argparse

from .base import Agent


def build_agent(args: argparse.Namespace) -> Agent:
    name = args.agent
    if name == "oracle":
        from .simple import OracleAgent
        return OracleAgent()
    if name == "nop":
        from .simple import NopAgent
        return NopAgent()
    if name == "replay":
        from .simple import ReplayAgent
        if not args.replay_src or len(args.replay_src) != 1:
            raise SystemExit("replay 需要 --replay-src <轨迹目录>（目录结构：<family>/<instance>.json）")
        return ReplayAgent(args.replay_src[0])
    raise SystemExit(f"未知 agent: {name}")


AGENT_NAMES = ("oracle", "nop", "replay")
