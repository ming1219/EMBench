"""
参考解（oracle）：在 agent 容器里通过 MCP 真实调用工具，复现参考轨迹，最后写出参考回答。
由 solution/solve.sh 调用；Harbor 用它证明「这道题可解，且判分器给满分」。
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

HERE = Path(__file__).resolve().parent
MCP_URL = os.environ.get("EMBENCH_MCP_URL", "http://tool-server:8000/mcp")
ANSWER_PATH = Path(os.environ.get("EMBENCH_ANSWER_PATH", "/app/answer.md"))


async def main() -> None:
    reference = json.loads((HERE / "trajectory.json").read_text(encoding="utf-8"))
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for step in reference["steps"]:
                await session.call_tool(step["tool"], step.get("args") or {})
    ANSWER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANSWER_PATH.write_text(reference.get("final_answer") or "", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
