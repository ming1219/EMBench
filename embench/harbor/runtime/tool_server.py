"""
EMBench fixture tool server（运行在 Harbor 题目的 tool-server 容器里）
=================================================================
- 按 tools.json 动态注册 MCP 工具（streamable-http，地址 http://tool-server:8000/mcp）
- 每次调用按参数回放 fixtures.json 里记录的结果（匹配规则见 embench/harbor/fixtures.py）；
  EMBENCH_TOOLS=open 时，公开工具（embench/tools/open_tools.py）真实计算，其余工具仍回放记录
- 每次调用都在**服务端**记录（内存 + /var/log/embench/calls.jsonl）。agent 运行在另一个容器里，
  读不到也改不了这份记录
- GET /embench/calls：返回调用记录，必须带请求头 X-Embench-Token（密钥只在判分阶段注入）
"""
from __future__ import annotations

import inspect
import json
import os
import threading
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from embench.harbor.fixtures import answer_call

HERE = Path(__file__).resolve().parent
TOOLS = json.loads(Path(os.environ.get("EMBENCH_TOOLS_JSON", HERE / "tools.json")).read_text(encoding="utf-8"))["tools"]
FIXTURES = json.loads(Path(os.environ.get("EMBENCH_FIXTURES", HERE / "fixtures.json")).read_text(encoding="utf-8"))
TOKEN = os.environ["EMBENCH_LOG_TOKEN"]
MODE = os.environ.get("EMBENCH_TOOLS", "fixture")  # fixture：全部回放；open：公开工具真实计算
PORT = int(os.environ.get("EMBENCH_PORT", "8000"))
HOST = os.environ.get("EMBENCH_HOST", "0.0.0.0")
if MODE == "open":
    from embench.tools import OPEN_TOOLS, call_open_tool
else:
    OPEN_TOOLS, call_open_tool = set(), None
LOG_PATH = Path(os.environ.get("EMBENCH_CALL_LOG", "/var/log/embench/calls.jsonl"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()
_calls: list[dict] = []

_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool,
          "string_list": list[str] | str, "object": dict}


def _record(tool: str, args: dict, result) -> None:
    with _lock:
        entry = {"seq": len(_calls), "ts": time.time(), "tool": tool, "args": args, "result": result}
        _calls.append(entry)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _make_tool(spec: dict):
    """按 spec 生成带显式签名的函数，FastMCP 据此生成参数 schema。

    可选参数一律以 None 作缺省，这样只记录 agent 真正传入的参数；默认值写在参数说明里。
    """
    required = {p["name"] for p in spec["parameters"] if p.get("required")}
    params = []
    for p in spec["parameters"]:
        ann = _TYPES[p["type"]]
        if p["name"] in required:
            params.append(inspect.Parameter(p["name"], inspect.Parameter.KEYWORD_ONLY, annotation=ann))
        else:
            params.append(inspect.Parameter(p["name"], inspect.Parameter.KEYWORD_ONLY,
                                            default=None, annotation=ann | None))

    def tool(**kwargs) -> str:
        args = {k: v for k, v in kwargs.items() if v is not None or k in required}
        if spec["name"] in OPEN_TOOLS:
            result = call_open_tool(spec["name"], args)
        else:
            result = answer_call(FIXTURES, spec["name"], args)
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)
        _record(spec["name"], args, result)
        return result

    tool.__name__ = spec["name"]
    tool.__signature__ = inspect.Signature(params, return_annotation=str)
    tool.__annotations__ = {p.name: p.annotation for p in params} | {"return": str}
    return tool


def _describe(spec: dict) -> str:
    lines = [spec["description"]]
    for p in spec["parameters"]:
        default = "" if p.get("required") or p.get("default") is None else f" (default: {p['default']})"
        lines.append(f"- {p['name']}: {p.get('description', '')}{default}")
    return "\n".join(lines)


mcp = FastMCP(
    "embench-tools",
    host=HOST,
    port=PORT,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)
for _spec in TOOLS:
    mcp.add_tool(_make_tool(_spec), name=_spec["name"], description=_describe(_spec))


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


@mcp.custom_route("/embench/calls", methods=["GET"])
async def calls(request: Request) -> JSONResponse:
    if request.headers.get("X-Embench-Token") != TOKEN:
        return JSONResponse({"error": "forbidden"}, status_code=403)
    with _lock:
        return JSONResponse(list(_calls))


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
