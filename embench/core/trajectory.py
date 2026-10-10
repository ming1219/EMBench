"""
统一轨迹格式
============
所有 agent 适配器都产出同一种 Trajectory（「答题卡」），verifier 只认这一种格式：

    {
      "schema_version": "1.0",
      "task_id": "ob-kj/tnt",
      "agent": {"name": "my-agent", "model": "some-llm"},
      "steps": [{"tool": "calculateKJ", "args": {...}, "result": "<str|obj>"}, ...],
      "final_answer": "...",
      "error": null,
      "metrics": {"wall_time_s": 12.3, "input_tokens": 0, "output_tokens": 0}
    }

有些框架的日志把调用与结果分开存（tool_calls 列表 + 按工具名聚合的 tool_results），
from_legacy() 负责把第 k 次调用与该工具的第 k 个结果配对，转成本格式。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass
class Step:
    tool: str
    args: dict = field(default_factory=dict)
    result: Any = None
    has_result: bool = True

    def to_dict(self) -> dict:
        d = {"tool": self.tool, "args": self.args}
        if self.has_result:
            d["result"] = self.result
        return d


@dataclass
class Trajectory:
    task_id: str = ""
    steps: list[Step] = field(default_factory=list)
    final_answer: Any = ""
    error: str | None = None
    agent: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    # ---------- 查询辅助 ----------
    def calls(self, tool: str | None = None) -> list[Step]:
        return [s for s in self.steps if tool is None or s.tool == tool]

    @property
    def tool_names(self) -> list[str]:
        return [s.tool for s in self.steps]

    @property
    def has_any_result(self) -> bool:
        return any(s.has_result for s in self.steps)

    # ---------- 序列化 ----------
    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.task_id,
            "agent": self.agent,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "error": self.error,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        steps = [
            Step(tool=s["tool"], args=s.get("args") or {}, result=s.get("result"), has_result="result" in s)
            for s in d.get("steps", [])
        ]
        return cls(
            task_id=d.get("task_id", ""),
            steps=steps,
            final_answer=d.get("final_answer", ""),
            error=d.get("error"),
            agent=d.get("agent", {}),
            metrics=d.get("metrics", {}),
        )

    @classmethod
    def load(cls, path: Path) -> "Trajectory":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_legacy(cls, workflow: dict, task_id: str = "") -> "Trajectory":
        """旧格式 {"tool_calls":[{name:args}], "tool_results":{name:[...]}, "result":...} → Trajectory。"""
        results = {k: list(v) for k, v in (workflow.get("tool_results") or {}).items()}
        steps = []
        for call in workflow.get("tool_calls", []):
            if not isinstance(call, dict) or not call:
                continue
            name, args = next(iter(call.items()))
            queue = results.get(name) or []
            if queue:
                steps.append(Step(name, args or {}, queue.pop(0), True))
            else:  # 日志里没记录结果：标记为「无结果」，L3 检查据此 skip 而非判错
                steps.append(Step(name, args or {}, None, False))
        final = workflow.get("result", "")
        error = final if isinstance(final, str) and final.startswith("ERROR") else None
        return cls(task_id=task_id, steps=steps, final_answer=final, error=error)
