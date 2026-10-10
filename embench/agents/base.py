from __future__ import annotations

from ..core.task import Task
from ..core.trajectory import Trajectory


class Agent:
    name = "base"
    # 每次运行结果都完全相同（如回放固定答案）。这类 agent 不允许 -k > 1，
    # 否则同一份结果被重复计数，会虚增样本量。
    deterministic = False

    def info(self) -> dict:
        """写进 job manifest 的被测对象描述（模型、模式等）。"""
        return {"name": self.name}

    async def setup(self) -> None:
        pass

    async def run(self, task: Task, trial: int) -> Trajectory:
        raise NotImplementedError

    async def teardown(self) -> None:
        pass
