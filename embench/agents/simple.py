"""内置适配器：oracle / nop / replay。"""
from __future__ import annotations

import json
from pathlib import Path

from ..core.task import Task
from ..core.trajectory import Trajectory
from .base import Agent


class OracleAgent(Agent):
    name = "oracle"

    async def run(self, task: Task, trial: int) -> Trajectory:
        traj = Trajectory.load(task.solution_path)
        traj.task_id = task.id
        traj.agent = self.info()
        return traj


class NopAgent(Agent):
    name = "nop"

    async def run(self, task: Task, trial: int) -> Trajectory:
        return Trajectory(task_id=task.id, agent=self.info())


class ReplayAgent(Agent):
    """回放你自己的 agent 产出的轨迹文件（任何 agent / 任何语言都能接入的通用入口）。

    目录结构与 tasks/ 一一对应：<src>/<family>/<instance>.json，内容为统一轨迹格式
    （见 core/trajectory.py）。某题没有文件 → 记为错误并判失败。
    """
    name = "replay"

    def __init__(self, source: str):
        self.source = Path(source)

    def info(self) -> dict:
        return {"name": self.name, "source": self.source.as_posix()}

    async def run(self, task: Task, trial: int) -> Trajectory:
        path = self.source / f"{task.id}.json"
        if not path.is_file():
            return Trajectory(task_id=task.id, agent=self.info(), error=f"没有找到轨迹文件 {path.as_posix()}")
        traj = Trajectory.load(path)
        traj.task_id = task.id
        traj.agent = {**self.info(), **traj.agent}
        return traj
