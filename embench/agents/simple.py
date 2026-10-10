"""内置适配器：oracle / nop / replay。"""
from __future__ import annotations

import json
from pathlib import Path

from ..core.task import Task
from ..core.trajectory import Trajectory
from .base import Agent


class OracleAgent(Agent):
    name = "oracle"
    deterministic = True

    async def run(self, task: Task, trial: int) -> Trajectory:
        traj = Trajectory.load(task.solution_path)
        traj.task_id = task.id
        traj.agent = self.info()
        return traj


class NopAgent(Agent):
    name = "nop"
    deterministic = True

    async def run(self, task: Task, trial: int) -> Trajectory:
        return Trajectory(task_id=task.id, agent=self.info())


class ReplayAgent(Agent):
    """回放你自己的 agent 产出的轨迹文件（任何 agent / 任何语言都能接入的通用入口）。

    目录结构与 tasks/ 一一对应，内容为统一轨迹格式（见 core/trajectory.py）：

        <src>/<family>/<instance>.json              只有一次试验时（视为第 0 次）
        <src>/<family>/<instance>/trial_<k>.json    多次试验时，每次一份独立记录

    多次试验必须各有一份独立记录，同一份文件不会被重复计为多次试验；
    缺少某次试验的文件 → 记为错误并判失败。
    """
    name = "replay"

    def __init__(self, source: str):
        self.source = Path(source)

    def info(self) -> dict:
        return {"name": self.name, "source": self.source.resolve().as_posix()}

    def trial_path(self, task: Task, trial: int) -> Path | None:
        per_trial = self.source / task.id / f"trial_{trial}.json"
        if per_trial.is_file():
            return per_trial
        single = self.source / f"{task.id}.json"
        return single if trial == 0 and single.is_file() else None

    async def run(self, task: Task, trial: int) -> Trajectory:
        path = self.trial_path(task, trial)
        if path is None:
            want = f"{task.id}/trial_{trial}.json" + ("" if trial else f" 或 {task.id}.json")
            return Trajectory(task_id=task.id, agent=self.info(), error=f"没有找到第 {trial} 次试验的轨迹文件（{want}）")
        traj = Trajectory.load(path)
        traj.task_id = task.id
        traj.agent = {**self.info(), **traj.agent}
        return traj
