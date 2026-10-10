"""
EMBench — 工具调用智能体评测框架
========================================
采用「任务即目录」的组织方式：

    tasks/<family>/<instance>/
        task.toml          元数据 + 声明式 verifier（检查项列表）
        instruction.md     给智能体的自然语言指令
        solution/trajectory.json   oracle 参考轨迹（必须 100% 通过 verifier）
        solution/negatives/*.json  可选：必须判失败的反例轨迹

核心层（core/）只依赖 Python 标准库（RDKit 可选，用于分子写法归一），可离线运行与单测；
被测智能体通过 agents/ 中的适配器接入，最通用的是 replay：回放你自己的智能体产出的轨迹文件。
"""

__version__ = "0.1.0"
