"""
公开工具（open tools）
====================
EMBench 种子题用到的 11 个工具中，能只用开源库实现的部分在这里给出真实实现：

    nameToSmiles            内置常用分子名 + PubChem 在线查询
    validateSmiles          RDKit 解析，返回分子式、原子数、键数
    drawMolecule            RDKit 绘制 2D 结构图（PNG）
    smilesTo3D              RDKit 生成 3D 坐标（ETKDG + MMFF 优化）
    calculateOxygenBalance  按分子式计算氧平衡 OB_CO / OB_CO2

其余工具（KJ 爆轰计算、性质预测、分子生成、晶体结构生成）依赖不公开的模型或受许可证限制的代码，
由 fixture 回放题目里记录好的结果。
"""
from .open_tools import OPEN_TOOLS, call_open_tool

__all__ = ["OPEN_TOOLS", "call_open_tool"]
