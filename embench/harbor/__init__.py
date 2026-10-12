"""
Harbor 集成
==========
把 EMBench 题目导出为 Harbor 题目，在 Docker 里真实运行任意 agent：

- 模拟工具服务（fixtures.py + runtime/tool_server.py）：按参数回放参考轨迹的工具结果，
  并在服务端记录每次调用；判分以这份记录为准，agent 无法伪造
- 判分（runtime/grade.py）：取服务端调用记录 + agent 写下的最终回答，用 EMBench 声明式检查判分
- 导出（export.py）：python -m embench export-harbor -o <目录>
"""
