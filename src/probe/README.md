# probe 模块说明

本目录实现 Phase 1 探针评估（确定性评分）。

- `loader.py`: 探针数据集加载与基础校验
- `evaluator.py`: 上下文构建、断言执行、结果汇总、Markdown 输出
- `models.py`: 数据结构定义

主入口：

- `evaluate_probes(...)`
- `write_probe_evaluation_md(...)`
