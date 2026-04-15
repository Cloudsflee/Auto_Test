# probe 模块说明

本目录实现探针评估引擎（Phase 1 确定性 + Phase 2 主观 LLM）。

- `loader.py`: 探针数据集加载与基础校验
- `llm_judge.py`: LLM 裁判调用、重试、重复采样与聚合
- `evaluator.py`: 上下文构建、断言执行、LLM 评分、结果汇总、Markdown 输出
- `models.py`: 数据结构定义

主入口：

- `evaluate_probes(...)`
- `write_probe_evaluation_md(...)`
