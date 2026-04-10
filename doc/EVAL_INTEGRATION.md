# 评估集成说明（精简版）

## 1. 文档定位

本文档只描述评估本身的稳定约定，不绑定具体脚本实现细节。

---

## 2. 评估输出

每次测试建议至少输出：

- 文本报告：`evaluation.md`
- 结构化结果：`non_text/evaluation.json`

其中：

- `evaluation.md` 便于人工快速查看
- `evaluation.json` 便于程序统计与趋势分析

---

## 3. 当前模式：LLM-only（必选）

当前默认模式为 LLM-only，不再依赖固定规则评分与固定事实集。

要求：

- 必须启用并配置 `llm_eval`
- 评估结论来自 `llm_evaluation` 字段
- `evaluation_mode` 为 `llm_only`

关键配置（推荐写入 `config/*.json`）：

- `llm_eval.enabled=true`
- `llm_eval.base_url=...`
- `llm_eval.model=...`
- `llm_eval.api_key=...`
- `llm_eval.timeout_sec=...`

---

## 4. 用户模拟器（推荐）

测试执行阶段建议启用 `user_simulator`（默认 `provider_context`）生成多样化用户输入：

- `user_simulator.enabled=true`
- `user_simulator.mode=provider_context`
- `user_simulator.base_url=...`
- `user_simulator.model=...`
- `user_simulator.api_key=...`
- `user_simulator.max_turns=...`

提示词建议放在：

- `prompts/user_simulator_system.prompt`
- `prompts/user_simulator_scenario.prompt`

---

## 5. 记忆/压缩专项扩展建议（保留）

建议在当前 smoke 评估基础上增加：

1. 长会话场景（如 50+ 轮）
2. 压缩触发前后 checkpoint 问答
3. 多主题切换后的事实回收测试

建议指标：

- 事实召回率
- 事实冲突率
- 主题回切恢复质量

如后端可提供压缩元数据，建议同时记录：

- 触发轮次
- 压缩前后 token 大小
- 摘要版本或摘要 ID
- 保留/丢弃事实标识
