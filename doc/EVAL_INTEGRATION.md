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

## 3. 默认规则评估（必选）

当前基线规则：

- `run_success`：轮次完成且无 `RUN_ERROR`
- `non_empty_response`：助手回复非空
- `no_tool_leak`：用户可见文本不泄漏工具载荷
- `memory_recall_turn4`：指定轮次记忆召回
- `summary_coverage_turn5`：总结轮次覆盖关键事实

评分方式：

- 加权得分区间 `[0, 1]`
- 默认通过阈值：`0.80`
- 可通过环境变量覆盖：`AUTO_TEST_EVAL_PASS_THRESHOLD`

---

## 4. 可选 LLM 评估（增强）

默认关闭，可通过环境变量开启：

- `AUTO_TEST_ENABLE_LLM_EVAL=true`
- `AUTO_TEST_EVAL_LLM_URL=...`
- `AUTO_TEST_EVAL_LLM_MODEL=...`
- `AUTO_TEST_EVAL_LLM_API_KEY=...`
- `AUTO_TEST_EVAL_LLM_TIMEOUT_SEC=30`（可选）

启用后，LLM 评估结果应写入：

- `non_text/evaluation.json`
- `evaluation.md`

---

## 5. 记忆/压缩专项扩展建议

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

