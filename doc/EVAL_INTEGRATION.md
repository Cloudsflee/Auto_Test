# 评估集成说明（精简版）

## 1. 文档定位

本文档只描述评估本身的稳定约定，不绑定具体脚本实现细节。

---

## 2. 评估输出

每次测试建议至少输出：

- 文本报告：`evaluation.md`
- 结构化结果：`run_data/evaluation.json`
- 探针结构化结果（可选）：`run_data/probe_results.json`

其中：

- `evaluation.md` 便于人工快速查看
- `evaluation.json` 便于程序统计与趋势分析
- `evaluation.json.evaluation_v2_shadow`（若存在）用于并行观察“通用底盘 + profile”评分，不影响旧结论字段
- `evaluation.json.evaluation_primary` 为当前主判定（可配置来源）

探针评估（开启时）支持两类评分：

- `judge_mode=deterministic`：字符串/正则/路径/字段断言
- `judge_mode=llm`：主观探针六维评分（仅 continuation/decision）

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
- `llm_eval.primary_mode=foundation_v2`（推荐）
- `llm_eval.foundation.*`（影子模式：通用底盘维度与权重）
- `llm_eval.profile.*`（影子模式：profile 激活与融合权重）
- `llm_eval.profiles.memory_compression.*`（影子模式：记忆/压缩 profile 维度权重）

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

- `prompts/framework/simulator/system/user_simulator_system.prompt`
- `prompts/targets/<target_name>/scenarios/intent_pool.yaml`
- `prompts/targets/<target_name>/scenarios/policies.yaml`

---

## 5. 记忆/压缩专项扩展建议（保留）

## 5.1 Probe LLM 裁判开关（Phase 2）

- `AUTO_TEST_ENABLE_PROBE_EVAL=true`
- `AUTO_TEST_ENABLE_PROBE_LLM_JUDGE=true`（可选）
- `AUTO_TEST_PROBE_LLM_URL / MODEL / API_KEY`（可选）

输出中可关注：

- `summary.deterministic_score`
- `summary.llm_subjective_score`
- `summary.final_weighted_score`
- `summary.by_judge_mode`
- `results[].llm_judge.aggregate`

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

---

## Phase C Update (A/B Compare)

- `run_data/evaluation.json` now includes `evaluation_compare`.
- `evaluation_compare.llm_v1`: legacy score/pass from `llm_evaluation.response_json`.
- `evaluation_compare.foundation_v2`: base-foundation score/pass from `evaluation_v2_shadow.foundation`.
- `evaluation_compare.final_v2`: merged score/pass from `evaluation_v2_shadow.final`.
- `evaluation_compare.delta`: score deltas for `foundation_minus_llm_v1`, `final_minus_foundation`, `final_minus_llm_v1`.
- `evaluation.md` now includes `## Mode Comparison (A/B)` for quick human comparison.
- Backward compatibility remains unchanged: legacy fields and `evaluation_primary` are still preserved.

## Phase D Update (Multi-Profile + Routing)

- Added config-driven profile routing by capability mode.
- Added multi-profile score composition within `evaluation_v2_shadow`.
- Added new observability nodes:
  - `profile_router`
  - `profile_combined`
  - `profiles.details`
- Compatibility note: `profile` legacy summary is still kept for old consumers.
