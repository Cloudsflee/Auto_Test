# 通用评估底盘 + 可插拔记忆/压缩 Profile：已改动整理

## 1. 范围说明

本文件只整理这一条主线已落地的改动：  
“让外部 AI 专门帮你设计通用评估底盘 + 可插拔记忆/压缩 profile”。

---

## 2. 目标到实现的映射

## 2.1 通用评估底盘（Foundation）

已落地：

1. 在影子评估中固定了通用维度：
  - `task_completion`
  - `instruction_following`
  - `coherence`
  - `safety`
  - `tool_correctness`
2. 支持底盘权重配置与归一化计算。
3. 底盘分可作为主判定来源（`primary_mode=foundation_v2`）。

关键实现文件：

1. `auto_test/src/eval/score_orchestrator.py`
2. `auto_test/src/tests/run_5turn_session_test.py`

## 2.2 可插拔记忆/压缩 profile

已落地：

1. `memory_compression` profile 维度化评估：
  - `memory_recall`
  - `compression_fidelity`
  - `state_continuity`
2. 支持 profile 权重、启停、fallback（profile 不可用时回退到底盘）。
3. 支持 profile 与底盘融合得分（`final_v2`）。

关键实现文件：

1. `auto_test/src/eval/score_orchestrator.py`
2. `auto_test/src/tests/run_5turn_session_test.py`

---

## 3. 分阶段改动摘要

## Phase A（影子评估）

新增：

1. `evaluation_v2_shadow` 输出（不替换旧字段）。
2. `score_orchestrator.py` 作为新评分编排层。

## Phase B（主判定切换能力）

新增：

1. `evaluation_primary` 输出。
2. `llm_eval.primary_mode` 支持：
  - `llm_v1`
  - `foundation_v2`
  - `final_v2`

## Phase C（A/B 对比可观测）

新增：

1. `evaluation_compare` 输出（`llm_v1/foundation_v2/final_v2`）。
2. `evaluation.md` 增加 `Mode Comparison (A/B)` 区块。
3. 对比分差字段：
  - `foundation_minus_llm_v1`
  - `final_minus_foundation`
  - `final_minus_llm_v1`

## Phase D（多 profile + 路由第一版）

新增：

1. 多 profile 配置能力：
  - `profile.active_profiles`
  - `profile.active_profiles_by_capability_mode`
  - `profiles.<name>.merge_weight`
2. capability_mode 路由选择 profile（由 user_simulator 模式驱动）。
3. `evaluation_v2_shadow` 新增可观测节点：
  - `profile_router`
  - `profile_combined`
  - `profiles.details`

---

## 4. 关键文件改动清单

## 4.1 代码

1. `auto_test/src/eval/score_orchestrator.py`
  - 新增/扩展评分编排逻辑（底盘、profile、多 profile 合并、路由上下文）。
2. `auto_test/src/eval/dialogue_evaluator.py`
  - `LLMEvalConfig` 扩展（foundation/profile 多字段）。
  - 评估 markdown 增加 A/B 与 profile 路由/合并展示。
3. `auto_test/src/tests/run_5turn_session_test.py`
  - 解析新配置。
  - 构建 `evaluation_primary`、`evaluation_compare`。
  - 运行时注入 profile 路由上下文。

## 4.2 配置

1. `auto_test/config/config.example.json`
2. `auto_test/config/config.local.json`

新增配置键（核心）：

1. `llm_eval.primary_mode`
2. `llm_eval.shadow_pass_threshold_0_100`
3. `llm_eval.profile.active_profiles`
4. `llm_eval.profile.active_profiles_by_capability_mode`
5. `llm_eval.profiles.<name>.merge_weight`

## 4.3 文档

1. `auto_test/README.md`（Phase C / D 更新说明）
2. `auto_test/config/README.md`（新配置项说明）
3. `auto_test/doc/EVAL_INTEGRATION.md`（输出结构与阶段说明）

---

## 5. 结果结构变化（保持兼容）

保持不变（旧消费者可继续读取）：

1. `llm_evaluation.response_json.pass/score_0_100/...`

新增（向后兼容）：

1. `evaluation_v2_shadow`
2. `evaluation_primary`
3. `evaluation_compare`
4. `evaluation_v2_shadow.profile_router`
5. `evaluation_v2_shadow.profile_combined`
6. `evaluation_v2_shadow.profiles.details`

---

## 6. 已完成验证（样例）

已跑通并验证字段落盘的 run（示例）：

1. `session_autotest_20260416_054614_ec646a34`（Phase C 验证）
2. `session_autotest_20260416_061244_a4f698b1`（Phase D 基础验证）
3. `session_autotest_20260416_061340_a4bbb500`（Phase D 路由覆盖验证）
4. `session_autotest_20260416_061805_592173ed`（5轮回归，workspace md/图片导出正常）

---

## 7. 当前结论

这条主线的核心改造已从“设计”推进到“可运行 + 可观测 + 可配置 + 可回滚”的阶段：

1. 通用底盘已独立成评分底层。
2. 记忆/压缩 profile 已可插拔并参与融合。
3. 主判定、A/B 对比、多 profile 路由与合并已形成完整输出链路。
4. 旧结果结构未被破坏。

