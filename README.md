# auto_test README（外部 AI 协作版）

这份 README 的目标不是“泛介绍”，而是让外部 AI 在最短时间内理解当前项目，并直接参与下一阶段核心设计：

1. 把“通用评估维度”做成默认底盘  
2. 把“记忆/压缩”做成可插拔 profile（按场景配权重，不再全局主导）

## 1. 你现在接手的是什么

`auto_test` 是一个多轮对话自动评测子项目，当前已经具备：

1. 自动跑会话（create -> execute -> SSE 解析 -> 落盘）
2. 自动导出 workspace（文本和图片，支持回退下载）
3. 对话整体 LLM 评估
4. 探针评估（deterministic + llm）

入口脚本：`auto_test/src/tests/run_5turn_session_test.py`

## 2. 当前目录（只列关键）

```text
auto_test/
  config/
    config.example.json
    config.local.json
  prompts/
    framework/                 # 通用机制层
    targets/<target_name>/     # 目标业务层（advoo / agent_x）
  datasets/probes/
    clinic_memory_v1.json      # deterministic
    clinic_memory_v2_mixed.json# deterministic + llm
  src/
    tests/run_5turn_session_test.py
    tests/user_simulator_engine.py
    tests/workspace_pipeline.py
    eval/dialogue_evaluator.py
    probe/evaluator.py
    probe/loader.py
    probe/llm_judge.py
  results/
    cleanup_keep_latest.py
```

## 3. 当前评估链路（真实实现）

### 3.1 对话整体评估（LLM）

实现文件：`src/eval/dialogue_evaluator.py`

当前约定输出字段为：

1. `pass`
2. `score_0_100`
3. `memory_score`
4. `coherence_score`
5. `findings`
6. `summary`

说明：字段能支持基本总评，但维度仍偏“记忆/连贯”，还不是通用评估底盘。

### 3.2 探针评估（Probe）

实现文件：`src/probe/evaluator.py`

当前支持两类 judge：

1. `judge_mode=deterministic`（规则断言）
2. `judge_mode=llm`（主观裁判）

并可按权重合并：

1. `deterministic_weight`（默认 0.65）
2. `llm_weight`（默认 0.35）

输出在 `run_data/probe_results.json`，含 `deterministic_score / llm_subjective_score / final_weighted_score`。

## 4. 现阶段核心问题（本 README 的重点）

当前系统虽可跑通，但评估目标仍偏“记忆/压缩导向”。  
下一阶段要从“专项测试”升级到“通用底盘 + 专项 profile”：

1. 通用底盘（默认启用）  
维度建议：任务完成、指令遵循、连贯性、安全、工具正确性

2. 记忆/压缩 profile（可插拔）  
按场景开启、设权重，不再默认压过所有维度

## 5. 你（外部 AI）要基于当前代码解决什么

请基于现有实现，设计并给出落地方案，目标是：

1. 保留现有执行链路、结果目录结构和兼容性
2. 引入“评估维度层”与“profile 层”的分离机制
3. 支持不同业务场景切换 profile（如 memory_heavy / safety_heavy / balanced）
4. 让总体分数由“底盘维度 + profile 维度”组合产生
5. 评估报告能明确显示：
   - 底盘分数
   - profile 分数
   - 最终加权分
   - 各维度权重来源（默认/配置/profile 覆盖）

## 6. 设计时必须遵守的约束

1. 不破坏已有结果产物路径（`results/session_autotest_<run_id>/...`）
2. 兼容现有配置（旧配置不填新字段时仍可运行）
3. 保留 `prompts/framework + prompts/targets` 分层
4. 保留 Probe 两种模式（deterministic + llm）
5. 新增机制必须可配置、可禁用、可回滚

## 7. 建议复用点（不要重复造轮子）

1. 维度化输出与 markdown 写入：`src/eval/dialogue_evaluator.py`
2. 混合评分与权重归一：`src/probe/evaluator.py`
3. 配置读取与 env 覆盖：`src/tests/run_5turn_session_test.py`
4. prompt 分层加载与 target 注入：`build_user_simulator_config(...)`

## 8. 我们希望的目标形态（不是实现，先是设计）

建议先产出一个统一评分模型（示例）：

1. `base_dimensions`（通用底盘）
2. `profiles`（可插拔专项）
3. `score_plan`（权重合并策略）

例如：

1. `base_dimensions`：task_completion / instruction_following / coherence / safety / tool_correctness
2. `profile.memory_compression`：memory_recall / compression_fidelity / state_continuity
3. `final_score = base_score * base_weight + profile_score * profile_weight`

并允许 profile 覆盖某些底盘维度权重，而不是写死。

## 9. 快速运行与验证

最小冒烟：

```bash
python auto_test/src/tests/run_5turn_session_test.py --max-turns 1
```

启用探针：

```bash
$env:AUTO_TEST_ENABLE_PROBE_EVAL="true"
python auto_test/src/tests/run_5turn_session_test.py --max-turns 10
```

启用 llm 探针裁判：

```bash
$env:AUTO_TEST_ENABLE_PROBE_EVAL="true"
$env:AUTO_TEST_ENABLE_PROBE_LLM_JUDGE="true"
python auto_test/src/tests/run_5turn_session_test.py --max-turns 10
```

清理旧结果（保留最新 4 个）：

```bash
python auto_test/results/cleanup_keep_latest.py --dry-run
python auto_test/results/cleanup_keep_latest.py
```

## 10. 给外部 AI 的可复制上下文（直接贴）

```text
你现在协助 auto_test 项目做“评估体系升级设计”，不是从零重做。

当前事实：
1) 入口是 auto_test/src/tests/run_5turn_session_test.py
2) prompts 采用 framework + targets 分层
3) 对话整体评估在 src/eval/dialogue_evaluator.py，当前字段含 pass/score_0_100/memory_score/coherence_score/findings/summary
4) 探针评估在 src/probe/evaluator.py，支持 deterministic + llm，并做加权融合
5) 结果输出固定在 auto_test/results/session_autotest_<run_id>/

我要你基于当前项目设计：
A. 通用评估底盘（任务完成、指令遵循、连贯性、安全、工具正确性）
B. 记忆/压缩作为可插拔 profile（按场景启用与配权，不再全局主导）

约束：
1) 兼容现有配置和结果结构
2) 不破坏当前执行链路
3) 可渐进上线（可开关、可回滚）

请先给：
1) 数据模型与配置模型（字段级）
2) 评分流程图（从 turn_results 到 final_score）
3) 与现有文件的改造映射（按文件列改动）
4) 迁移策略（旧配置如何无痛兼容）
5) 最小验证方案（命令 + 验收标准）
```

## 11. 相关文档

1. `auto_test/config/README.md`
2. `auto_test/prompts/README.md`
3. `auto_test/datasets/probes/README.md`
4. `auto_test/doc/STABLE_INTERFACE_AND_RESULT_SPEC.md`
5. `auto_test/doc/EVAL_INTEGRATION.md`
6. `auto_test/Phase1-探针体系MVP详细设计.md`
7. `auto_test/Phase2-主观探针LLM评分详细设计.md`
