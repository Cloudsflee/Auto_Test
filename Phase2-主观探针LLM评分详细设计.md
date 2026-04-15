# Phase 2：主观探针 LLM 评分详细设计（延续/决策）

## 1. 文档定位

本文档是 `auto_test/递进开发方案.md` 中 Phase 2 的工程化落地稿，目标是在 **不破坏 Phase 1 确定性探针能力** 的前提下，引入“仅针对主观探针”的 LLM 评分体系。

本阶段聚焦：

1. `continuation / decision` 探针的 LLM 评分
2. 与现有 `src/probe`、`run_5turn_session_test.py` 的最小侵入集成
3. 评分稳定性、可复核性和可配置性

不包含：

1. `pass@k / pass^k`（Phase 3）
2. Elo 对战（Phase 5）

---

## 2. 当前基线与差距

基于当前项目代码（2026-04-15）：

1. Phase 1 已具备 `datasets/probes/*.json` + `src/probe/{loader,evaluator,models}.py`
2. `judge_mode` 在 loader 中被强约束为 `deterministic`
3. 结果已稳定输出到：
   - `results/<run>/run_data/probe_results.json`
   - `results/<run>/probe_evaluation.md`
4. 主流程已支持代理配置、workspace 导出、LLM 对话总评

差距在于：

1. 无法表达 `judge_mode=llm` 的探针
2. 无法按六维标准对“延续性/决策性”进行主观评分
3. 无稳定性控制（重复采样、方差、失败降级）

---

## 3. 设计目标

## 3.1 业务目标

1. 将“压缩后能否继续工作、是否保留决策理由”从被动观察转为可量化评分
2. 保持“确定性优先，LLM 补主观”的混合范式
3. 输出可审计：每个分数都能追溯到 probe、turn、证据片段与评分理由

## 3.2 工程目标

1. 对 Phase 1 数据集完全兼容（老数据集无需修改即可继续跑）
2. 新增能力默认关闭，通过配置显式开启
3. 失败可控：支持 fail-closed（严格失败）和 fail-open（标记跳过）

---

## 4. 总体方案

采用“**两段式评估**”：

1. 第一段：确定性评估（现有逻辑，覆盖 recall/artifact + 部分 continuation/decision）
2. 第二段：LLM 主观评估（仅 continuation/decision 且 `judge_mode=llm` 的探针）

总分拆分为三个值：

1. `deterministic_score`：确定性探针加权分
2. `llm_subjective_score`：LLM 主观探针加权分
3. `final_weighted_score`：两者按权重融合后的总分

---

## 5. 数据契约扩展（Probe Dataset）

## 5.1 Probe 新增字段

在现有 probe schema 上扩展（兼容旧字段）：

```json
{
  "probe_id": "decision_strategy_consistency_v1",
  "probe_type": "decision",
  "judge_mode": "llm",
  "weight": 1.2,
  "critical": false,
  "target": {
    "source": "turn_assistant_text",
    "turn": 5
  },
  "llm_judge": {
    "rubric_id": "probe_subjective_v1",
    "pass_threshold_0_5": 3.2,
    "dimensions": [
      "accuracy",
      "context_awareness",
      "artifact_trail",
      "completeness",
      "continuity",
      "instruction_following"
    ],
    "turn_window": {
      "start": 1,
      "end": 5
    },
    "require_evidence_paths": true
  },
  "description": "检查助手在最终决策中是否与历史上下文一致"
}
```

## 5.2 兼容策略

1. `judge_mode=deterministic`：沿用现有 `assertions` 必填规则
2. `judge_mode=llm`：
   - `assertions` 可为空
   - 必须存在 `llm_judge` 配置
3. 未识别 `judge_mode`：按数据集错误处理

---

## 6. 评分 Rubric（六维 0-5）

Phase 2 固化六维（来自 `doc/测试.md`）：

1. `accuracy`：事实/细节是否正确
2. `context_awareness`：是否理解并利用上下文
3. `artifact_trail`：是否能追踪文件与制品状态
4. `completeness`：是否完整覆盖问题要求
5. `continuity`：是否能在现有记忆上持续推进
6. `instruction_following`：是否遵循 probe 的任务格式与约束

评分输出要求：

1. 每维 `0~5` 浮点分
2. `overall_0_5`（六维加权平均）
3. `pass`（由 `overall_0_5 >= pass_threshold_0_5` 判定）
4. `reason`（不超过 120 字）
5. `evidence_refs`（引用 turn 或 workspace 路径）

---

## 7. Prompt 设计与文件位置

新增模板文件（`auto_test/prompts/`）：

1. `framework/evaluator/probe/probe_judge_system.prompt`
2. `framework/evaluator/probe/probe_judge_user.prompt`

建议占位符：

1. `{{PROBE_ID}}`
2. `{{PROBE_TYPE}}`
3. `{{PROBE_DESCRIPTION}}`
4. `{{PROBE_TARGET}}`
5. `{{PROBE_CONTEXT}}`
6. `{{PROBE_RUBRIC}}`
7. `{{PASS_THRESHOLD_0_5}}`

返回 JSON 约束（强制）：

```json
{
  "pass": true,
  "overall_0_5": 4.1,
  "dimension_scores": {
    "accuracy": 4.0,
    "context_awareness": 4.5,
    "artifact_trail": 3.5,
    "completeness": 4.0,
    "continuity": 4.5,
    "instruction_following": 4.0
  },
  "reason": "简要结论",
  "evidence_refs": ["turn:4", "workspace:/workspace/advoo/workbook.md"]
}
```

---

## 8. 代码模块改造方案

## 8.1 `src/probe/models.py`

新增 dataclass：

1. `ProbeLLMJudgeSpec`
2. `ProbeLLMJudgeRun`
3. `ProbeLLMJudgeAggregate`

并在 `ProbeSpec` 中新增字段：

1. `llm_judge: ProbeLLMJudgeSpec | None = None`

## 8.2 `src/probe/loader.py`

改造点：

1. 接受 `judge_mode in {"deterministic","llm"}`
2. `deterministic` 探针保持 assertions 校验
3. `llm` 探针新增 `llm_judge` 结构校验
4. 增加 dataset 级别的错误定位信息（`probe_id + field`）

## 8.3 `src/probe/evaluator.py`

改造为双分支：

1. `evaluate_deterministic_probe(...)`
2. `evaluate_llm_probe(...)`

公共能力抽取：

1. `build_probe_context_slice(...)`：按 turn_window + target 构建轻量上下文
2. `aggregate_probe_scores(...)`：分 judge_mode 汇总 + 总分融合

## 8.4 新增 `src/probe/llm_judge.py`

职责：

1. 构造 prompt
2. 调用评估模型
3. 解析与校验 JSON
4. 重试、重复采样、聚合统计

备注：

1. 底层 HTTP 调用建议复用 `src/eval/dialogue_evaluator.py` 中的 wire-api 兼容逻辑
2. 如需进一步解耦，可在后续提取公共 `src/eval/llm_client.py`（Phase 2 可先不做）

---

## 9. 运行配置设计

## 9.1 环境变量（新增）

1. `AUTO_TEST_ENABLE_PROBE_LLM_JUDGE`：默认 `false`
2. `AUTO_TEST_PROBE_LLM_URL`
3. `AUTO_TEST_PROBE_LLM_MODEL`
4. `AUTO_TEST_PROBE_LLM_API_KEY`
5. `AUTO_TEST_PROBE_LLM_TIMEOUT_SEC`：默认 `45`
6. `AUTO_TEST_PROBE_LLM_REPEATS`：默认 `3`
7. `AUTO_TEST_PROBE_LLM_MAX_RETRIES`：默认 `2`
8. `AUTO_TEST_PROBE_LLM_FAIL_OPEN`：默认 `false`
9. `AUTO_TEST_PROBE_LLM_FINAL_WEIGHT`：默认 `0.35`

## 9.2 配置文件（`config.example.json` / `config.local.json`）

新增块：

```json
{
  "probe_eval": {
    "enabled": true,
    "dataset_path": "datasets/probes/clinic_memory_v1.json",
    "llm_judge": {
      "enabled": true,
      "base_url": "https://your-llm-endpoint/v1/chat/completions",
      "model": "gpt-4.1",
      "api_key": "",
      "timeout_sec": 45,
      "repeats": 3,
      "max_retries": 2,
      "fail_open": false
    },
    "score_merge": {
      "deterministic_weight": 0.65,
      "llm_weight": 0.35
    }
  }
}
```

优先级建议：

1. 环境变量 > config.local.json > config.example.json 默认值

---

## 10. 稳定性与降级策略

## 10.1 重复采样

对每个 LLM probe 执行 `N=repeats` 次，记录：

1. `overall_mean`
2. `overall_stddev`
3. 每维 `mean/stddev`
4. `pass_rate`

最终判定：

1. `final_pass = (pass_rate >= 0.5)` 且 `overall_mean >= threshold`

## 10.2 失败策略

1. `fail_open=false`：LLM 失败即该 probe 失败，并标注 `error`
2. `fail_open=true`：该 probe `skipped=true`，不计入主观分分母

## 10.3 防漂移机制

1. 固定 `temperature=0`
2. 固定 rubric 版本号（`rubric_id`）
3. 输出中记录 `prompt_version_hash`

---

## 11. 输出结构扩展

## 11.1 `run_data/probe_results.json`

在 `results[]` 每条 probe 追加：

```json
{
  "probe_id": "decision_strategy_consistency_v1",
  "judge_mode": "llm",
  "passed": true,
  "llm_judge": {
    "model": "gpt-4.1",
    "repeats": 3,
    "attempts": [
      { "overall_0_5": 4.0, "pass": true },
      { "overall_0_5": 3.8, "pass": true },
      { "overall_0_5": 4.2, "pass": true }
    ],
    "aggregate": {
      "overall_mean": 4.0,
      "overall_stddev": 0.1633,
      "pass_rate": 1.0
    }
  }
}
```

在 `summary` 追加：

1. `deterministic_score`
2. `llm_subjective_score`
3. `final_weighted_score`
4. `llm_probe_count`
5. `llm_probe_failed`
6. `llm_probe_skipped`

## 11.2 `probe_evaluation.md`

新增章节：

1. `LLM Subjective Summary`
2. `High Variance Probes`（按 stddev 降序）
3. `Top Failed LLM Probes`

## 11.3 `run_meta.md`

写入本次 probe-llm 配置快照：

1. URL（可脱敏）
2. model
3. repeats / retries
4. fail_open
5. merge weights

---

## 12. 与现有主流程集成点

位置：`src/tests/run_5turn_session_test.py`

保持当前顺序，仅在 `evaluate_probes(...)` 内部扩展：

1. 生成 `turn_results.json`
2. 导出 `workspace/_manifest.json`
3. 调用 probe evaluator（内部按 judge_mode 分流）
4. 写 `run_data/probe_results.json` 与 `probe_evaluation.md`
5. `write_meta_md(..., probe_eval=...)` 增加主观分汇总

---

## 13. 实施拆分（建议）

## Task 1：数据契约与加载器

1. 扩展 `models.py`
2. 扩展 `loader.py` 支持 `judge_mode=llm`
3. 为 `datasets/probes/clinic_memory_v1.json` 增补少量 llm 探针样例

## Task 2：LLM 评分执行器

1. 新增 `src/probe/llm_judge.py`
2. 增加 prompt 模板
3. 增加 JSON 校验与重试逻辑

## Task 3：聚合与报告

1. 改造 `evaluator.py` 汇总结构
2. 改造 `write_probe_evaluation_md(...)`
3. 更新 `run_meta.md` 的 Probe 概览字段

## Task 4：配置与文档同步

1. 更新 `config/config.example.json`
2. 更新 `config/config.local.json`
3. 更新 `README.md`、`doc/STABLE_INTERFACE_AND_RESULT_SPEC.md`

---

## 14. 验收标准（Phase 2 Gate）

必须全部满足：

1. 关闭 LLM 主观探针时，Phase 1 行为与结果不变
2. 开启后，`probe_results.json` 正确区分 deterministic 与 llm 结果
3. 至少 1 个 `continuation` + 1 个 `decision` 的 llm probe 可稳定产出评分
4. 重复 3 次同批评测时，输出包含方差统计，可识别高波动探针
5. 失败与跳过路径可控（`fail_open` 生效）

---

## 15. 评审关注点（请重点确认）

1. `final_weighted_score` 权重是否采用 `0.65/0.35`
2. LLM 通过阈值是否统一 `3.2/5`，还是按 probe 单独配置
3. `fail_open` 默认是否保持 `false`（建议保持严格）
4. 是否在 Phase 2 就引入公共 `llm_client` 抽象，还是先复用现有调用逻辑

---

## 16. 评审通过后的执行顺序

1. 先做 Task 1（数据契约）
2. 再做 Task 2（LLM 评分引擎）
3. 然后 Task 3（聚合与输出）
4. 最后 Task 4（配置与文档）

这样可以保证每一步都可单独验证，避免一次性改动过大带来的回归风险。
