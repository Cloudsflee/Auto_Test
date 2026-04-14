# Phase 1（先做）：探针体系 MVP 详细设计（确定性优先）

## 1. 文档定位

本文档是 [递进开发方案.md](/d:/04_projects/advoo_prod/ai-backend/auto_test/递进开发方案.md) 中 Phase 1 的工程化细化稿，仅覆盖：

- 探针数据结构与执行引擎
- 确定性评分器（字符串/正则/断言）
- 与当前 `run_5turn_session_test.py` 的最小侵入集成

不包含：

- LLM 探针评分（放到 Phase 2）
- pass@k / pass^k（放到 Phase 3）
- Elo 对战（放到 Phase 5）

---

## 2. 设计目标

## 2.1 业务目标

1. 把“记忆有没有被有效保留”变成可批量执行、可复核的结构化结果。
2. 覆盖 `Recall / Artifact / Continuation / Decision` 四类探针的 MVP。
3. 优先使用确定性评分，保证低噪声、低成本、可解释。

## 2.2 工程目标

1. 对现有运行链路最小侵入，不破坏当前可跑能力。
2. 新能力以“后处理插件”形式接入，复用已有结果文件：
   - `run_data/turn_results.json`
   - `workspace/_manifest.json`
   - `raw_events.jsonl`（可选）
3. 输出结构可直接支持后续 Phase 2/3 扩展。

---

## 3. 现状约束（基于当前项目）

当前已稳定产物（可直接作为探针输入）：

1. `run_data/turn_results.json`：包含每轮用户输入、助手输出、run_end/run_error、workspace_snapshot 等。
2. `workspace/_manifest.json`：包含导出的 md/image 文件、来源、未解析项。
3. `raw_events.jsonl`：原始事件轨迹，可在复杂断言时补充证据。

结论：

- Phase 1 无需改动后端协议，可完全在 `auto_test` 本地完成探针判定。

---

## 4. 目录与模块设计

建议新增目录：

```text
auto_test/
  datasets/
    probes/
      README.md
      clinic_memory_v1.json
      content_workflow_v1.json
  src/
    probe/
      __init__.py
      models.py
      loader.py
      context_builder.py
      deterministic_judges.py
      executor.py
      reporter.py
```

职责划分：

1. `models.py`：探针/断言/执行结果数据结构定义。
2. `loader.py`：加载并校验探针数据集。
3. `context_builder.py`：把现有结果文件构造成统一查询上下文。
4. `deterministic_judges.py`：断言执行器（仅确定性）。
5. `executor.py`：按探针顺序执行并聚合得分。
6. `reporter.py`：输出 `probe_results.json` / `probe_results.md`。

---

## 5. 数据契约设计（核心）

## 5.1 Probe Dataset 顶层结构

```json
{
  "dataset_id": "clinic_memory_v1",
  "dataset_version": "1.0.0",
  "description": "口腔门店多轮营销场景探针集",
  "owner": "auto_test",
  "created_at": "2026-04-14",
  "probes": []
}
```

字段约束：

1. `dataset_id`：唯一标识，建议语义化命名。
2. `dataset_version`：语义化版本（后续用于回归对比）。
3. `probes`：至少 1 条。

## 5.2 Probe 定义结构

```json
{
  "probe_id": "recall_clear_memory_instruction",
  "probe_type": "recall",
  "priority": "high",
  "weight": 1.0,
  "judge_mode": "deterministic",
  "target": {
    "source": "turn_assistant_text",
    "turn": 1
  },
  "assertions": [
    {
      "assert_type": "contains_any",
      "expect": ["重置", "初始模板"]
    }
  ],
  "evidence_policy": {
    "capture_chars": 200
  },
  "tags": ["memory", "first_turn", "must_pass"],
  "critical": true,
  "description": "首轮必须明确响应记忆重置动作"
}
```

字段约束：

1. `probe_type`：`recall/artifact/continuation/decision`
2. `judge_mode`：Phase 1 仅允许 `deterministic`
3. `weight`：`> 0`，默认 `1.0`
4. `critical`：是否作为硬门槛（失败直接标红）

## 5.3 Assertion 结构

```json
{
  "assert_type": "regex_match",
  "expect": "已(生成|保存).+workspace",
  "flags": "i",
  "negate": false
}
```

扩展字段：

- `actual_from`：从 target 结果提取子字段（例如 json path）
- `operator`：数值比较（`eq/gte/lte/between`）
- `tolerance`：浮点容差（预留）

---

## 6. 执行上下文模型

探针执行时统一上下文 `ProbeContext`：

1. `turn_results`：解析自 `turn_results.json`
2. `workspace_manifest`：解析自 `_manifest.json`
3. `assistant_by_turn`：`{turn -> assistant_text}`
4. `user_by_turn`：`{turn -> user_text}`
5. `workspace_paths`：合并 `all_paths`
6. `workspace_exported_text_files` / `workspace_exported_image_files`
7. `run_meta`（可选，从 `run_meta.md` 摘要）

目标：让 assertion 实现不依赖原始文件细节，统一通过 `context` 查询。

---

## 7. 确定性评分器设计

## 7.1 Phase 1 必做断言类型

1. `contains_any`：文本包含任一关键字
2. `contains_all`：文本必须同时包含全部关键字
3. `not_contains_any`：文本不得包含敏感片段
4. `regex_match`：正则匹配
5. `equals`：严格等值
6. `file_exists`：workspace 指定路径存在
7. `file_ext_count_gte`：某后缀文件数量下限（如 `>=1` 张图）
8. `json_field_equals`：JSON 指定字段值断言
9. `json_field_in`：JSON 字段属于候选集合
10. `bool_field_true`：布尔字段断言（如 `run_end=true`）
11. `numeric_field_gte`：数值字段下限
12. `list_contains_path_pattern`：路径列表匹配模式

## 7.2 目标源（target.source）定义

1. `turn_assistant_text`
2. `turn_user_text`
3. `turn_object`
4. `workspace_manifest`
5. `workspace_paths`
6. `global_summary`（执行器计算的聚合对象）

## 7.3 评分规则

探针通过规则（Phase 1）：

1. 单个 assertion：`true/false`
2. 单个 probe：`assertions` 全部通过才算 `passed=true`
3. 总分：
   - `weighted_score = sum(weight * pass) / sum(weight)`
4. 关键失败：
   - `critical=true` 且失败的 probe 进入 `critical_failed`

---

## 8. 执行流程设计

执行顺序：

1. 加载数据集并做 schema 校验
2. 构建 `ProbeContext`
3. 按 probe 顺序执行
4. 生成 probe 级结果（含证据）
5. 聚合统计并输出报告

伪代码：

```python
dataset = load_probe_dataset(path)
ctx = build_probe_context(turn_results_path, workspace_manifest_path, raw_events_path)
probe_results = []
for probe in dataset.probes:
    result = execute_probe(probe, ctx)
    probe_results.append(result)
summary = aggregate(probe_results)
write_probe_results_json(summary, probe_results)
write_probe_results_md(summary, probe_results)
```

---

## 9. 输出契约设计

新增输出文件：

1. `results/<run>/run_data/probe_results.json`
2. `results/<run>/probe_evaluation.md`

`probe_results.json` 建议结构：

```json
{
  "generated_at": "2026-04-14T10:00:00",
  "dataset_id": "clinic_memory_v1",
  "dataset_version": "1.0.0",
  "summary": {
    "total_probes": 24,
    "passed_probes": 19,
    "failed_probes": 5,
    "weighted_score": 0.83,
    "critical_failed": ["artifact_workspace_image_exists"],
    "by_type": {
      "recall": {"total": 8, "passed": 7, "score": 0.88},
      "artifact": {"total": 6, "passed": 4, "score": 0.67},
      "continuation": {"total": 6, "passed": 5, "score": 0.83},
      "decision": {"total": 4, "passed": 3, "score": 0.75}
    }
  },
  "results": []
}
```

单条 `results[]` 建议结构：

```json
{
  "probe_id": "artifact_workspace_image_exists",
  "probe_type": "artifact",
  "passed": false,
  "critical": true,
  "weight": 1.0,
  "assertions": [
    {
      "assert_type": "file_ext_count_gte",
      "passed": false,
      "expected": {"ext": ["jpg", "png"], "gte": 1},
      "actual": {"count": 0}
    }
  ],
  "evidence": {
    "turn": 4,
    "snippet": "海报已生成...",
    "paths_preview": ["/workspace/advoo/xxx.md"]
  },
  "failure_reason": "expected >=1 image file in workspace, got 0"
}
```

---

## 10. 与现有主流程集成方案

## 10.1 集成位置（最小侵入）

在 `run_5turn_session_test.py` 的现有流程中，建议插入点：

1. `turn_results.json` 写完后
2. `workspace_export` 完成后
3. `llm_eval` 前或后都可（建议前，便于把 probe 摘要写入 `evaluation.md`）

## 10.2 配置开关（新增环境变量）

1. `AUTO_TEST_ENABLE_PROBE_EVAL`：默认 `false`
2. `AUTO_TEST_PROBE_DATASET_PATH`：默认 `auto_test/datasets/probes/clinic_memory_v1.json`
3. `AUTO_TEST_PROBE_FAIL_ON_DATASET_ERROR`：默认 `true`
4. `AUTO_TEST_PROBE_MAX_FAIL_DETAILS`：默认 `20`（报告中最多展示失败项）

说明：

- 默认关闭是为了保证现有流程完全不受影响；评审通过后可改默认开启。

## 10.3 与已有评估文件协同

`evaluation.md` 增加一个 `Probe Evaluation` 区块，展示：

1. 通过率
2. weighted score
3. critical_failed
4. Top-N 失败探针

---

## 11. MVP 初始探针集设计建议（至少 20 条）

建议分布：

1. Recall：8 条
2. Artifact：6 条
3. Continuation：4 条（Phase 1 先做可确定性的子集）
4. Decision：2 条（先做关键词级，主观评分留到 Phase 2）

示例（建议首批）：

1. 首轮记忆重置响应包含关键语义
2. 首轮 run_end=true 且 run_error 为空
3. 图片任务轮次中，assistant 回复应提及“已生成/已保存”
4. `_manifest.counts.exported_image_files >= 1`
5. `_manifest.unresolved_files == 0`
6. 文案任务轮次出现 md 文件导出
7. 不出现工具载荷泄漏关键字（`<tool>`, `"tool_type"`）
8. 后续轮次可引用前轮约束关键词（简单 continuation）

---

## 12. 错误处理与降级策略

1. 数据集加载失败：
   - 若 `AUTO_TEST_PROBE_FAIL_ON_DATASET_ERROR=true`：run 失败
   - 否则：写入 `probe_results.json`，标记 `skipped=true`
2. 单探针执行异常：
   - 不中断整体执行，记录 `error` 并判该探针失败
3. 证据截断：
   - 长文本按 `capture_chars` 截断，避免结果文件过大

---

## 13. 测试策略（Phase 1 必做）

## 13.1 单元测试

1. `deterministic_judges.py` 每种断言至少 2 个正反样例
2. loader schema 校验测试
3. context_builder 对缺失文件/空字段容错测试

## 13.2 集成测试

1. 使用固定历史 `results` 目录跑 probe 引擎，输出与金标比对
2. 设计 1 组“故意失败”的样例，验证 critical_failed 生效

---

## 14. 里程碑与任务拆分（可直接执行）

## Task A（数据与模型）

1. 新建 `src/probe/models.py`
2. 定义 dataset/probe/assertion/result dataclass
3. 新建 `datasets/probes/clinic_memory_v1.json`（首版 20+ 探针）

## Task B（执行引擎）

1. `loader.py` + schema 校验
2. `context_builder.py`
3. `deterministic_judges.py`
4. `executor.py`

## Task C（集成与输出）

1. 在 `run_5turn_session_test.py` 接入执行入口
2. 新增 `probe_results.json` 输出
3. 更新 `evaluation.md` 渲染
4. 更新 `README` 与 `doc/STABLE_INTERFACE_AND_RESULT_SPEC.md`

---

## 15. 验收标准（Phase 1 Gate）

必须全部满足：

1. 在不启用开关时，现有测试行为完全不变。
2. 启用开关后，能稳定输出 `probe_results.json` 与 `probe_evaluation.md`。
3. 首版探针集 >= 20 条，确定性探针占比 >= 70%。
4. 报告包含 probe 级失败原因与证据片段。
5. 在至少 3 个历史 run 上复跑无崩溃。

---

## 16. 评审关注点（请重点拍板）

1. Probe schema 是否足够稳定（是否要立刻上 JSON Schema 文件）
2. `critical` 门槛策略是否合理（是否允许按 probe_type 分门槛）
3. 默认开关是否保持关闭（建议先关闭，待评审通过后开启）
4. 首版 20+ 探针清单是否需要你先指定优先业务场景

---

## 17. 评审通过后的执行顺序（建议）

1. 先落 `Task A`（数据与模型）
2. 再落 `Task B`（执行引擎）
3. 最后做 `Task C`（主流程接入与文档更新）

这样可确保每一步都可独立验证，避免大改后难回滚。
