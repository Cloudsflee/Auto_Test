# probes 数据集说明

本目录存放 `auto_test` 使用的探针数据集（JSON）。

## 现有数据集

- `clinic_memory_v1.json`
  - 第一版通用探针集（确定性规则优先）。
- `clinic_memory_v2_mixed.json`
  - 第二版混合探针集（deterministic + llm）。
- `compression_retention_contract_v1.json`
  - 面向“压缩评估”的分层保留契约数据集（must/should/may）。
  - 关键字段：
    - `anchor.points`：分层锚点（`must_keep/should_keep/may_drop`）。
    - `plant_text_template`：第 2 轮埋点模板。
    - `probe_text_templates`：优先使用的追问模板（若为空才走 LLM 生成）。
    - `judge_rubric`：失效判定口径约束。
    - `failure_policy`：聚合层 gate 参考阈值。

## 使用方式

1. 开启探针评估
   - `AUTO_TEST_ENABLE_PROBE_EVAL=true`
2. 指定数据集路径
   - `AUTO_TEST_PROBE_DATASET_PATH=datasets/probes/clinic_memory_v1.json`
3. 若数据集包含 `judge_mode=llm`，可开启主观裁判
   - `AUTO_TEST_ENABLE_PROBE_LLM_JUDGE=true`
   - 并配置 `AUTO_TEST_PROBE_LLM_URL/MODEL/API_KEY`

## 输出文件

- `results/<run>/run_data/probe_results.json`
- `results/<run>/probe_evaluation.md`
