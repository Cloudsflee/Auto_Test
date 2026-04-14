# probes 数据集说明

本目录存放 Phase 1 探针数据集（JSON）。

- `clinic_memory_v1.json`: 首版通用探针集（确定性评分）。

使用方式：

1. 开启探针评估：
   - `AUTO_TEST_ENABLE_PROBE_EVAL=true`
2. 可选指定数据集路径：
   - `AUTO_TEST_PROBE_DATASET_PATH=datasets/probes/clinic_memory_v1.json`

输出文件：

- `results/<run>/run_data/probe_results.json`
- `results/<run>/probe_evaluation.md`
