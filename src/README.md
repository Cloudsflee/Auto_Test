# src 代码说明

## 1. 目录职责

```text
src/
  tests/   # 测试执行代码（发请求、跑场景、收集结果、落盘）
  eval/    # 评估代码（规则评估、LLM评估、评估报告）
  probe/   # 探针评估代码（数据集加载、确定性断言、汇总报告）
```

---

## 2. tests 目录

- `tests/run_5turn_session_test.py`
  - 作用：执行 5 轮 smoke 测试（可作为后续多场景框架基线）
  - 职责：
    1. 读取配置
    2. 创建会话
    3. 按轮执行并解析 SSE
    4. 落盘原始结果与元数据
    5. 调用 `eval` 模块完成评估

---

## 3. eval 目录

- `eval/dialogue_evaluator.py`
  - 作用：统一管理评估逻辑
  - 职责：
    1. 规则评估（稳定基线）
    2. 可选 LLM 评估（通过 `.prompt` 模板构造提示词）
    3. 输出 `evaluation.md` 与 `evaluation.json` 所需数据

---

## 4. probe 目录

- `probe/evaluator.py`
  - 作用：执行 Phase 1+2 探针评估（确定性 + 主观 LLM）
  - 职责：
    1. 读取探针数据集
    2. 从 `turn_results.json` 与 `workspace/_manifest.json` 构建上下文
    3. 根据 `judge_mode` 执行确定性断言或 LLM 裁判
    4. 输出 `probe_results.json` 与 `probe_evaluation.md`

---

## 5. 扩展建议

后续如果新增长会话测试，可继续在 `tests/` 下增加：

- `run_long_memory_test.py`
- `run_topic_shift_test.py`
- `run_compression_checkpoint_test.py`

并尽量复用 `eval/dialogue_evaluator.py`，避免评估逻辑分散。
