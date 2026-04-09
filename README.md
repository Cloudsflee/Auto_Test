# auto_test（可迭代说明）

## 1. 目录目标

`auto_test` 用于记忆与压缩专项自动测试。  
文档分为两类：

- 稳定文档：放在 `auto_test/doc/`，用于长期复用
- 临时/迭代文档：放在 `auto_test/` 根目录，随代码快速更新

---

## 2. 目录结构（当前）

```text
auto_test/
  doc/                           # 稳定契约与规范
  src/                           # 测试与评估代码
    tests/
      run_5turn_session_test.py  # 5轮 smoke 测试执行入口
    eval/
      dialogue_evaluator.py      # 规则/LLM 评估实现
    README.md
  prompts/                       # 评估提示词模板（.prompt）
  config/                        # 推荐配置位置（json）
  results/                       # 每次运行输出（默认被 .gitignore 忽略）
  README.md
  实验目的re.md
```

---

## 3. 文档索引

## 3.1 稳定文档（doc）

- `doc/STABLE_INTERFACE_AND_RESULT_SPEC.md`  
  内容：接口契约、SSE 契约、results 输出结构、评估输出结构（长期基线）
- `doc/EVAL_INTEGRATION.md`  
  内容：规则评估与可选 LLM 评估的集成约定

## 3.2 可变文档（随迭代更新）

- 本文件 `README.md`：目录职责、运行入口、日常维护说明
- `src/README.md`：`src/tests` 与 `src/eval` 职责说明
- `config/README.md`：配置格式与加载优先级
- `prompts/README.md`：评估提示词模板说明

---

## 4. 代码组织约定

- `src/tests/`：负责“发请求、跑场景、收集原始结果、落盘”
- `src/eval/`：负责“规则评估、可选 LLM 评估、评估报告输出”
- `prompts/`：评估提示词模板，不写死在代码里

建议后续新增场景时保持：

1. 测试逻辑只放 `src/tests/`
2. 评估逻辑只放 `src/eval/`
3. 评估提示词只放 `prompts/`

---

## 5. 运行入口与配置

- 当前测试入口：`src/tests/run_5turn_session_test.py`
- 推荐运行命令：

```bash
python src/tests/run_5turn_session_test.py
```

- 配置加载优先级（高 -> 低）：
  1. `config/config.local.json`
  2. `config/config.json`
  3. `test.txt`（旧格式兼容，可选）
  4. `test_re.txt`（旧格式兼容，可选）

说明：

- 仓库默认提供 `config/config.example.json` 作为模板。
- `test.txt` / `test_re.txt` 为兼容历史格式，仓库内默认不提供示例文件。
- 评估 LLM 的接口信息建议写入 `config/*.json` 的 `llm_eval` 字段（`enabled/base_url/model/api_key/timeout_sec`）。

---

## 6. 结果产物

每次运行会创建独立目录：`results/session_5turn_<run_id>/`，包含：

- `run_meta.md`
- `dialogue.md`
- `evaluation.md`
- `raw_events.jsonl`
- `non_text/turn_results.json`
- `non_text/evaluation.json`

---

## 7. 维护建议

1. 若后端协议变化，先更新 `doc/STABLE_INTERFACE_AND_RESULT_SPEC.md`
2. 若仅测试流程变化，优先更新 `README.md` 与 `src/` 代码
3. 每次运行必须落独立目录，不覆盖历史 `results`
