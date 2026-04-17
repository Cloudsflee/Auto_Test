# auto_test 后续完整计划（审阅版）

## 1. 目的与约束

本计划用于你人工审阅当前代码后，作为下一轮迭代执行清单。

约束如下：

1. 当前轮停止新增功能代码，只输出计划文档。
2. 后续改动默认仅限 `auto_test/`；如需改 `auto_test/` 外文件（除你明确要求文档），先明文申请。
3. 保持现有结果结构兼容，不破坏历史脚本可运行性。

---

## 2. 当前状态快照（截至本次）

已落地：

1. Phase A：`evaluation_v2_shadow`（影子评分）已接入。
2. Phase B：`evaluation_primary`（主判定）已接入，当前默认 `foundation_v2`。
3. Phase C：`evaluation_compare`（`llm_v1/foundation_v2/final_v2` 对比）+ `evaluation.md` A/B 区块已接入。
4. Phase D（第一版）：已接入多 profile 配置结构与 capability_mode 路由，`evaluation_v2_shadow` 已包含：
   - `profile_router`
   - `profile_combined`
   - `profiles.details`

当前待完善点：

1. 多 profile 在默认配置下尚未形成“常态化激活验证样例”（例如 `safety_heavy` 默认关闭）。
2. 编码历史遗留（部分中文输出乱码）仍在，需单独做稳定化处理。
3. `run_5turn_session_test.py` 体量偏大，仍需模块化拆分。

---

## 3. 总体目标（下一阶段）

围绕“通用底盘 + 可插拔 profile”完成可用化收口：

1. 让多 profile 路由成为“可控、可观测、可回滚”的正式机制。
2. 保持旧字段兼容，保障历史评测链路不受影响。
3. 把关键逻辑从超大脚本拆成可维护模块，降低后续迭代风险。

---

## 4. 分阶段执行计划

## Phase D-2（收口与验证）

目标：把已接入的多 profile 机制做成稳定可用版本。

任务：

1. 配置收口：
   - 明确 `active / active_profiles / active_profiles_by_capability_mode` 的优先级规则并固化文档。
   - 明确 `profiles.<name>.enabled / merge_weight` 的默认行为。
2. 输出收口：
   - 校验 `evaluation_v2_shadow.profile_combined` 与 `profiles.details` 字段稳定性。
   - 增补 `evaluation.md` 中多 profile 状态展示（已初步接入，做格式统一）。
3. 回归验证：
   - 最小 1/5/10 轮三档回归。
   - 至少 1 组 capability 路由覆盖（例如 `alternating -> [memory_compression, safety_heavy]`）。

验收标准：

1. `evaluation.json` 始终包含旧字段 + `evaluation_primary` + `evaluation_compare`。
2. 配置切换 profile 时结果可观测、无异常报错。
3. 结果目录中 workspace 导出（md/图片）保持正常。

---

## Phase E（工程化与模块化）

目标：降低 `run_5turn_session_test.py` 复杂度，保持行为不变。

拆分建议：

1. `src/tests/config_loader.py`
   - 读取与归一化 config/env
2. `src/tests/evaluation_pipeline.py`
   - 封装 `llm_eval -> primary -> compare -> write` 流程
3. `src/tests/profile_router.py`
   - 独立路由逻辑，提供单测入口
4. `src/tests/report_writer.py`
   - 统一 `run_data/README`、meta 等文本输出

验收标准：

1. 拆分后外部入口命令不变：
   - `python auto_test/src/tests/run_5turn_session_test.py --max-turns N`
2. 结果文件路径与关键字段不变。
3. 通过 1/5/10 轮回归。

---

## Phase F（评估结构强化）

目标：让评估结果结构更稳健，便于外部分析工具消费。

任务：

1. 增加 `evaluation schema` 校验（运行时轻校验即可）。
2. 对 `evaluation_compare` 增加“不可用原因”字段（当某模式不可用时）。
3. 对 `profile_combined` 增加标准化元信息：
   - `effective_profiles`
   - `disabled_profiles`
   - `merge_weights_normalized`

验收标准：

1. 下游读取不会因字段缺失崩溃。
2. 非法配置可给出清晰错误信息。
3. 兼容历史产物读取。

---

## Phase G（测试与发布前门禁）

目标：将“可运行”升级为“可持续回归”。

任务：

1. 增加最小自动化回归脚本（建议 `src/tests/regression_smoke.py`）：
   - 1 轮 smoke
   - 5 轮 workspace 导出验证
   - profile 路由覆盖验证
2. 增加结果完整性检查：
   - `evaluation.json` 必含关键键
   - `run_data` 文件齐全
3. 文档化回滚策略：
   - `primary_mode` 快速回退到 `llm_v1` 或 `foundation_v2`
   - profile 全量禁用开关

验收标准：

1. 一条命令可完成最小回归。
2. 失败时可定位到具体阶段（会话、评估、导出、探针）。

---

## 5. 执行顺序建议（优先级）

1. 先做 Phase D-2（功能收口 + 回归验证）。
2. 再做 Phase E（模块化，降低后续风险）。
3. 然后 Phase F（结构强化）。
4. 最后 Phase G（门禁与回滚体系完善）。

---

## 6. 每阶段产物清单

1. 代码变更（仅 `auto_test/`）。
2. 文档同步（`auto_test/README.md`、`auto_test/config/README.md`、`auto_test/doc/EVAL_INTEGRATION.md`）。
3. 验证记录（run_id + 关键日志 + 结果字段截图/节选）。

---

## 7. 风险与对应策略

1. 风险：多 profile 逻辑引入回归。
   - 策略：默认保守配置（单 profile），多 profile 走显式开关。
2. 风险：字段增加导致下游脚本解析失败。
   - 策略：旧字段不删不改；新增字段只增不替。
3. 风险：编码问题导致文本报告异常。
   - 策略：统一 UTF-8 写入并加最小校验。

---

## 8. 下一轮开始前确认项

请你确认以下再开工：

1. 是否将 `safety_heavy` 在 `config.local.json` 默认启用用于多 profile 常态测试。
2. 是否优先推进模块化（Phase E）还是先做评估结构强化（Phase F）。
3. 是否需要把 10 轮回归升级为 25 轮作为每次合并前基线。

