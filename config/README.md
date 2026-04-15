# 配置说明

## 推荐方式

在本目录新增 `config.local.json`（不要提交真实凭证），结构参考 `config.example.json`。

加载优先级（高 -> 低）：

1. `config/config.local.json`
2. `config/config.json`
3. `test.txt`（旧格式兼容，可选）
4. `test_re.txt`（旧格式兼容，可选）

---

## 字段说明

- `base_url`: 后端基地址（不带末尾 `/` 更稳妥）
- `proxy.http`: 可选 HTTP 代理地址（如 `http://127.0.0.1:7890`）
- `proxy.https`: 可选 HTTPS 代理地址（如 `http://127.0.0.1:7890`）
- `proxy.no_proxy`: 可选 NO_PROXY（逗号分隔）
- `auth.token`: `Bearer xxx` 或 raw token（代码会自动补 Bearer）
- `auth.uid`: 用户 ID
- `auth.email`: 用户邮箱
- `llm_eval.enabled`: 是否启用 LLM 评估（`true/false`）
- `llm_eval.base_url`: 评估模型接口地址（支持 `.../chat/completions` 或 `.../responses`）
- `llm_eval.model`: 评估模型名（如 `gpt-5.3-codex`）
- `llm_eval.api_key`: 评估模型 API Key
- `llm_eval.timeout_sec`: 评估请求超时（秒）
- `user_simulator.enabled`: 是否启用“模型扮演客户”模式
- `user_simulator.mode`: 上下文模式，默认 `provider_context`（可选：`provider_context` / `explicit_context`）
- `user_simulator.target_name`: 目标配置名（默认 `advoo`，对应 `prompts/targets/<target_name>/`）
- `user_simulator.base_url`: 用户模拟模型接口地址（支持 `.../chat/completions` 或 `.../responses`）
- `user_simulator.model`: 用户模拟模型名
- `user_simulator.api_key`: 用户模拟模型 API Key
- `user_simulator.timeout_sec`: 用户模拟请求超时（秒）
- `user_simulator.max_turns`: 计划对话轮次上限
- `user_simulator.capability_mode`: 能力考核模式（`alternating` / `mixed` / `single_random` / `copy_only` / `image_only`）
- `user_simulator.user_temperature`: 用户发言生成温度（建议 `0.7~1.2`）
- `user_simulator.role_temperature`: 角色生成温度（建议 `0.8~1.3`）
- `user_simulator.first_user_message`: first-turn user message. The first sentence is forcibly fixed to "请你把当前记忆文件重置为系统初始模板"; extra text is appended on the next line.
- `user_simulator.system_prompt_file`: system 提示词文件名（相对 `prompts/`，默认 `framework/simulator/system/user_simulator_system.prompt`）
- `user_simulator.scenario_prompt_file`: 可选场景模板文件（未配置时自动由 `targets/<target_name>/scenarios/*.yaml` 组装）
- `user_simulator.role_system_prompt_file`: 角色生成 system 提示词文件名（默认 `framework/simulator/role_generation/role_system.prompt`）
- `user_simulator.role_user_prompt_file`: 角色生成 user 提示词文件名（默认 `framework/simulator/role_generation/role_user.prompt`）
- `user_simulator.system_prompt`: 可选内联 system 提示词（存在时覆盖文件）
- `user_simulator.scenario_prompt`: 可选内联场景提示词（存在时覆盖文件）
- `user_simulator.role_system_prompt`: 可选内联角色生成 system 提示词（存在时覆盖文件）
- `user_simulator.role_user_prompt`: 可选内联角色生成 user 提示词（存在时覆盖文件）
- `probe_eval.enabled`: 是否启用探针评估
- `probe_eval.dataset_path`: 探针数据集路径（相对 `auto_test/`）
- `probe_eval.fail_on_error`: 数据集/执行错误时是否直接失败
- `probe_eval.max_fail_details`: Markdown 报告中展示失败条目上限
- `probe_eval.llm_judge.enabled`: 是否启用主观探针 LLM 裁判
- `probe_eval.llm_judge.base_url`: 探针裁判模型接口地址
- `probe_eval.llm_judge.model`: 探针裁判模型名
- `probe_eval.llm_judge.api_key`: 探针裁判 API Key
- `probe_eval.llm_judge.timeout_sec`: 单次裁判超时（秒）
- `probe_eval.llm_judge.repeats`: 每个 LLM 探针重复采样次数
- `probe_eval.llm_judge.max_retries`: 单次采样失败后的重试次数
- `probe_eval.llm_judge.fail_open`: 裁判失败时是否跳过而非判失败
- `probe_eval.llm_judge.system_prompt_file`: 裁判 system prompt 文件名（相对 `prompts/`，默认 `framework/evaluator/probe/probe_judge_system.prompt`）
- `probe_eval.llm_judge.user_prompt_file`: 裁判 user prompt 文件名（相对 `prompts/`，默认 `framework/evaluator/probe/probe_judge_user.prompt`）
- `probe_eval.score_merge.deterministic_weight`: 确定性分权重
- `probe_eval.score_merge.llm_weight`: 主观 LLM 分权重

说明：

- 当前测试已切换为 LLM-only：必须配置并启用 `llm_eval` 与 `user_simulator`。
- 代理可直接在配置中声明（`proxy.http/proxy.https/proxy.no_proxy`）；代码会在运行时自动写入环境变量。
- 环境变量若已存在（如 `HTTP_PROXY`），会优先于配置中的代理值。
- LLM 评估优先读取 `llm_eval` 配置；未配置字段会回退到环境变量。
- 兼容环境变量：`AUTO_TEST_ENABLE_LLM_EVAL`、`AUTO_TEST_EVAL_LLM_URL`、`AUTO_TEST_EVAL_LLM_MODEL`、`AUTO_TEST_EVAL_LLM_API_KEY`、`AUTO_TEST_EVAL_LLM_TIMEOUT_SEC`。
- 用户模拟优先读取 `user_simulator` 配置；未配置字段会回退到 `AUTO_TEST_USER_SIM_*` 环境变量，最后回退到 `llm_eval` 同名字段。
- 可用环境变量 `AUTO_TEST_USER_SIM_CAPABILITY_MODE` 覆盖能力考核模式。
- 用户模拟默认从 `prompts/framework/simulator/...` 读取通用模板，并注入 `prompts/targets/<target_name>/...` 的目标上下文。
- 每次测试开始前会先调用“角色生成器”随机产出一个客户角色，再进入正式多轮对话。
- `user_simulator` 与 `llm_eval` 职责不同：前者在测试执行时生成“下一轮用户输入”，后者在测试结束后做 LLM 评估。
- 轮次可直接调节：优先 `user_simulator.max_turns`，也可通过 `AUTO_TEST_MAX_TURNS` 或命令行 `--max-turns` 临时覆盖。
- 图片评测场景可通过 `AUTO_TEST_TURN_TIMEOUT_SEC` 调整单轮超时（默认 `240` 秒）。

---

## 备注

- 仓库默认不包含 `test.txt` / `test_re.txt`，仅用于兼容历史本地配置文件。
- 建议统一迁移到 `config.local.json` 或 `config.json`。

---


## Tool Registration Defaults (auto_test)
- By default, `runSettings.tools` enables memory/compression focused tools:
  - `read_text`, `ls`, `edit`, `write`, `time_now`
  - `image_chat`, `image_gen_edit`
  - `web_search`, `web_crawler`
  - `option_card`
- By default, these advanced interact tools are OFF:
  - `post_submit`, `ui_cmd`
- To enable them:
  - `AUTO_TEST_ENABLE_INTERACT_TOOLS=true` (enable both)
  - or fine-grained flags: `AUTO_TEST_ENABLE_POST_SUBMIT_TOOL=true`, `AUTO_TEST_ENABLE_UI_CMD_TOOL=true`
- Interrupt callback simulation is enabled by default:
  - `AUTO_TEST_SIMULATE_INTERACT_CALLBACK=true`
  - Max rounds safeguard: `AUTO_TEST_MAX_SIM_CALLBACK_ROUNDS` (default `8`)
- Optional toggles:
  - Disable interact tools: `AUTO_TEST_DISABLE_INTERACT_TOOLS=true`
  - Disable simulation: `AUTO_TEST_SIMULATE_INTERACT_CALLBACK=false`
- You can still override everything with `AUTO_TEST_RUN_SETTINGS_JSON`.

---

## Workspace Binary Export (DotAI FS fallback)

- Optional config key: `dotai_base_url`
  - Example: `http://dotai-backend.prod.api.dotai.internal`
- Optional env override: `AUTO_TEST_DOTAI_BASE_URL`
  - Higher priority than `config.dotai_base_url`.
- If neither is provided, auto-test will try to derive from `base_url`:
  - replace host prefix `ai-backend` -> `dotai-backend`.
- Workspace file export order:
  1. `files/session` API
  2. DotAI FS `POST /dotai/fs/stat` + `POST /dotai/fs/download`

---

## Probe Evaluation (Phase 1 + Phase 2)

- Enable probe evaluation:
  - `AUTO_TEST_ENABLE_PROBE_EVAL=true`
- Optional dataset path override:
  - `AUTO_TEST_PROBE_DATASET_PATH=datasets/probes/clinic_memory_v1.json`
- Error strategy:
  - `AUTO_TEST_PROBE_FAIL_ON_DATASET_ERROR=true|false` (default `true`)
- Markdown failure detail cap:
  - `AUTO_TEST_PROBE_MAX_FAIL_DETAILS` (default `20`)
- Enable subjective LLM judge (for `judge_mode=llm` probes):
  - `AUTO_TEST_ENABLE_PROBE_LLM_JUDGE=true`
- LLM judge endpoint/model/key:
  - `AUTO_TEST_PROBE_LLM_URL`
  - `AUTO_TEST_PROBE_LLM_MODEL`
  - `AUTO_TEST_PROBE_LLM_API_KEY`
- LLM judge sampling controls:
  - `AUTO_TEST_PROBE_LLM_TIMEOUT_SEC` (default `45`)
  - `AUTO_TEST_PROBE_LLM_REPEATS` (default `3`)
  - `AUTO_TEST_PROBE_LLM_MAX_RETRIES` (default `2`)
  - `AUTO_TEST_PROBE_LLM_FAIL_OPEN=true|false` (default `false`)
- Optional LLM judge prompt overrides:
  - `AUTO_TEST_PROBE_LLM_SYSTEM_PROMPT_FILE`
  - `AUTO_TEST_PROBE_LLM_USER_PROMPT_FILE`
- Score merge controls:
  - `AUTO_TEST_PROBE_DETERMINISTIC_WEIGHT` (default `0.65`)
  - `AUTO_TEST_PROBE_LLM_FINAL_WEIGHT` (default `0.35`)
- Outputs when enabled:
  - `results/<run>/run_data/probe_results.json`
  - `results/<run>/probe_evaluation.md`
- 说明：
  - 若数据集中存在 `judge_mode=llm`，但未启用 LLM 裁判，则该类探针会记为 `skipped`。
