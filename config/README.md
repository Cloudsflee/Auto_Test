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
- `user_simulator.base_url`: 用户模拟模型接口地址（支持 `.../chat/completions` 或 `.../responses`）
- `user_simulator.model`: 用户模拟模型名
- `user_simulator.api_key`: 用户模拟模型 API Key
- `user_simulator.timeout_sec`: 用户模拟请求超时（秒）
- `user_simulator.max_turns`: 计划对话轮次上限
- `user_simulator.first_user_message`: 第一轮用户消息（会自动强制前缀“请清空当前notebook内的所有内容，包括files和workbook”）
- `user_simulator.system_prompt_file`: system 提示词文件名（相对 `prompts/`，默认 `user_simulator_system.prompt`）
- `user_simulator.scenario_prompt_file`: 场景提示词文件名（相对 `prompts/`，默认 `user_simulator_scenario.prompt`）
- `user_simulator.system_prompt`: 可选内联 system 提示词（存在时覆盖文件）
- `user_simulator.scenario_prompt`: 可选内联场景提示词（存在时覆盖文件）

说明：

- LLM 评估优先读取 `llm_eval` 配置；未配置字段会回退到环境变量。
- 兼容环境变量：`AUTO_TEST_ENABLE_LLM_EVAL`、`AUTO_TEST_EVAL_LLM_URL`、`AUTO_TEST_EVAL_LLM_MODEL`、`AUTO_TEST_EVAL_LLM_API_KEY`、`AUTO_TEST_EVAL_LLM_TIMEOUT_SEC`。
- 用户模拟优先读取 `user_simulator` 配置；未配置字段会回退到 `AUTO_TEST_USER_SIM_*` 环境变量，最后回退到 `llm_eval` 同名字段。
- 用户模拟默认从 `prompts/user_simulator_system.prompt` 与 `prompts/user_simulator_scenario.prompt` 读取提示词。
- `user_simulator` 与 `llm_eval` 职责不同：前者在测试执行时生成“下一轮用户输入”，后者在测试结束后做裁判评估。

---

## 备注

- 仓库默认不包含 `test.txt` / `test_re.txt`，仅用于兼容历史本地配置文件。
- 建议统一迁移到 `config.local.json` 或 `config.json`。
