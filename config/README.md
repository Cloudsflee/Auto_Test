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

说明：

- LLM 评估优先读取 `llm_eval` 配置；未配置字段会回退到环境变量。
- 兼容环境变量：`AUTO_TEST_ENABLE_LLM_EVAL`、`AUTO_TEST_EVAL_LLM_URL`、`AUTO_TEST_EVAL_LLM_MODEL`、`AUTO_TEST_EVAL_LLM_API_KEY`、`AUTO_TEST_EVAL_LLM_TIMEOUT_SEC`。

---

## 备注

- 仓库默认不包含 `test.txt` / `test_re.txt`，仅用于兼容历史本地配置文件。
- 建议统一迁移到 `config.local.json` 或 `config.json`。
