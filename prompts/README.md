# prompts 目录说明

本目录存放评估提示词模板（`.prompt`）。

- `llm_eval_system.prompt`：评估模型 system prompt
- `llm_eval_user.prompt`：评估模型 user prompt（含占位符）
- `user_simulator_system.prompt`：用户模拟器 system prompt
- `user_simulator_scenario.prompt`：用户模拟器场景 prompt（强调随机对话）

占位符约定：

- `{{EXPECTED_FACTS}}`
- `{{CONVERSATION}}`
- `{{MAX_TURNS}}`
