# tests 目录说明

- 本目录存放可执行的 auto-test 测试流程。
- 主要职责：组装请求、执行多轮会话、解析 SSE 事件并写出结果。
- 统一约定：首轮固定为 `请你把当前记忆文件重置为系统初始模板`。
- `user_simulator` 支持动态角色扮演用户输入（默认模式 `provider_context`）。
- 默认工具集保留记忆/压缩相关工具和 `option_card`。
- `post_submit` 与 `ui_cmd` 在当前测试范围内默认关闭。
- 交互回调默认启用“模拟回传”。
- 可通过 `AUTO_TEST_SIMULATE_INTERACT_CALLBACK=false` 关闭模拟回传。
- 安全保护：`AUTO_TEST_MAX_SIM_CALLBACK_ROUNDS`（默认 `8`）。
- 轮次可由 `user_simulator.max_turns`、`AUTO_TEST_MAX_TURNS` 或命令行 `--max-turns` 控制。
- 能力考核模式可由 `user_simulator.capability_mode`（或环境变量 `AUTO_TEST_USER_SIM_CAPABILITY_MODE`）配置：
  - `alternating` / `mixed` / `single_random` / `copy_only` / `image_only`
- 工作区图片/二进制导出采用链式回退：
  - 先走 `files/session` API
  - 失败后走 DotAI FS（`/dotai/fs/stat` + `/dotai/fs/download`）
- DotAI 地址可通过 `config.dotai_base_url` 或环境变量 `AUTO_TEST_DOTAI_BASE_URL` 指定。
- 探针评估（Phase 1，确定性）可通过环境变量开启：
  - `AUTO_TEST_ENABLE_PROBE_EVAL=true`
  - 可选：`AUTO_TEST_PROBE_DATASET_PATH=datasets/probes/clinic_memory_v1.json`
  - 输出：`run_data/probe_results.json` 与 `probe_evaluation.md`

当前主入口文件：
- `run_5turn_session_test.py`：多轮会话测试基线入口。

模块拆分（提升可维护性）：
- `user_simulator_engine.py`：能力模式策略 + 角色生成 + 用户发言生成
- `workspace_pipeline.py`：workspace 快照识别 + workspace 导出 + files API / DotAI FS 下载回退

- 如需启用高级交互工具：
  - `AUTO_TEST_ENABLE_INTERACT_TOOLS=true`（同时开启 `post_submit` 与 `ui_cmd`）
  - 或细粒度开关：`AUTO_TEST_ENABLE_POST_SUBMIT_TOOL=true`、`AUTO_TEST_ENABLE_UI_CMD_TOOL=true`
- 可通过 `AUTO_TEST_DISABLE_INTERACT_TOOLS=true` 关闭全部交互工具注册。
