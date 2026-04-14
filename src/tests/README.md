# tests Directory Notes

- This directory contains executable auto-test flows.
- Main responsibilities: build requests, run multi-turn sessions, parse SSE events, and write outputs.
- Unified convention: first turn is fixed to `请你把当前记忆文件重置为系统初始模板`.
- `user_simulator` supports dynamic role-play user turns (default mode: `provider_context`).
- Default run tools keep memory/compression related tools and `option_card`.
- `post_submit` and `ui_cmd` are disabled by default for this test scope.
- Interrupt callbacks are simulated by default via tool-response replay.
- You can disable callback simulation with `AUTO_TEST_SIMULATE_INTERACT_CALLBACK=false`.
- Safety guard: `AUTO_TEST_MAX_SIM_CALLBACK_ROUNDS` (default `8`).
- Turn count can be controlled by `user_simulator.max_turns`, `AUTO_TEST_MAX_TURNS`, or CLI `--max-turns`.
- Capability test mode can be configured by `user_simulator.capability_mode` (or env `AUTO_TEST_USER_SIM_CAPABILITY_MODE`):
  - `alternating` / `mixed` / `single_random` / `copy_only` / `image_only`
- Workspace image/binary export uses chained download fallback:
  - `files/session` API first
  - DotAI FS fallback (`/dotai/fs/stat` + `/dotai/fs/download`)
- DotAI endpoint can be set by `config.dotai_base_url` or env `AUTO_TEST_DOTAI_BASE_URL`.

Current entry file:
- `run_5turn_session_test.py`: multi-turn session test baseline.

- If needed, you can enable advanced interact tools:
  - `AUTO_TEST_ENABLE_INTERACT_TOOLS=true` (enables both `post_submit` and `ui_cmd`)
  - or use fine-grained flags: `AUTO_TEST_ENABLE_POST_SUBMIT_TOOL=true`, `AUTO_TEST_ENABLE_UI_CMD_TOOL=true`
- You can disable all interact tool registration with `AUTO_TEST_DISABLE_INTERACT_TOOLS=true`.
