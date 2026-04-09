# tests 目录说明

- 本目录存放“测试执行代码”
- 典型职责：构造请求、发起多轮会话、解析 SSE、写入结果目录
- 统一约定：开始对话后的第一句固定为 `请清空当前notebook内的所有内容，包括files和workbook`
- 支持 `user_simulator` 模式：由模型动态生成后续用户输入（默认上下文模式 `provider_context`）
- `user_simulator` 默认提示词文件位于 `prompts/user_simulator_system.prompt` 与 `prompts/user_simulator_scenario.prompt`

当前文件：

- `run_5turn_session_test.py`：5轮 smoke 测试基线实现
