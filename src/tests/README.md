# tests 目录说明

本目录存放 `auto_test` 的可执行测试入口与测试编排代码。

## 运行前置行为

- 所有主入口脚本在正式执行前会自动清理历史遗留的 `auto_test/src/tests` 相关后台进程（例如上一次未退出的 `python/powershell` 测试进程）。
- 如需临时跳过，可使用 CLI 参数 `--skip-preflight-cleanup`，或设置环境变量 `AUTO_TEST_SKIP_PREFLIGHT_CLEANUP=true`。

## 断流兜底默认值

- `AUTO_TEST_TERMINAL_RECONCILE_GRACE_SEC` 默认 `90` 秒（SSE 中断后用于补偿查询终态事件）。
- `AUTO_TEST_EXECUTE_TRANSIENT_RETRIES` 默认 `3`（execute 请求瞬时错误重试次数）。
- `AUTO_TEST_QUERY_UI_EVENTS_RETRIES` 默认 `3`（补偿查询接口重试次数）。
- `AUTO_TEST_QUERY_UI_EVENTS_READ_TIMEOUT_SEC` 默认 `45` 秒（补偿查询单次读取超时）。

## 主要脚本

- `run_5turn_session_test.py`
  - 通用多轮会话自动测试入口（`create_session -> execute_session -> SSE 解析 -> 结果落盘`）。
  - 默认首轮也由角色模拟 LLM 生成，`first_user_message` 仅作为首轮失败兜底。
  - 支持用户模拟器、整体评估（LLM）、探针评估、workspace 导出。
- `run_memory_compression_failure_scan.py`
  - 记忆压缩失效轮次扫描脚本。
  - 关键行为：
    - 每个 session 第 1 轮使用普通开场语，第 2 轮埋入探针信息。
    - 若存在 `compression_retention_contract_v1.json`，优先使用其中 `plant_text_template` 和 `probe_text_templates`。
    - 非探针轮次优先使用 `user_simulator_engine`（角色 LLM）生成用户输入。
    - 默认持续对话，直到首次探针失效（被遗忘）或达到终止条件。
    - 前 `warmup_sessions` 用于粗定位，后续围绕估计失效轮次聚焦测试。
  - 输出新增（兼容旧结构）：
    - `run_data/compaction_events.json`：压缩相关事件时间线。
    - `run_data/probe_timeline.json`：按探针检查展开的分层判定时间线（tier/state/cause/phase）。
    - `aggregate/compression_effect_summary.json`：聚合统计（retention/post-compaction delta/false alarm）。
    - `aggregate/compression_effect_report.md`：聚合统计说明报告。
- `run_memory_failure_campaign.py`
  - 按 “3 seed -> 3 轮 focused(每轮 3 session)” 执行完整 campaign。

## 运行示例

```powershell
# 通用会话测试（test 环境，10 轮）
python auto_test/src/tests/run_5turn_session_test.py --env test --max-turns 10

# 记忆压缩失效扫描（test 环境，10 个 session）
python auto_test/src/tests/run_memory_compression_failure_scan.py --env test --sessions 10

# 带硬上限
python auto_test/src/tests/run_memory_compression_failure_scan.py --env test --sessions 10 --hard-max-turns 40

# 临时跳过启动前清理
python auto_test/src/tests/run_5turn_session_test.py --env test --max-turns 10 --skip-preflight-cleanup
```

## 结果目录（失效扫描）

```text
auto_test/results/memory_failure/<run_id>/
  run_manifest.json
  sessions/
    session_01/
      raw_events.jsonl
      session_meta.json
      run_data/
        turn_results.json
        probe_checks.json
        compaction_events.json
        probe_timeline.json
      workspace/
        _manifest.json
        ...
  aggregate/
    summary.json
    summary.md
    memory_failure_llm_report.md
    compression_effect_summary.json
    compression_effect_report.md
    failure_turn_distribution.svg
```
