# tests 目录说明

本目录存放 `auto_test` 的可执行测试入口与测试编排代码。

## 主要脚本

- `run_5turn_session_test.py`
  - 通用多轮会话自动测试入口（create_session -> execute_session -> SSE 解析 -> 结果落盘）。
  - 支持用户模拟器、整体评估（LLM）、探针评估、workspace 导出。

- `run_memory_compression_failure_scan.py`
  - 记忆压缩失效轮次扫描脚本。
  - 关键行为：
    - 每个 session 第1轮固定重置模板，第2轮埋入“编号探针”。
    - 非探针轮次优先使用 `user_simulator_engine`（角色 LLM）生成用户输入。
    - 默认持续对话，直到首次探针失效（即被遗忘）才结束该 session。
    - 前 `warmup_sessions`（默认2）用于粗定位，后续围绕估计失效轮次聚焦测试。
    - 支持并行 session（warmup 阶段仍顺序执行）。
  - 可选安全开关：
    - `--hard-max-turns`：每个 session 的硬上限（默认 `0`，表示不设上限）。

- `run_memory_failure_campaign.py`
  - 按“3并发 seed -> 估计失效轮 -> 3轮x3并发 focused”执行完整 campaign。
  - 结果统一放在一个 `memory_failure/<campaign_id>/` 下，便于整体溯源。

## 运行示例

```powershell
# 通用会话测试（test 环境，10轮）
python auto_test/src/tests/run_5turn_session_test.py --env test --max-turns 10

# 记忆压缩失效扫描（test 环境，10 个 session，默认跑到失效）
python auto_test/src/tests/run_memory_compression_failure_scan.py --env test --sessions 10

# 记忆压缩失效扫描（并行 + 安全上限）
python auto_test/src/tests/run_memory_compression_failure_scan.py --env test --sessions 10 --parallel-sessions 3 --hard-max-turns 80

# campaign：先3并发seed，再3轮x3并发focused
python auto_test/src/tests/run_memory_failure_campaign.py --env test --parallel-sessions 3
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
      workspace/
        _manifest.json
        ...
    session_02/
    ...
  aggregate/
    summary.json
    summary.md
    failure_turn_distribution.svg
```

```text
auto_test/results/memory_failure/<campaign_id>/
  seed_3sessions/
    sessions/...
    aggregate/summary.json
  focused_round_1/
    sessions/...
    aggregate/summary.json
  focused_round_2/
    sessions/...
    aggregate/summary.json
  focused_round_3/
    sessions/...
    aggregate/summary.json
  aggregate/
    campaign_summary.json
    campaign_summary.md
    focused_combined_distribution.svg
  run_manifest.json
```
