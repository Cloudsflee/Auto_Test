# auto_test 项目说明（当前现状版）

更新时间：2026-04-22  
适用对象：需要快速理解本项目自动化测试能力、产物结构和压缩测试现状的同学

如果你是评审/业务同学，不看代码，优先阅读：
1. `auto_test/doc/自动化测试全局现状与理论总览.md`（全局）
2. `auto_test/doc/压缩测试现状与理论说明.md`（压缩专题）

## 1. 这个目录是做什么的
`auto_test` 是 `ai-backend` 的自动化测试子项目，目标是稳定、可复现地跑多轮会话测试，并输出可审计的测试产物。

当前重点能力：
1. 自动创建会话并执行多轮对话。
2. 解析 SSE 事件并落盘原始事件流。
3. 跑探针（probe）检查记忆保持/遗忘情况。
4. 导出 workspace 文本与图片产物。
5. 产出聚合报告（健康度 + 压缩效果），并支持中文解析报告。

## 2. 你最需要关注的脚本
1. `auto_test/src/tests/run_5turn_session_test.py`  
通用会话测试入口（create -> execute -> SSE -> 落盘 -> 评估）。

2. `auto_test/src/tests/run_memory_compression_failure_scan.py`  
压缩测试主脚本。按 session 运行多轮对话，混合探针轮与普通轮，输出压缩相关聚合指标。

3. `auto_test/src/tests/postprocess_memory_failure_cn_report.py`  
对某次 `memory_failure/<run_id>` 产物做中文解析，输出“是否具备统计意义”的判断。

## 3. 角色模拟 LLM 现状
1. 当前压缩测试默认使用角色模拟器生成用户输入（`provider_context` 模式）。
2. 首轮用户消息优先由模拟器生成。
3. `first_user_message` 仍保留在配置中作为“首轮失败兜底”，不是固定首句。
4. 当模拟器异常或空输出时，脚本会回退到兜底文案，避免会话启动失败。
5. 非探针轮次同样优先走模拟器，探针轮次由探针模板或探针生成逻辑接管。

## 4. 压缩完成判定口径（当前实现）
压缩完成采用“双通道”判定，避免漏识别：

1. 主证据（tool lifecycle）  
同轮出现：`memory_file_compaction` + `summary_compaction` 的 `TOOL_CALL_START/TOOL_RESULT`，并且后续有 `RUN_END`。

2. 兼容证据（indicator text）  
同轮出现压缩完成文本（如“压缩完成”）且后续有 `RUN_END`。

说明：前端出现“压缩完成”但工具生命周期事件缺失时，仍可通过兼容证据判定为完成。

## 5. 探针样本的两种口径（务必区分）
1. 独立探针口径（推荐用于统计判断）  
每条 `probe_check` 只计 1 次；只统计 `turn > first_compaction_done_turn` 的样本。

2. 展开口径（系统内部时间线）  
`probe_timeline` 会按 anchor points 展开（当前通常是 3 个点），所以计数会放大。

结论：看统计意义时优先看“独立探针口径”。

## 6. 结果目录结构
```text
auto_test/results/memory_failure/<run_id>/
  run_manifest.json
  aggregate/
    summary.json
    summary.md
    memory_failure_llm_report.md
    pipeline_health_summary.json
    pipeline_health_report.md
    compression_effect_summary.json
    compression_effect_report.md
    failure_turn_distribution.svg
    cn_parsed_summary.json
    cn_parsed_report.md
  sessions/
    session_01/
      raw_events.jsonl
      dialogue.md
      session_meta.json
      run_data/
        turn_results.json
        probe_checks.json
        compaction_events.json
        probe_timeline.json
      workspace/
        _manifest.json
        ...
```

## 7. 常用命令（直接可跑）
1. 通用 smoke：
```powershell
python auto_test/src/tests/run_5turn_session_test.py --max-turns 5
```

2. 压缩测试（串行 10 session，每个 35 轮）：
```powershell
python auto_test/src/tests/run_memory_compression_failure_scan.py --sessions 10 --warmup-sessions 0 --hard-max-turns 35 --parallel-sessions 1 --turn-timeout-sec 900
```

3. 生成中文解析报告：
```powershell
python auto_test/src/tests/postprocess_memory_failure_cn_report.py --run-id <run_id>
```

## 8. 当前现状快照（基于最新 10x35 串行测试）
参考 run：`20260422_112931_dc745e42`

1. 配置：`10 sessions`、`35 轮/每 session`、`parallel_sessions=1`（不并发）。
2. 压缩完成验证 session：`4/10`。
3. 独立探针后压缩样本：`must_keep=20, should_keep=21, may_drop=21`（总计 62）。
4. 链路健康：出现部分 `run_error`（主要是 `ChunkedEncodingError/502`），导致有效 session 比例下降。
5. 中文解析结论：当前更适合“流程验证”，统计结论仍需更多“压缩完成且稳定跑完”的 session。

## 9. 如何判断“统计意义够不够”
推荐基线：
1. 完成压缩验证的有效 session 至少 `>=3`（更稳建议 `>=10`）。
2. 每个 tier 的后压缩“独立探针样本”至少 `>=5`（更稳建议 `>=10`）。
3. 健康指标稳定：`timeout/stream_closed/run_error` 比率尽量低。

## 10. 已知问题与建议
1. 上游网络或服务波动会导致 `ChunkedEncodingError` 或 `502`，影响有效样本。
2. 单次 run 的“总 session 数”不等于“有效统计 session 数”，需要看 `sessions_verified_compaction`。
3. 对外汇报时建议同时附带：
`cn_parsed_report.md` + `cn_parsed_summary.json` + `run_manifest.json`。

## 11. 相关文档
1. `auto_test/src/tests/README.md`
2. `auto_test/config/README.md`
3. `auto_test/prompts/README.md`
4. `auto_test/datasets/probes/README.md`
5. `auto_test/datasets/probes/compression_retention_contract_v1.json`
