# Auto Test 稳定契约与结果结构说明

## 1. 文档定位

本文档只记录**短期内不易变动**的内容，作为 `auto_test` 长期复用基线：

- 后端接口契约（创建会话、执行会话）
- 鉴权约定
- SSE 事件解析约定
- 结果目录结构与关键字段结构

> 说明：脚本名称、代码组织、运行命令等易变内容，不放在这里；请查看 `auto_test/README.md`。

---

## 2. 鉴权与请求头契约

自动测试请求需携带以下头部：

- `Authorization: Bearer <token>`
- `uid: <uid>`
- `email: <email>`
- `Content-Type: application/json`

说明：

- token 也可在配置中以 raw JWT 提供，代码层会自动补 `Bearer ` 前缀。
- 非 dev 环境通常会做 token 校验，建议把鉴权失败（401）作为明确失败类型记录。

---

## 3. 接口契约（稳定部分）

## 3.1 创建会话

- 方法：`POST`
- 路径：`/v1/lorevo/create_session`
- 最小请求体：

```json
{
  "title": "auto-test-session-title"
}
```

- 业务成功判定：
  - HTTP 2xx
  - 响应 `code` 为 `0` 或 `200`
  - 存在 `data.sessionId`

## 3.2 执行会话（SSE）

- 方法：`POST`
- 路径：`/v1/lorevo/execute_session`
- 最小请求体：

```json
{
  "sessionId": "session-id",
  "runSettings": {},
  "messages": [
    {
      "role": "user",
      "content": [
        { "text": "用户输入" }
      ]
    }
  ],
  "exec": {
    "maxTurns": 8
  }
}
```

说明：

- `runSettings` 结构可扩展，但留空对象 `{}` 仍是合法基线。
- 压缩/记忆专项测试可在场景层扩展 `messages` 与 `runSettings`，不改变接口主契约。

---

## 4. SSE 解析契约（稳定部分）

执行接口返回 `text/event-stream`，测试侧解析建议遵守：

1. 仅处理 `data:` 行
2. 忽略注释/心跳行（以 `:` 开头）
3. `data:` 载荷可能是：
   - 直接 `UiEvent` JSON
   - 外层对象内嵌 `UiEvent`
   - 字符串化 JSON

## 4.1 当前重点事件类型

- `TEXT_DELTA`：增量文本
- `TEXT`：聚合文本（优先作为最终回复）
- `RUN_END`：本轮运行结束
- `RUN_ERROR`：本轮运行失败

## 4.2 单轮成功判定（推荐）

- 收到 `RUN_END`
- 未收到 `RUN_ERROR`
- 助手回复非空

---

## 5. Trace 追踪信息契约

建议记录每次响应中的追踪头，便于后端排障：

- `traceparent`
- `X-Request-Id`
- `X-Trace-Id`（若存在）

并建议从 `traceparent` 解析 `trace_id` 作为主查询键。

---

## 6. Results 目录结构（当前基线）

每次运行必须落到独立目录，推荐：

```text
auto_test/results/session_<scenario>_<run_id>/
  run_meta.md
  dialogue.md
  evaluation.md
  raw_events.jsonl
  non_text/
    turn_results.json
    evaluation.json
    README.md
```

---

## 7. 测试输出结构（turn_results.json）

`turn_results.json` 为“每轮执行结果数组”，典型字段：

```json
[
  {
    "turn": 1,
    "request_id": "string",
    "traceparent": "string",
    "tracestate": "string",
    "x_trace_id": "string",
    "backend_trace_id": "string",
    "user_text": "string",
    "assistant_text": "string",
    "event_count": 12,
    "run_end": true,
    "run_error": "",
    "duration_sec": 3.21
  }
]
```

字段说明：

- `run_end` / `run_error`：轮次结果核心信号
- `assistant_text`：用于评估的用户可见回复
- `backend_trace_id`：排障索引字段

---

## 8. 评估输出结构（evaluation.json）

`evaluation.json` 顶层建议包含：

```json
{
  "generated_at": "2026-04-09T12:00:00",
  "rule_evaluation": {},
  "llm_evaluation": {}
}
```

## 8.1 规则评估结构（rule_evaluation）

核心字段建议：

- `mode`: `"rule"`
- `pass_threshold`: 数值阈值
- `overall_score`: 0~1
- `overall_pass`: 布尔
- `critical_failed`: 关键失败项列表
- `metrics`: 聚合指标（如 total_turns/success_turns）
- `expected_facts`: 期望硬事实集合
- `checks`: 逐项检查数组

`checks` 单项建议：

- `check_id`
- `name`
- `weight`
- `passed`
- `score`
- `critical`
- `detail`

## 8.2 LLM 评估结构（llm_evaluation）

建议兼容三类状态：

1. 未启用：`enabled=false`
2. 启用但跳过：`enabled=true, skipped=true, reason=...`
3. 启用并完成：`enabled=true, skipped=false, ...`

---

## 9. 变更边界建议

当后端有以下变化时，需要同步更新此文档：

1. `/v1/lorevo/create_session` 或 `/v1/lorevo/execute_session` 协议字段变化
2. SSE 事件结构或事件类型变化
3. trace 头部约定变化
4. results 产物结构变化

其余测试策略、脚本命名、运行入口、目录说明等，请在 `auto_test/README.md` 维护。

