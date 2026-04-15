# prompts 目录说明（Framework + Targets）

本目录采用两层架构，目标是把“通用测试机制”与“具体业务上下文”彻底解耦，避免提示词长期演进后和某个被测 Agent 强耦合。

## 1. 设计目标

1. 通用框架可复用：同一套评估与模拟逻辑，能复用于多个目标 Agent。
2. 目标配置可插拔：新增被测 Agent 时，不改框架 prompt，只新增 `targets/<target_name>/...`。
3. 占位符可追踪：每个模板变量都能明确来源，便于排查“为什么本轮输出异常”。
4. 迁移成本可控：保留统一目录约定，减少历史命名带来的维护负担。

## 2. 目录结构与职责

```text
prompts/
  framework/
    evaluator/
      overall/
        llm_eval_system.prompt
        llm_eval_user.prompt
      probe/
        probe_judge_system.prompt
        probe_judge_user.prompt
    simulator/
      system/
        user_simulator_system.prompt
        # 可选：scenario_template.prompt（不放时由代码内置模板兜底）
      role_generation/
        role_system.prompt
        role_user.prompt
  targets/
    advoo/
      scenarios/
        intent_pool.yaml
        policies.yaml
      personas/
        advoo_users.json
      rubrics/
        probe_rubrics.yaml
    agent_x/
      scenarios/
        intent_pool.yaml
        policies.yaml
      personas/
        agent_x_users.json
      rubrics/
        probe_rubrics.yaml
```

### 2.1 framework（通用层）

只放“机制”和“格式约束”，不放 advoo 或其他业务专属事实。

1. `evaluator/overall/*`
   - 用于整段多轮对话的 LLM 总评（连贯性、记忆等）。
2. `evaluator/probe/*`
   - 用于单条探针的 LLM 裁判评分。
3. `simulator/system/user_simulator_system.prompt`
   - 约束“模拟用户”的基本行为和输出协议。
4. `simulator/role_generation/*`
   - 负责“随机但可扮演”的用户角色生成。

### 2.2 targets（目标层）

按被测 Agent 隔离具体业务上下文与策略，不污染 framework。

1. `scenarios/intent_pool.yaml`
   - 目标领域的意图池（模拟用户回合级任务来源）。
2. `scenarios/policies.yaml`
   - 目标能力边界、限制、可接受行为。
3. `personas/*.json`
   - 目标场景下的用户画像样本（供 role 生成参考）。
4. `rubrics/probe_rubrics.yaml`
   - 目标专属探针评分细则（供探针裁判与场景注入参考）。

## 3. 运行时加载逻辑（关键）

`run_5turn_session_test.py` 会组合 framework + target：

1. 读取 `user_simulator.target_name`（默认 `advoo`）。
2. 加载 `prompts/targets/<target_name>/...` 下的意图池、策略、画像、rubric。
3. 将 target 内容注入 framework 模板占位符。
4. 生成本轮用户模拟与角色生成提示词。
5. 对话结束后，整体评估与探针评估分别读取 evaluator 对应模板。

说明：

1. `scenario_prompt_file` 是可选项；不配置时，系统会用内置 scenario 模板并注入 target 内容。
2. `framework/simulator/system/scenario_template.prompt` 是可选文件；不存在时仍可运行（代码内置模板兜底）。

## 4. 占位符清单与来源

### 4.1 对话总评（overall evaluator）

1. `{{EXPECTED_FACTS}}`：由评估器拼装的期望事实摘要。
2. `{{CONVERSATION}}`：完整多轮对话文本。

### 4.2 探针裁判（probe evaluator）

1. `{{PROBE_ID}}`
2. `{{PROBE_TYPE}}`
3. `{{PROBE_DESCRIPTION}}`
4. `{{PROBE_TARGET}}`
5. `{{PROBE_CONTEXT}}`
6. `{{PROBE_RUBRIC}}`
7. `{{PASS_THRESHOLD_0_5}}`

以上均来自探针数据集与评分配置拼装。

### 4.3 用户模拟与角色生成（simulator）

1. `{{MAX_TURNS}}`：本轮测试上限。
2. `{{REQUIRED_NOTEBOOK_CLEAR_TEXT}}`：首轮固定动作文本。
3. `{{CAPABILITY_MODE}}`：能力考验模式（如 alternating）。
4. `{{CAPABILITY_POLICY}}`：能力模式对应策略文本。
5. `{{TARGET_NAME}}`：当前 target 名称。
6. `{{TARGET_INTENT_POOL}}`：`targets/<target>/scenarios/intent_pool.yaml` 内容。
7. `{{TARGET_POLICIES}}`：`targets/<target>/scenarios/policies.yaml` 内容。
8. `{{TARGET_PERSONAS}}`：`targets/<target>/personas/*.json` 内容。
9. `{{TARGET_RUBRICS}}`：`targets/<target>/rubrics/probe_rubrics.yaml` 内容。
10. `{{INDUSTRY_OPTIONS}}`：行业随机池（可由配置覆盖）。
11. `{{IDENTITY_OPTIONS}}`：身份随机池（可由配置覆盖）。

## 5. 新增一个 target 的步骤

以 `agent_x` 为例：

1. 新建目录：`prompts/targets/agent_x/`
2. 填写 `scenarios/intent_pool.yaml`
3. 填写 `scenarios/policies.yaml`
4. 填写 `personas/agent_x_users.json`（文件名建议与 target 同名）
5. 填写 `rubrics/probe_rubrics.yaml`
6. 在配置中设置：`"user_simulator.target_name": "agent_x"`
7. 运行一次短测验证：`python auto_test/src/tests/run_5turn_session_test.py --max-turns 1`

## 6. 编写规范（建议强执行）

1. framework prompt 不写任何目标专属名词。
2. target 文件只写业务事实与边界，不写通用输出协议。
3. 所有 prompt 以“可结构化解析”为目标，避免含糊输出要求。
4. 占位符命名大写蛇形，避免同义重复变量。
5. 修改 prompt 后至少跑一次 `--max-turns 1` 冒烟测试。

## 7. 常见问题

### 7.1 为什么我改了 target 文件但效果没变？

优先检查：

1. `config.local.json` 的 `user_simulator.target_name` 是否指向正确 target。
2. 是否误改了未被当前配置引用的文件。
3. 占位符是否拼写错误（例如少一个大括号）。

### 7.2 为什么没有 `scenario_template.prompt` 也能跑？

因为 scenario 模板有代码内置兜底。只有你需要强控模板语气/结构时，才需要显式创建该文件。

### 7.3 framework 要不要兼容旧命名？

当前建议不再新增旧命名依赖，统一使用新路径，降低认知负担与维护成本。
