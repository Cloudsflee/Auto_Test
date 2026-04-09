from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# 允许直接运行本文件时从 src/ 下导入评估模块。
AUTO_TEST_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = AUTO_TEST_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.dialogue_evaluator import LLMEvalConfig, evaluate_rules, evaluate_with_llm, write_evaluation_md


CFG_PATH_CANDIDATES = [
    AUTO_TEST_DIR / "config" / "config.local.json",
    AUTO_TEST_DIR / "config" / "config.json",
    AUTO_TEST_DIR / "test.txt",
    AUTO_TEST_DIR / "test_re.txt",
]


@dataclass
class AuthConfig:
    base_url: str
    token: str
    uid: str
    email: str
    source_path: Path
    llm_eval: dict[str, Any]
    user_simulator: dict[str, Any]


@dataclass
class TraceInfo:
    request_id: str
    traceparent: str
    tracestate: str
    x_trace_id: str
    backend_trace_id: str


@dataclass
class UserSimulatorConfig:
    enabled: bool
    mode: str
    base_url: str
    model: str
    api_key: str
    timeout_sec: int
    max_turns: int
    first_user_message: str
    system_prompt: str
    scenario_prompt: str


@dataclass
class UserSimulatorState:
    previous_response_id: str


REQUIRED_NOTEBOOK_CLEAR_TEXT = "请清空当前notebook内的所有内容，包括files和workbook"
DEFAULT_FIRST_USER_TEXT = (
    REQUIRED_NOTEBOOK_CLEAR_TEXT
    + "。我叫阿龙，我的品牌叫星野咖啡，主打低糖拿铁。请记住这三个信息。"
)
PROMPTS_DIR = AUTO_TEST_DIR / "prompts"
DEFAULT_USER_SIM_SYSTEM_PROMPT_FILE = PROMPTS_DIR / "user_simulator_system.prompt"
DEFAULT_USER_SIM_SCENARIO_PROMPT_FILE = PROMPTS_DIR / "user_simulator_scenario.prompt"


def _pick_text_value(text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _resolve_prompt_path(file_name: str, default_path: Path) -> Path:
    name = (file_name or "").strip()
    if not name:
        return default_path
    as_path = Path(name)
    if as_path.is_absolute():
        return as_path
    return PROMPTS_DIR / as_path


def _load_prompt_template(prompt_path: Path, fallback: str) -> str:
    if not prompt_path.exists():
        return fallback
    text = prompt_path.read_text(encoding="utf-8", errors="ignore").strip()
    return text or fallback


def _parse_trace_id_from_traceparent(traceparent: str) -> str:
    tp = (traceparent or "").strip()
    parts = tp.split("-")
    if len(parts) != 4:
        return ""
    trace_id = parts[1].strip()
    if len(trace_id) == 32 and all(c in "0123456789abcdefABCDEF" for c in trace_id):
        return trace_id.lower()
    return ""


def _env_float(key: str, default: float) -> float:
    import os

    value = os.getenv(key, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _env_int(key: str, default: int) -> int:
    import os

    value = os.getenv(key, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _env_bool(key: str, default: bool) -> bool:
    import os

    value = os.getenv(key, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _cfg_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    return default


def _cfg_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return default
        try:
            return int(raw)
        except Exception:
            return default
    return default


def resolve_config_path() -> Path:
    for path in CFG_PATH_CANDIDATES:
        if path.exists():
            return path
    return CFG_PATH_CANDIDATES[0]


def _load_json_config(path: Path) -> AuthConfig:
    obj = _safe_json_loads(path.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"配置文件不是合法 JSON: {path.as_posix()}")

    auth = obj.get("auth")
    if not isinstance(auth, dict):
        auth = {}

    base_url = str(obj.get("base_url", "")).strip().rstrip("/")
    token = str(auth.get("token", "") or obj.get("token", "")).strip()
    uid = str(auth.get("uid", "") or obj.get("uid", "")).strip()
    email = str(auth.get("email", "") or obj.get("email", "")).strip()
    llm_eval = obj.get("llm_eval")
    if not isinstance(llm_eval, dict):
        llm_eval = {}
    user_simulator = obj.get("user_simulator")
    if not isinstance(user_simulator, dict):
        user_simulator = {}

    if not base_url or not token or not uid or not email:
        raise RuntimeError(f"缺少必填字段: base_url/token/uid/email, path={path.as_posix()}")

    return AuthConfig(
        base_url=base_url,
        token=token,
        uid=uid,
        email=email,
        source_path=path,
        llm_eval=llm_eval,
        user_simulator=user_simulator,
    )


def _load_text_config(path: Path) -> AuthConfig:
    text = path.read_text(encoding="utf-8", errors="ignore")
    base_url = _pick_text_value(text, "base_url").rstrip("/")
    token = _pick_text_value(text, "token")
    uid = _pick_text_value(text, "uid")
    email = _pick_text_value(text, "email")
    if not base_url or not token or not uid or not email:
        raise RuntimeError(f"缺少必填字段: base_url/token/uid/email, path={path.as_posix()}")
    return AuthConfig(
        base_url=base_url,
        token=token,
        uid=uid,
        email=email,
        source_path=path,
        llm_eval={},
        user_simulator={},
    )


def load_config(path: Path) -> AuthConfig:
    if not path.exists():
        raise RuntimeError(f"配置文件不存在: {path.as_posix()}")
    if path.suffix.lower() == ".json":
        return _load_json_config(path)
    return _load_text_config(path)


def normalize_authz(token: str) -> str:
    value = token.strip()
    if value.lower().startswith("bearer "):
        return value
    return f"Bearer {value}"


def mask_token(token: str) -> str:
    raw = token.strip()
    if raw.lower().startswith("bearer "):
        raw = raw.split(" ", 1)[1].strip()
    if len(raw) <= 16:
        return "*" * len(raw)
    return f"{raw[:8]}...{raw[-8:]}"


def build_llm_eval_config(cfg: AuthConfig) -> LLMEvalConfig:
    llm_cfg = cfg.llm_eval if isinstance(cfg.llm_eval, dict) else {}
    return LLMEvalConfig(
        enabled=_cfg_bool(
            llm_cfg.get("enabled"),
            _env_bool("AUTO_TEST_ENABLE_LLM_EVAL", False),
        ),
        base_url=str(llm_cfg.get("base_url") or llm_cfg.get("url") or os.getenv("AUTO_TEST_EVAL_LLM_URL", "")).strip(),
        model=str(llm_cfg.get("model") or os.getenv("AUTO_TEST_EVAL_LLM_MODEL", "")).strip(),
        api_key=str(llm_cfg.get("api_key") or llm_cfg.get("key") or os.getenv("AUTO_TEST_EVAL_LLM_API_KEY", "")).strip(),
        timeout_sec=_cfg_int(
            llm_cfg.get("timeout_sec"),
            _env_int("AUTO_TEST_EVAL_LLM_TIMEOUT_SEC", 30),
        ),
    )


def _ensure_required_note_clear_text(text: str) -> str:
    normalized = (text or "").strip()
    if not normalized:
        return DEFAULT_FIRST_USER_TEXT
    if normalized.startswith(REQUIRED_NOTEBOOK_CLEAR_TEXT):
        return normalized
    return f"{REQUIRED_NOTEBOOK_CLEAR_TEXT}。{normalized}"


def _detect_wire_api(url: str) -> str:
    lower = (url or "").lower()
    if lower.endswith("/responses") or "/responses?" in lower:
        return "responses"
    return "chat_completions"


def _extract_text_from_content_field(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _extract_text_from_chat_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""

    msg = first.get("message")
    if isinstance(msg, dict):
        return _extract_text_from_content_field(msg.get("content")).strip()

    delta = first.get("delta")
    if isinstance(delta, dict):
        return _extract_text_from_content_field(delta.get("content")).strip()
    return ""


def _extract_text_from_responses_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    parts: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "output_text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
    merged = "".join(parts).strip()
    if merged:
        return merged
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()
    return ""


def _extract_text_from_sse_stream(raw_text: str) -> str:
    chunks: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        chunk = _safe_json_loads(data)
        if not isinstance(chunk, dict):
            continue
        text = _extract_text_from_chat_payload(chunk)
        if text:
            chunks.append(text)
    return "".join(chunks).strip()


def build_user_simulator_config(cfg: AuthConfig, llm_eval_cfg: LLMEvalConfig) -> UserSimulatorConfig:
    sim_cfg = cfg.user_simulator if isinstance(cfg.user_simulator, dict) else {}

    mode = str(sim_cfg.get("mode") or os.getenv("AUTO_TEST_USER_SIM_MODE", "provider_context")).strip().lower()
    if mode not in {"provider_context", "explicit_context"}:
        mode = "provider_context"

    first_user_message = _ensure_required_note_clear_text(
        str(sim_cfg.get("first_user_message") or "").strip() or DEFAULT_FIRST_USER_TEXT
    )

    default_system_prompt = (
        "你是自动化测试中的“客户用户模拟器”。"
        "你的任务是基于助手上一轮回复，生成下一轮用户提问。"
        "你必须只返回 JSON，对象字段包含：user_text, stop, reason。"
        "其中 user_text 必须是给助手的自然语言用户输入，不要输出解释。"
    )
    default_scenario_prompt = (
        "你在扮演随机客户，与助手进行多轮沟通。\n"
        "事实背景：\n{{EXPECTED_FACTS}}\n\n"
        "随机性要求：\n"
        "1. 每轮随机选择一个意图提问，不固定顺序。\n"
        "2. 意图池：口号创作/卖点追问/预算约束/竞品比较/复述核对/总结要求。\n"
        "3. 语气随机（正式、口语、追问、质疑），长度随机（10~80字）。\n"
        "4. 不要重复上一轮用户句式。\n\n"
        "轮次上限：{{MAX_TURNS}}。仅在你认为目标完成时再 stop=true。"
    )
    system_prompt_path = _resolve_prompt_path(
        str(sim_cfg.get("system_prompt_file") or "").strip(),
        DEFAULT_USER_SIM_SYSTEM_PROMPT_FILE,
    )
    scenario_prompt_path = _resolve_prompt_path(
        str(sim_cfg.get("scenario_prompt_file") or "").strip(),
        DEFAULT_USER_SIM_SCENARIO_PROMPT_FILE,
    )
    system_prompt = str(sim_cfg.get("system_prompt") or "").strip() or _load_prompt_template(
        system_prompt_path,
        default_system_prompt,
    )
    scenario_prompt = str(sim_cfg.get("scenario_prompt") or "").strip() or _load_prompt_template(
        scenario_prompt_path,
        default_scenario_prompt,
    )

    return UserSimulatorConfig(
        enabled=_cfg_bool(
            sim_cfg.get("enabled"),
            _env_bool("AUTO_TEST_ENABLE_USER_SIMULATOR", False),
        ),
        mode=mode,
        base_url=str(
            sim_cfg.get("base_url")
            or sim_cfg.get("url")
            or os.getenv("AUTO_TEST_USER_SIM_BASE_URL", "")
            or llm_eval_cfg.base_url
        ).strip(),
        model=str(sim_cfg.get("model") or os.getenv("AUTO_TEST_USER_SIM_MODEL", "") or llm_eval_cfg.model).strip(),
        api_key=str(sim_cfg.get("api_key") or sim_cfg.get("key") or os.getenv("AUTO_TEST_USER_SIM_API_KEY", "")).strip()
        or llm_eval_cfg.api_key,
        timeout_sec=_cfg_int(
            sim_cfg.get("timeout_sec"),
            _env_int("AUTO_TEST_USER_SIM_TIMEOUT_SEC", 60),
        ),
        max_turns=max(1, min(128, _cfg_int(sim_cfg.get("max_turns"), _env_int("AUTO_TEST_USER_SIM_MAX_TURNS", 5)))),
        first_user_message=first_user_message,
        system_prompt=system_prompt,
        scenario_prompt=scenario_prompt,
    )


def _render_history_for_user_sim(results: list[dict[str, Any]], max_items: int = 6) -> str:
    if not results:
        return "(empty)"
    tail = results[-max_items:]
    lines: list[str] = []
    for item in tail:
        lines.extend(
            [
                f"Turn {item.get('turn')}",
                f"User: {item.get('user_text') or ''}",
                f"Assistant: {item.get('assistant_text') or ''}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _render_user_sim_scenario(template: str, expected_facts: dict[str, str], max_turns: int) -> str:
    facts = "\n".join(f"- {k}: {v}" for k, v in expected_facts.items())
    return (
        (template or "")
        .replace("{{EXPECTED_FACTS}}", facts)
        .replace("{{MAX_TURNS}}", str(max_turns))
    )


def _parse_user_sim_output(content: str) -> tuple[str, bool, str]:
    parsed = _safe_json_loads(content)
    if isinstance(parsed, dict):
        user_text = str(
            parsed.get("user_text") or parsed.get("next_user_message") or parsed.get("message") or ""
        ).strip()
        stop = _cfg_bool(parsed.get("stop"), False)
        reason = str(parsed.get("reason") or "").strip()
        return user_text, stop, reason
    return content.strip(), False, ""


def generate_user_turn_with_simulator(
    sim_cfg: UserSimulatorConfig,
    sim_state: UserSimulatorState,
    results: list[dict[str, Any]],
    expected_facts: dict[str, str],
    turn_idx: int,
) -> tuple[str, bool, UserSimulatorState]:
    scenario_prompt = _render_user_sim_scenario(sim_cfg.scenario_prompt, expected_facts, sim_cfg.max_turns)
    latest_assistant = str((results[-1] if results else {}).get("assistant_text") or "")
    history_text = _render_history_for_user_sim(results)

    user_prompt = (
        f"当前请生成第 {turn_idx} 轮用户输入。\n\n"
        f"场景要求：\n{scenario_prompt}\n\n"
        f"最近多轮记录：\n{history_text}\n\n"
        f"上一轮助手回复：\n{latest_assistant}\n\n"
        "输出要求：\n"
        "1. 只输出 JSON。\n"
        "2. JSON 格式：{\"user_text\":\"...\",\"stop\":false,\"reason\":\"\"}\n"
        "3. user_text 里不要出现工具调用、JSON、XML。\n"
        "4. 如果应该结束，stop=true 且 user_text 置空。"
    )

    headers = {
        "Authorization": f"Bearer {sim_cfg.api_key}",
        "Content-Type": "application/json",
    }
    wire_api = _detect_wire_api(sim_cfg.base_url)
    body: dict[str, Any]
    if wire_api == "responses":
        body = {
            "model": sim_cfg.model,
            "temperature": 0.7,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": sim_cfg.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            "text": {"format": {"type": "json_object"}},
        }
        if sim_cfg.mode == "provider_context" and sim_state.previous_response_id:
            body["previous_response_id"] = sim_state.previous_response_id
    else:
        body = {
            "model": sim_cfg.model,
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": sim_cfg.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

    resp = requests.post(sim_cfg.base_url, headers=headers, json=body, timeout=sim_cfg.timeout_sec)
    resp.raise_for_status()
    raw_text = resp.text if isinstance(resp.text, str) else ""
    payload = _safe_json_loads(raw_text)

    if wire_api == "responses":
        content = _extract_text_from_responses_payload(payload)
    else:
        content = _extract_text_from_chat_payload(payload)

    if not content and raw_text.lstrip().startswith("data:"):
        content = _extract_text_from_sse_stream(raw_text)

    user_text, stop, _ = _parse_user_sim_output(content)
    next_prev_id = ""
    if isinstance(payload, dict):
        next_prev_id = str(payload.get("id") or "").strip()

    if not user_text and not stop:
        raise RuntimeError(f"user_simulator 输出为空: {content[:200]}")
    return user_text, stop, UserSimulatorState(previous_response_id=next_prev_id or sim_state.previous_response_id)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def trace_info_from_response(resp: requests.Response) -> TraceInfo:
    request_id = (resp.headers.get("X-Request-Id") or "").strip()
    traceparent = (resp.headers.get("traceparent") or "").strip()
    tracestate = (resp.headers.get("tracestate") or "").strip()
    x_trace_id = (resp.headers.get("X-Trace-Id") or "").strip()
    backend_trace_id = _parse_trace_id_from_traceparent(traceparent) or x_trace_id
    return TraceInfo(
        request_id=request_id,
        traceparent=traceparent,
        tracestate=tracestate,
        x_trace_id=x_trace_id,
        backend_trace_id=backend_trace_id,
    )


def extract_ui_event(payload: str) -> dict[str, Any] | None:
    obj = _safe_json_loads(payload)
    if isinstance(obj, dict) and "seq" in obj and "data" in obj:
        return obj
    if isinstance(obj, dict) and "data" in obj:
        inner = obj.get("data")
        if isinstance(inner, str):
            inner_json = _safe_json_loads(inner)
            if isinstance(inner_json, dict) and "seq" in inner_json and "data" in inner_json:
                return inner_json
        if isinstance(inner, dict) and "seq" in inner and "data" in inner:
            return inner
    if isinstance(obj, str):
        text_json = _safe_json_loads(obj)
        if isinstance(text_json, dict) and "seq" in text_json and "data" in text_json:
            return text_json
    return None


def create_session(
    base_url: str,
    headers: dict[str, str],
    persist_type: int,
    settings_json: str | None,
) -> tuple[str, TraceInfo]:
    url = f"{base_url}/v1/lorevo/create_session"
    body: dict[str, Any] = {
        "title": f"auto-test-5turn-{int(time.time())}",
        "persistType": persist_type,
    }
    if settings_json:
        body["settings"] = settings_json
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    trace = trace_info_from_response(resp)
    resp.raise_for_status()

    payload = resp.json()
    code = payload.get("code")
    if code not in (0, 200):
        raise RuntimeError(f"create_session 业务错误: {payload}")

    session_id = ((payload.get("data") or {}).get("sessionId") or "").strip()
    if not session_id:
        raise RuntimeError(f"create_session 缺少 sessionId: {payload}")
    return session_id, trace


def execute_turn(
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    persist_type: int,
    exec_max_turns: int,
    run_settings: dict[str, Any],
    user_text: str,
    turn_idx: int,
    raw_events_fp,
) -> dict[str, Any]:
    url = f"{base_url}/v1/lorevo/execute_session"
    req_body = {
        "sessionId": session_id,
        "persistType": persist_type,
        "runSettings": run_settings,
        "messages": [{"role": "user", "content": [{"text": user_text}]}],
        "exec": {"maxTurns": exec_max_turns},
    }

    started = time.time()
    resp = requests.post(url, headers=headers, json=req_body, stream=True, timeout=180)
    trace = trace_info_from_response(resp)
    resp.raise_for_status()

    event_count = 0
    run_end = False
    run_error = ""
    assistant_final = ""
    assistant_deltas: list[str] = []

    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            continue
        if line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue

        payload = line[5:].strip()
        ui = extract_ui_event(payload)
        if not ui:
            continue
        event_count += 1

        data_wrap = ui.get("data")
        event = data_wrap.get("data") if isinstance(data_wrap, dict) else None
        if not isinstance(event, dict):
            continue

        event_type = str(event.get("type") or "")
        role = str(event.get("role") or "")
        content = event.get("content")

        raw_events_fp.write(
            json.dumps(
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "turn": turn_idx,
                    "request_id": trace.request_id,
                    "traceparent": trace.traceparent,
                    "backend_trace_id": trace.backend_trace_id,
                    "session_id": session_id,
                    "event_type": event_type,
                    "event_role": role,
                    "event_seq": ui.get("seq"),
                    "event_sub_seq": ui.get("subSeq", ui.get("sub_seq")),
                    "event_raw": event,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        if event_type == "TEXT_DELTA":
            if isinstance(content, str) and content:
                assistant_deltas.append(content)
        elif event_type == "TEXT":
            if role.lower() != "user" and isinstance(content, str) and content:
                assistant_final = content
        elif event_type == "RUN_ERROR":
            run_error = str(event.get("message") or "RUN_ERROR")
            break
        elif event_type == "RUN_END":
            run_end = True
            break

        if time.time() - started > 160:
            break

    if not assistant_final and assistant_deltas:
        assistant_final = "".join(assistant_deltas)

    duration_sec = round(time.time() - started, 2)
    return {
        "turn": turn_idx,
        "request_id": trace.request_id,
        "traceparent": trace.traceparent,
        "tracestate": trace.tracestate,
        "x_trace_id": trace.x_trace_id,
        "backend_trace_id": trace.backend_trace_id,
        "user_text": user_text,
        "assistant_text": assistant_final,
        "event_count": event_count,
        "run_end": run_end,
        "run_error": run_error,
        "duration_sec": duration_sec,
    }


def build_turns() -> list[str]:
    # 说明：用 Unicode 转义，避免 Windows 终端编码导致中文变成 ????。
    return [
        DEFAULT_FIRST_USER_TEXT,
        "\u57fa\u4e8e\u6211\u7684\u54c1\u724c\u5b9a\u4f4d\uff0c\u7ed9\u6211\u4e09\u6761\u53e3\u53f7\uff0c\u6bcf\u6761\u4e0d\u8d85\u8fc712\u4e2a\u5b57\u3002",
        "\u5148\u4e0d\u7528\u53e3\u53f7\u4e86\uff0c\u4f60\u7b80\u5355\u8bf4\u8bf4\u4f4e\u7cd6\u996e\u98df\u6709\u54ea\u4e9b\u597d\u5904\u3002",
        "\u73b0\u5728\u8bf7\u56de\u7b54\uff1a\u6211\u53eb\u4ec0\u4e48\uff1f\u54c1\u724c\u540d\u53eb\u4ec0\u4e48\uff1f\u4e3b\u6253\u4ec0\u4e48\uff1f",
        "\u6700\u540e\u8bf7\u7528\u4e00\u6bb5\u8bdd\u603b\u7ed3\uff0c\u5305\u542b\u6211\u7684\u540d\u5b57\u3001\u54c1\u724c\u540d\u548c\u4e3b\u6253\u5356\u70b9\u3002",
    ]


def build_run_settings() -> dict[str, Any]:
    import os

    # 对齐前端：默认开启记忆流程依赖的非中断工具，避免“要求记住/清空 notebook”
    # 场景下模型因工具不可用而反复自述执行计划。
    default_settings: dict[str, Any] = {
        "tools": {
            "read_text": None,
            "ls": None,
            "edit": None,
            "write": None,
            "time_now": None,
        }
    }

    raw = os.getenv("AUTO_TEST_RUN_SETTINGS_JSON", "").strip()
    if not raw:
        return default_settings

    custom = _safe_json_loads(raw)
    if not isinstance(custom, dict):
        raise RuntimeError("AUTO_TEST_RUN_SETTINGS_JSON 必须是 JSON 对象")
    return custom


def build_persist_type() -> int:
    persist_type = _env_int("AUTO_TEST_PERSIST_TYPE", 0)
    if persist_type not in (0, 1, 2):
        raise RuntimeError("AUTO_TEST_PERSIST_TYPE 仅支持 0(db) / 1(cache) / 2(one-shot)")
    return persist_type


def build_exec_max_turns() -> int:
    max_turns = _env_int("AUTO_TEST_EXEC_MAX_TURNS", 8)
    if max_turns < 1 or max_turns > 128:
        raise RuntimeError("AUTO_TEST_EXEC_MAX_TURNS 需在 1~128 之间")
    return max_turns


def build_create_settings_json(run_settings: dict[str, Any]) -> str | None:
    import os

    # 优先使用显式配置；用于对齐前端“会话创建即保存 settings”的路径。
    raw = os.getenv("AUTO_TEST_CREATE_SETTINGS_JSON", "").strip()
    if raw:
        custom = _safe_json_loads(raw)
        if not isinstance(custom, dict):
            raise RuntimeError("AUTO_TEST_CREATE_SETTINGS_JSON 必须是 JSON 对象")
        return json.dumps(custom, ensure_ascii=False, sort_keys=True)

    # 默认把 runSettings 同步到 create_session.settings，避免 skillIds 只传 execute_session
    # 但未进入初始系统 prompt 的情况。
    if _env_bool("AUTO_TEST_CREATE_SETTINGS_FROM_RUN_SETTINGS", True):
        return json.dumps(run_settings, ensure_ascii=False, sort_keys=True)
    return None


def build_expected_facts() -> dict[str, str]:
    return {
        "name": "\u963f\u9f99",
        "brand": "\u661f\u91ce\u5496\u5561",
        "product": "\u4f4e\u7cd6\u62ff\u94c1",
    }


def write_meta_md(
    path: Path,
    cfg: AuthConfig,
    session_id: str,
    run_id: str,
    user_sim_cfg: UserSimulatorConfig,
    persist_type: int,
    exec_max_turns: int,
    create_settings_json: str | None,
    run_settings: dict[str, Any],
    create_trace: TraceInfo,
    results: list[dict[str, Any]],
) -> None:
    ok_turns = sum(1 for r in results if r.get("run_end") and not r.get("run_error"))
    lines = [
        "# 5轮对话测试元信息",
        "",
        f"- run_id: `{run_id}`",
        f"- session_id: `{session_id}`",
        f"- base_url: `{cfg.base_url}`",
        f"- uid: `{cfg.uid}`",
        f"- email: `{cfg.email}`",
        f"- token(masked): `{mask_token(cfg.token)}`",
        f"- config_source: `{cfg.source_path.as_posix()}`",
        f"- created_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- user_simulator_enabled: `{user_sim_cfg.enabled}`",
        f"- user_simulator_mode: `{user_sim_cfg.mode}`",
        f"- user_simulator_max_turns: `{user_sim_cfg.max_turns}`",
        f"- user_simulator_model: `{user_sim_cfg.model}`",
        f"- user_simulator_base_url: `{user_sim_cfg.base_url}`",
        f"- persist_type: `{persist_type}`",
        f"- exec_max_turns: `{exec_max_turns}`",
        f"- run_settings: `{json.dumps(run_settings, ensure_ascii=False, sort_keys=True)}`",
        f"- create_settings: `{create_settings_json or ''}`",
        "",
        "## Create Session Trace",
        "",
        f"- request_id: `{create_trace.request_id}`",
        f"- backend_trace_id: `{create_trace.backend_trace_id}`",
        f"- traceparent: `{create_trace.traceparent}`",
        f"- x_trace_id: `{create_trace.x_trace_id}`",
        "",
        "## Turn Status",
        "",
        "| turn | request_id | backend_trace_id | run_end | run_error | event_count | duration_sec |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result['turn']} | `{result['request_id']}` | `{result['backend_trace_id']}` | "
            f"`{result['run_end']}` | `{result['run_error'] or ''}` | {result['event_count']} | {result['duration_sec']} |"
        )

    lines.extend(
        [
            "",
            f"- success_turns: `{ok_turns}/{len(results)}`",
            "",
            "## 追踪字段说明",
            "",
            "- trace 来源于后端响应头，不在客户端自造 trace id。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dialogue_md(path: Path, results: list[dict[str, Any]]) -> None:
    lines = ["# 用户视角对话记录", ""]
    for result in results:
        lines.extend(
            [
                f"## Turn {result['turn']}",
                "",
                f"**User**: {result['user_text']}",
                "",
                f"**Assistant**: {result['assistant_text'] or '(empty)'}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    cfg_path = resolve_config_path()
    cfg = load_config(cfg_path)
    print(f"[INFO] config_path={cfg.source_path.as_posix()}")

    run_id = f"{now_stamp()}_{uuid.uuid4().hex[:8]}"
    result_dir = AUTO_TEST_DIR / "results" / f"session_5turn_{run_id}"
    non_text_dir = result_dir / "non_text"
    result_dir.mkdir(parents=True, exist_ok=True)
    non_text_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": normalize_authz(cfg.token),
        "uid": cfg.uid,
        "email": cfg.email,
        "Content-Type": "application/json",
    }

    print(f"[INFO] run_id={run_id}")
    print(f"[INFO] result_dir={result_dir.as_posix()}")
    print("[INFO] creating session...")

    turns = build_turns()
    expected_facts = build_expected_facts()
    run_settings = build_run_settings()
    llm_eval_cfg = build_llm_eval_config(cfg)
    user_sim_cfg = build_user_simulator_config(cfg, llm_eval_cfg)
    persist_type = build_persist_type()
    exec_max_turns = build_exec_max_turns()
    create_settings_json = build_create_settings_json(run_settings)
    results: list[dict[str, Any]] = []

    print(f"[INFO] persist_type={persist_type}")
    print(f"[INFO] exec_max_turns={exec_max_turns}")
    print(f"[INFO] run_settings={json.dumps(run_settings, ensure_ascii=False, sort_keys=True)}")
    if create_settings_json:
        print(f"[INFO] create_settings={create_settings_json}")
    print(f"[INFO] user_simulator_enabled={user_sim_cfg.enabled}")
    print(f"[INFO] user_simulator_mode={user_sim_cfg.mode}")
    if user_sim_cfg.enabled:
        print(f"[INFO] user_simulator_url={user_sim_cfg.base_url}")
        print(f"[INFO] user_simulator_model={user_sim_cfg.model}")
        print(f"[INFO] user_simulator_max_turns={user_sim_cfg.max_turns}")
    print(f"[INFO] llm_eval_enabled={llm_eval_cfg.enabled}")
    if llm_eval_cfg.enabled:
        print(f"[INFO] llm_eval_url={llm_eval_cfg.base_url}")
        print(f"[INFO] llm_eval_model={llm_eval_cfg.model}")

    if user_sim_cfg.enabled and (not user_sim_cfg.base_url or not user_sim_cfg.model or not user_sim_cfg.api_key):
        print("[WARN] user_simulator 配置不完整，回退为固定 5 轮脚本。")
        user_sim_cfg = UserSimulatorConfig(
            enabled=False,
            mode=user_sim_cfg.mode,
            base_url=user_sim_cfg.base_url,
            model=user_sim_cfg.model,
            api_key="",
            timeout_sec=user_sim_cfg.timeout_sec,
            max_turns=len(turns),
            first_user_message=user_sim_cfg.first_user_message,
            system_prompt=user_sim_cfg.system_prompt,
            scenario_prompt=user_sim_cfg.scenario_prompt,
        )

    target_turns = user_sim_cfg.max_turns if user_sim_cfg.enabled else len(turns)
    sim_state = UserSimulatorState(previous_response_id="")

    raw_events_path = result_dir / "raw_events.jsonl"
    with raw_events_path.open("w", encoding="utf-8") as raw_fp:
        session_id, create_trace = create_session(
            cfg.base_url,
            headers,
            persist_type=persist_type,
            settings_json=create_settings_json,
        )
        print(f"[INFO] session_id={session_id}")
        print(f"[INFO] create_request_id={create_trace.request_id}")
        print(f"[INFO] create_backend_trace_id={create_trace.backend_trace_id}")

        for idx in range(1, target_turns + 1):
            if idx == 1:
                user_text = user_sim_cfg.first_user_message if user_sim_cfg.enabled else turns[0]
            elif user_sim_cfg.enabled:
                try:
                    user_text, should_stop, sim_state = generate_user_turn_with_simulator(
                        sim_cfg=user_sim_cfg,
                        sim_state=sim_state,
                        results=results,
                        expected_facts=expected_facts,
                        turn_idx=idx,
                    )
                    if should_stop:
                        print(f"[INFO] user_simulator requested stop at turn={idx}")
                        break
                except Exception as exc:
                    print(f"[WARN] user_simulator turn={idx} 生成失败: {exc}")
                    fallback_idx = idx - 1
                    if fallback_idx < len(turns):
                        user_text = turns[fallback_idx]
                        print(f"[WARN] turn={idx} 使用固定脚本回退。")
                    else:
                        print("[WARN] 无可用回退话术，提前结束。")
                        break
            else:
                user_text = turns[idx - 1]

            print(f"[INFO] turn={idx}")
            one = execute_turn(
                base_url=cfg.base_url,
                headers=headers,
                session_id=session_id,
                persist_type=persist_type,
                exec_max_turns=exec_max_turns,
                run_settings=run_settings,
                user_text=user_text,
                turn_idx=idx,
                raw_events_fp=raw_fp,
            )
            results.append(one)
            print(
                "[INFO] turn_trace "
                f"turn={idx} request_id={one['request_id']} backend_trace_id={one['backend_trace_id']}"
            )
            if one.get("run_error"):
                print(f"[WARN] turn={idx} run_error={one['run_error']}")

    write_meta_md(
        result_dir / "run_meta.md",
        cfg,
        session_id,
        run_id,
        user_sim_cfg,
        persist_type,
        exec_max_turns,
        create_settings_json,
        run_settings,
        create_trace,
        results,
    )
    write_dialogue_md(result_dir / "dialogue.md", results)

    (non_text_dir / "turn_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    pass_threshold = _env_float("AUTO_TEST_EVAL_PASS_THRESHOLD", 0.80)
    rule_eval = evaluate_rules(results, expected_facts=expected_facts, pass_threshold=pass_threshold)
    llm_eval = evaluate_with_llm(
        results,
        expected_facts=expected_facts,
        prompts_dir=AUTO_TEST_DIR / "prompts",
        llm_cfg=llm_eval_cfg,
    )

    (non_text_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "rule_evaluation": rule_eval,
                "llm_evaluation": llm_eval,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_evaluation_md(result_dir / "evaluation.md", rule_eval=rule_eval, llm_eval=llm_eval)

    (non_text_dir / "README.md").write_text(
        "# non_text 结构说明\n\n"
        "- `turn_results.json`: 每轮结构化结果。\n"
        "- `evaluation.json`: 规则评估与可选LLM评估。\n",
        encoding="utf-8",
    )

    ok_turns = sum(1 for r in results if r.get("run_end") and not r.get("run_error"))
    print(
        "[INFO] evaluation "
        f"overall_pass={rule_eval.get('overall_pass')} score={rule_eval.get('overall_score')}"
    )
    print(f"[INFO] done: success_turns={ok_turns}/{len(results)}")
    print(f"[INFO] outputs: {result_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
