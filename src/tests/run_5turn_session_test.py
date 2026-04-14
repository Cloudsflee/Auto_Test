from __future__ import annotations

import argparse
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
from urllib.parse import urlparse, urlunparse

import requests

# 允许直接运行本文件时从 src/ 下导入评估模块。
AUTO_TEST_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = AUTO_TEST_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.dialogue_evaluator import LLMEvalConfig, evaluate_with_llm, write_evaluation_md
from tests.user_simulator_engine import (
    GeneratedRole,
    UserSimulatorConfig,
    UserSimulatorState,
    generate_role_for_session,
    generate_user_turn_with_simulator,
    normalize_capability_mode,
)
from tests.workspace_pipeline import (
    build_workspace_snapshot,
    export_workspace_view,
    extract_workspace_paths,
)


CFG_PATH_CANDIDATES = [
    AUTO_TEST_DIR / "config" / "config.local.json",
    AUTO_TEST_DIR / "config" / "config.json",
    AUTO_TEST_DIR / "test.txt",
    AUTO_TEST_DIR / "test_re.txt",
]


@dataclass
class AuthConfig:
    base_url: str
    dotai_base_url: str
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


REQUIRED_NOTEBOOK_CLEAR_TEXT = "请你把当前记忆文件重置为系统初始模板"
DEFAULT_FIRST_USER_TEXT = REQUIRED_NOTEBOOK_CLEAR_TEXT
PROMPTS_DIR = AUTO_TEST_DIR / "prompts"
DEFAULT_USER_SIM_SYSTEM_PROMPT_FILE = PROMPTS_DIR / "user_simulator_system.prompt"
DEFAULT_USER_SIM_SCENARIO_PROMPT_FILE = PROMPTS_DIR / "user_simulator_scenario.prompt"
DEFAULT_USER_SIM_ROLE_SYSTEM_PROMPT_FILE = PROMPTS_DIR / "user_simulator_role_system.prompt"
DEFAULT_USER_SIM_ROLE_USER_PROMPT_FILE = PROMPTS_DIR / "user_simulator_role_user.prompt"
INTERACT_TOOL_NAMES = {"option_card", "post_submit", "ui_cmd"}
DEFAULT_SIM_CALLBACK_MAX_ROUNDS = 8


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


def _cfg_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return default
        try:
            return float(raw)
        except Exception:
            return default
    return default


def parse_runtime_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run auto dialogue test (LLM-only mode).")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Override user_simulator.max_turns for this run only.",
    )
    return parser.parse_args(argv)


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
    dotai_base_url = str(obj.get("dotai_base_url", "")).strip().rstrip("/")
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
        dotai_base_url=dotai_base_url,
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
    dotai_base_url = _pick_text_value(text, "dotai_base_url").rstrip("/")
    token = _pick_text_value(text, "token")
    uid = _pick_text_value(text, "uid")
    email = _pick_text_value(text, "email")
    if not base_url or not token or not uid or not email:
        raise RuntimeError(f"缺少必填字段: base_url/token/uid/email, path={path.as_posix()}")
    return AuthConfig(
        base_url=base_url,
        dotai_base_url=dotai_base_url,
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


def _derive_dotai_base_url(base_url: str) -> str:
    raw = str(base_url or "").strip().rstrip("/")
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except Exception:
        return ""
    if not parsed.scheme or not parsed.netloc:
        return ""

    netloc = parsed.netloc
    replaced = netloc.replace("ai-backend.", "dotai-backend.", 1)
    if replaced == netloc:
        replaced = netloc.replace("ai-backend", "dotai-backend", 1)
    if replaced == netloc:
        return ""
    return urlunparse((parsed.scheme, replaced, "", "", "", "")).rstrip("/")


def resolve_dotai_base_url(cfg: AuthConfig) -> str:
    env_override = str(os.getenv("AUTO_TEST_DOTAI_BASE_URL", "")).strip().rstrip("/")
    if env_override:
        return env_override
    if cfg.dotai_base_url:
        return cfg.dotai_base_url.rstrip("/")
    return _derive_dotai_base_url(cfg.base_url)


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
    # Keep the first sentence strictly fixed, append custom text on the next line.
    return f"{REQUIRED_NOTEBOOK_CLEAR_TEXT}\n{normalized}"


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
        "随机性要求：\n"
        "1. 每轮随机选择一个意图提问，不固定顺序。\n"
        "2. 意图池：口号创作/卖点追问/预算约束/竞品比较/复述核对/总结要求。\n"
        "3. 语气随机（正式、口语、追问、质疑），长度随机（10~80字）。\n"
        "4. 不要重复上一轮用户句式。\n\n"
        "轮次上限：{{MAX_TURNS}}。仅在你认为目标完成时再 stop=true。"
    )
    default_role_system_prompt = (
        "你是测试数据设计器，专门生成“随机客户角色”。"
        "请只返回 JSON，不要输出解释。"
    )
    default_role_user_prompt = (
        "请生成一个用于多轮对话测试的随机客户角色。\n"
        "要求：\n"
        "1. 角色要有真实业务背景与沟通风格。\n"
        "2. 必须包含冲突约束（例如预算、时效、品牌调性冲突）。\n"
        "3. 尽量避免与常见模板同质化。\n"
        "4. 轮次上限参考：{{MAX_TURNS}}。\n\n"
        "输出 JSON 字段：\n"
        "- role_name\n"
        "- identity\n"
        "- business_background\n"
        "- communication_style\n"
        "- current_goal\n"
        "- constraints\n"
        "- success_criteria\n"
        "- speaking_habits\n"
    )
    system_prompt_path = _resolve_prompt_path(
        str(sim_cfg.get("system_prompt_file") or "").strip(),
        DEFAULT_USER_SIM_SYSTEM_PROMPT_FILE,
    )
    scenario_prompt_path = _resolve_prompt_path(
        str(sim_cfg.get("scenario_prompt_file") or "").strip(),
        DEFAULT_USER_SIM_SCENARIO_PROMPT_FILE,
    )
    role_system_prompt_path = _resolve_prompt_path(
        str(sim_cfg.get("role_system_prompt_file") or "").strip(),
        DEFAULT_USER_SIM_ROLE_SYSTEM_PROMPT_FILE,
    )
    role_user_prompt_path = _resolve_prompt_path(
        str(sim_cfg.get("role_user_prompt_file") or "").strip(),
        DEFAULT_USER_SIM_ROLE_USER_PROMPT_FILE,
    )
    system_prompt = str(sim_cfg.get("system_prompt") or "").strip() or _load_prompt_template(
        system_prompt_path,
        default_system_prompt,
    )
    scenario_prompt = str(sim_cfg.get("scenario_prompt") or "").strip() or _load_prompt_template(
        scenario_prompt_path,
        default_scenario_prompt,
    )
    role_system_prompt = str(sim_cfg.get("role_system_prompt") or "").strip() or _load_prompt_template(
        role_system_prompt_path,
        default_role_system_prompt,
    )
    role_user_prompt = str(sim_cfg.get("role_user_prompt") or "").strip() or _load_prompt_template(
        role_user_prompt_path,
        default_role_user_prompt,
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
        max_turns=max(
            1,
            min(
                128,
                _cfg_int(
                    sim_cfg.get("max_turns"),
                    _env_int("AUTO_TEST_MAX_TURNS", _env_int("AUTO_TEST_USER_SIM_MAX_TURNS", 5)),
                ),
            ),
        ),
        first_user_message=first_user_message,
        system_prompt=system_prompt,
        scenario_prompt=scenario_prompt,
        user_temperature=max(0.0, min(2.0, _cfg_float(sim_cfg.get("user_temperature"), 0.9))),
        role_temperature=max(0.0, min(2.0, _cfg_float(sim_cfg.get("role_temperature"), 1.0))),
        role_system_prompt=role_system_prompt,
        role_user_prompt=role_user_prompt,
        capability_mode=normalize_capability_mode(
            sim_cfg.get("capability_mode")
            or os.getenv("AUTO_TEST_USER_SIM_CAPABILITY_MODE", "alternating")
        ),
    )


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


def _extract_interact_tool_call(event: dict[str, Any]) -> dict[str, Any] | None:
    if str(event.get("type") or "") != "TOOL_CALL":
        return None
    tool_name = str(event.get("toolCallName") or event.get("tool_call_name") or "").strip()
    tool_call_id = str(event.get("toolCallId") or event.get("tool_call_id") or "").strip()
    if not tool_name or not tool_call_id:
        return None
    if tool_name not in INTERACT_TOOL_NAMES:
        return None

    args_raw = event.get("args")
    args: Any = {}
    if isinstance(args_raw, str):
        parsed = _safe_json_loads(args_raw.strip()) if args_raw.strip() else None
        args = parsed if parsed is not None else args_raw
    elif args_raw is not None:
        args = args_raw

    return {"name": tool_name, "tool_call_id": tool_call_id, "args": args}


def _build_simulated_interact_output(tool_name: str, args: Any) -> str:
    arg_dict = args if isinstance(args, dict) else {}

    if tool_name == "option_card":
        options = arg_dict.get("options")
        options = options if isinstance(options, list) else []
        has_options = len(options) > 0
        payload = {
            "action": "select" if has_options else "no_select",
            "selectedIds": [1] if has_options else [],
            "selectedOptions": [str(options[0])] if has_options else [],
            "note": "auto_test simulated callback",
        }
        return json.dumps(payload, ensure_ascii=False)

    if tool_name == "post_submit":
        image_paths = arg_dict.get("imagePaths")
        if not isinstance(image_paths, list):
            image_paths = []
        channels = arg_dict.get("channels")
        if not isinstance(channels, list):
            channels = []
        payload = {
            "action": "confirm",
            "title": str(arg_dict.get("title") or ""),
            "content": str(arg_dict.get("content") or ""),
            "imagePaths": image_paths,
            "channels": channels,
            "note": "auto_test simulated callback",
        }
        return json.dumps(payload, ensure_ascii=False)

    if tool_name == "ui_cmd":
        payload = {
            "ok": True,
            "command": str(arg_dict.get("command") or ""),
            "message": "auto_test simulated callback executed",
        }
        return json.dumps(payload, ensure_ascii=False)

    return json.dumps({"ok": True, "note": "auto_test simulated callback"}, ensure_ascii=False)


def _build_simulated_tool_response_message(tool_call: dict[str, Any]) -> dict[str, Any]:
    tool_name = str(tool_call.get("name") or "")
    tool_call_id = str(tool_call.get("tool_call_id") or "")
    args = tool_call.get("args")
    output = _build_simulated_interact_output(tool_name, args)
    return {
        "role": "tool",
        "content": [
            {
                "toolResponse": {
                    "name": tool_name,
                    "ref": tool_call_id,
                    "output": output,
                }
            }
        ],
    }


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
    started = time.time()
    overall_timeout_sec = max(60, min(900, _env_int("AUTO_TEST_TURN_TIMEOUT_SEC", 240)))
    simulate_interact_callback = _env_bool("AUTO_TEST_SIMULATE_INTERACT_CALLBACK", True)
    max_sim_callback_rounds = max(
        0,
        min(
            32,
            _env_int("AUTO_TEST_MAX_SIM_CALLBACK_ROUNDS", DEFAULT_SIM_CALLBACK_MAX_ROUNDS),
        ),
    )

    pending_messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": user_text}]}]
    event_count = 0
    run_end = False
    run_error = ""
    assistant_final = ""
    assistant_deltas: list[str] = []
    callback_rounds = 0
    request_ids: list[str] = []
    backend_trace_ids: list[str] = []
    traceparents: list[str] = []
    tracestates: list[str] = []
    x_trace_ids: list[str] = []
    first_trace: TraceInfo | None = None
    last_trace: TraceInfo | None = None
    event_workspace_paths: set[str] = set()

    while True:
        req_body = {
            "sessionId": session_id,
            "persistType": persist_type,
            "runSettings": run_settings,
            "messages": pending_messages,
            "exec": {"maxTurns": exec_max_turns},
        }
        resp = requests.post(url, headers=headers, json=req_body, stream=True, timeout=180)
        trace = trace_info_from_response(resp)
        if first_trace is None:
            first_trace = trace
        last_trace = trace
        request_ids.append(trace.request_id)
        backend_trace_ids.append(trace.backend_trace_id)
        traceparents.append(trace.traceparent)
        tracestates.append(trace.tracestate)
        x_trace_ids.append(trace.x_trace_id)
        resp.raise_for_status()

        run_end = False
        pending_tool_call: dict[str, Any] | None = None
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
            if event_type not in {"TOOL_CALL_DELTA", "TEXT_DELTA"}:
                event_workspace_paths.update(extract_workspace_paths(json.dumps(event, ensure_ascii=False)))

            raw_events_fp.write(
                json.dumps(
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "turn": turn_idx,
                        "callback_round": callback_rounds,
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
            elif event_type == "TOOL_CALL":
                maybe_call = _extract_interact_tool_call(event)
                if maybe_call is not None:
                    pending_tool_call = maybe_call
            elif event_type == "RUN_ERROR":
                run_error = str(event.get("message") or "RUN_ERROR")
                break
            elif event_type == "RUN_END":
                run_end = True
                break

            if time.time() - started > overall_timeout_sec:
                run_error = f"TURN_TIMEOUT_{overall_timeout_sec}s"
                break

        resp.close()

        if run_error:
            break

        if pending_tool_call is not None:
            if not simulate_interact_callback:
                run_error = (
                    f"interact tool callback required: {pending_tool_call['name']} "
                    "but AUTO_TEST_SIMULATE_INTERACT_CALLBACK=false"
                )
                break
            if callback_rounds >= max_sim_callback_rounds:
                run_error = (
                    f"simulated callback rounds exceeded limit={max_sim_callback_rounds} "
                    f"(last_tool={pending_tool_call['name']})"
                )
                break

            callback_rounds += 1
            tool_message = _build_simulated_tool_response_message(pending_tool_call)
            raw_events_fp.write(
                json.dumps(
                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "turn": turn_idx,
                        "callback_round": callback_rounds,
                        "request_id": trace.request_id,
                        "traceparent": trace.traceparent,
                        "backend_trace_id": trace.backend_trace_id,
                        "session_id": session_id,
                        "event_type": "SIMULATED_TOOL_RESPONSE",
                        "event_role": "tool",
                        "event_seq": None,
                        "event_sub_seq": None,
                        "event_raw": tool_message,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            pending_messages = [tool_message]
            event_workspace_paths.update(extract_workspace_paths(json.dumps(tool_message, ensure_ascii=False)))
            continue

        if run_end:
            break

        if time.time() - started > overall_timeout_sec:
            run_error = f"TURN_TIMEOUT_{overall_timeout_sec}s"
            break

        run_error = "STREAM_CLOSED_BEFORE_RUN_END"
        break

    if not assistant_final and assistant_deltas:
        assistant_final = "".join(assistant_deltas)

    workspace_snapshot = build_workspace_snapshot(
        base_url=base_url,
        headers=headers,
        session_id=session_id,
        event_paths=sorted(event_workspace_paths),
    )

    primary_trace = first_trace or TraceInfo("", "", "", "", "")
    final_trace = last_trace or primary_trace
    duration_sec = round(time.time() - started, 2)
    return {
        "turn": turn_idx,
        "request_id": primary_trace.request_id,
        "traceparent": primary_trace.traceparent,
        "tracestate": primary_trace.tracestate,
        "x_trace_id": primary_trace.x_trace_id,
        "backend_trace_id": primary_trace.backend_trace_id,
        "request_ids": request_ids,
        "backend_trace_ids": backend_trace_ids,
        "traceparents": traceparents,
        "tracestates": tracestates,
        "x_trace_ids": x_trace_ids,
        "last_request_id": final_trace.request_id,
        "last_backend_trace_id": final_trace.backend_trace_id,
        "user_text": user_text,
        "assistant_text": assistant_final,
        "event_count": event_count,
        "run_end": run_end,
        "run_error": run_error,
        "callback_rounds": callback_rounds,
        "simulate_interact_callback": simulate_interact_callback,
        "max_sim_callback_rounds": max_sim_callback_rounds,
        "turn_timeout_sec": overall_timeout_sec,
        "workspace_event_paths": sorted(event_workspace_paths)[:200],
        "workspace_snapshot": workspace_snapshot,
        "duration_sec": duration_sec,
    }


def build_run_settings() -> dict[str, Any]:
    import os

    # Default tool set for memory/compression auto tests:
    # keep core tools + option_card, but keep publish/UI command tools off.
    default_tools: dict[str, Any] = {
        "read_text": None,
        "ls": None,
        "edit": None,
        "write": None,
        "time_now": None,
        "image_chat": None,
        "image_gen_edit": None,
        "web_search": None,
        "web_crawler": None,
        "option_card": None,
    }
    # Compatibility switch: when enabled, turn on both post_submit and ui_cmd.
    enable_advanced_interact = _env_bool("AUTO_TEST_ENABLE_INTERACT_TOOLS", False)
    if _env_bool("AUTO_TEST_ENABLE_POST_SUBMIT_TOOL", enable_advanced_interact):
        default_tools["post_submit"] = None
    if _env_bool("AUTO_TEST_ENABLE_UI_CMD_TOOL", enable_advanced_interact):
        default_tools["ui_cmd"] = None

    if _env_bool("AUTO_TEST_DISABLE_INTERACT_TOOLS", False):
        default_tools.pop("option_card", None)
        default_tools.pop("post_submit", None)
        default_tools.pop("ui_cmd", None)

    default_settings: dict[str, Any] = {"tools": default_tools}

    raw = os.getenv("AUTO_TEST_RUN_SETTINGS_JSON", "").strip()
    if not raw:
        return default_settings

    custom = _safe_json_loads(raw)
    if not isinstance(custom, dict):
        raise RuntimeError("AUTO_TEST_RUN_SETTINGS_JSON ?????JSON ???")
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
    # LLM-only 模式下不再绑定固定事实集合。
    return {}


def write_meta_md(
    path: Path,
    cfg: AuthConfig,
    session_id: str,
    run_id: str,
    user_sim_cfg: UserSimulatorConfig,
    generated_role: GeneratedRole,
    persist_type: int,
    exec_max_turns: int,
    create_settings_json: str | None,
    run_settings: dict[str, Any],
    create_trace: TraceInfo,
    results: list[dict[str, Any]],
    workspace_export: dict[str, Any] | None = None,
) -> None:
    ok_turns = sum(1 for r in results if r.get("run_end") and not r.get("run_error"))
    lines = [
        "# 对话测试元信息",
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
        f"- user_simulator_capability_mode: `{user_sim_cfg.capability_mode}`",
        f"- user_simulator_model: `{user_sim_cfg.model}`",
        f"- user_simulator_base_url: `{user_sim_cfg.base_url}`",
        f"- user_simulator_user_temperature: `{user_sim_cfg.user_temperature}`",
        f"- user_simulator_role_temperature: `{user_sim_cfg.role_temperature}`",
        f"- generated_role_name: `{generated_role.role_name}`",
        f"- generated_role_json: `{json.dumps(generated_role.role_json, ensure_ascii=False, sort_keys=True)}`",
        "- generated_role_profile:",
        "",
        generated_role.role_profile,
        "",
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

    if isinstance(workspace_export, dict):
        counts = workspace_export.get("counts")
        if not isinstance(counts, dict):
            counts = {}
        lines.extend(
            [
                "",
                "## Workspace Export",
                "",
                f"- workspace_dir: `{(path.parent / 'workspace').as_posix()}`",
                f"- all_paths: `{counts.get('all_paths', 0)}`",
                f"- md_paths: `{counts.get('md_paths', 0)}`",
                f"- image_paths: `{counts.get('image_paths', 0)}`",
                f"- exported_text_files: `{counts.get('exported_text_files', 0)}`",
                f"- exported_image_files: `{counts.get('exported_image_files', 0)}`",
                f"- unresolved_files: `{counts.get('unresolved_files', 0)}`",
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
    args = parse_runtime_args(sys.argv[1:])
    cfg_path = resolve_config_path()
    cfg = load_config(cfg_path)
    dotai_base_url = resolve_dotai_base_url(cfg)
    print(f"[INFO] config_path={cfg.source_path.as_posix()}")

    run_id = f"{now_stamp()}_{uuid.uuid4().hex[:8]}"
    result_dir = AUTO_TEST_DIR / "results" / f"session_autotest_{run_id}"
    run_data_dir = result_dir / "run_data"

    headers = {
        "Authorization": normalize_authz(cfg.token),
        "uid": cfg.uid,
        "email": cfg.email,
        "Content-Type": "application/json",
    }

    print(f"[INFO] run_id={run_id}")
    print(f"[INFO] result_dir={result_dir.as_posix()}")
    print(f"[INFO] dotai_base_url={dotai_base_url or '(disabled)'}")
    print("[INFO] creating session...")

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
    print(f"[INFO] turn_timeout_sec={max(60, min(900, _env_int('AUTO_TEST_TURN_TIMEOUT_SEC', 240)))}")
    print(f"[INFO] run_settings={json.dumps(run_settings, ensure_ascii=False, sort_keys=True)}")
    if create_settings_json:
        print(f"[INFO] create_settings={create_settings_json}")
    print(f"[INFO] user_simulator_enabled={user_sim_cfg.enabled}")
    print(f"[INFO] user_simulator_mode={user_sim_cfg.mode}")
    print(f"[INFO] user_simulator_capability_mode={user_sim_cfg.capability_mode}")
    if user_sim_cfg.enabled:
        print(f"[INFO] user_simulator_url={user_sim_cfg.base_url}")
        print(f"[INFO] user_simulator_model={user_sim_cfg.model}")
        print(f"[INFO] user_simulator_max_turns={user_sim_cfg.max_turns}")
        print(f"[INFO] user_simulator_user_temperature={user_sim_cfg.user_temperature}")
        print(f"[INFO] user_simulator_role_temperature={user_sim_cfg.role_temperature}")
    print(f"[INFO] llm_eval_enabled={llm_eval_cfg.enabled}")
    if llm_eval_cfg.enabled:
        print(f"[INFO] llm_eval_url={llm_eval_cfg.base_url}")
        print(f"[INFO] llm_eval_model={llm_eval_cfg.model}")

    if args.max_turns > 0:
        user_sim_cfg.max_turns = max(1, min(128, args.max_turns))
        print(f"[INFO] override max_turns from CLI: {user_sim_cfg.max_turns}")

    if not user_sim_cfg.enabled:
        raise RuntimeError("已启用 LLM 全量模式：请在 config.user_simulator.enabled=true 后重试。")
    if not user_sim_cfg.base_url or not user_sim_cfg.model or not user_sim_cfg.api_key:
        raise RuntimeError("user_simulator 配置不完整：需要 base_url/model/api_key。")
    if not llm_eval_cfg.enabled:
        raise RuntimeError("已启用 LLM 全量模式：请在 config.llm_eval.enabled=true 后重试。")
    if not llm_eval_cfg.base_url or not llm_eval_cfg.model or not llm_eval_cfg.api_key:
        raise RuntimeError("llm_eval 配置不完整：需要 base_url/model/api_key。")

    result_dir.mkdir(parents=True, exist_ok=True)
    run_data_dir.mkdir(parents=True, exist_ok=True)

    target_turns = user_sim_cfg.max_turns
    # Generate role in an isolated call, then start a fresh simulator dialogue context.
    generated_role = generate_role_for_session(user_sim_cfg)
    sim_state = UserSimulatorState(previous_response_id="")
    print(f"[INFO] generated_role_name={generated_role.role_name}")

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
        latest_workspace_snapshot: dict[str, Any] | None = None

        for idx in range(1, target_turns + 1):
            if idx == 1:
                user_text = user_sim_cfg.first_user_message
            else:
                try:
                    user_text, should_stop, sim_state = generate_user_turn_with_simulator(
                        sim_cfg=user_sim_cfg,
                        sim_state=sim_state,
                        results=results,
                        role=generated_role,
                        turn_idx=idx,
                        workspace_snapshot=latest_workspace_snapshot,
                    )
                    if should_stop:
                        print(f"[INFO] user_simulator requested stop at turn={idx}")
                        break
                except Exception as exc:
                    raise RuntimeError(f"user_simulator turn={idx} 生成失败: {exc}") from exc

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
            snapshot = one.get("workspace_snapshot")
            latest_workspace_snapshot = snapshot if isinstance(snapshot, dict) else None
            print(
                "[INFO] turn_trace "
                f"turn={idx} request_id={one['request_id']} backend_trace_id={one['backend_trace_id']}"
            )
            if isinstance(snapshot, dict):
                counts = snapshot.get("counts")
                if isinstance(counts, dict):
                    print(
                        "[INFO] workspace_snapshot "
                        f"turn={idx} source={snapshot.get('source')} "
                        f"paths={counts.get('all_paths', 0)} "
                        f"md={counts.get('md_files', 0)} "
                        f"image={counts.get('image_files', 0)}"
                    )
            if one.get("run_error"):
                print(f"[WARN] turn={idx} run_error={one['run_error']}")

    workspace_export = export_workspace_view(
        base_url=cfg.base_url,
        headers=headers,
        session_id=session_id,
        raw_events_path=raw_events_path,
        result_dir=result_dir,
        dotai_base_url=dotai_base_url,
        user_id=cfg.uid,
    )
    workspace_counts = workspace_export.get("counts")
    if isinstance(workspace_counts, dict):
        print(
            "[INFO] workspace_export "
            f"paths={workspace_counts.get('all_paths', 0)} "
            f"md={workspace_counts.get('md_paths', 0)} "
            f"image={workspace_counts.get('image_paths', 0)} "
            f"exported_text={workspace_counts.get('exported_text_files', 0)} "
            f"exported_image={workspace_counts.get('exported_image_files', 0)} "
            f"unresolved={workspace_counts.get('unresolved_files', 0)}"
        )

    write_meta_md(
        result_dir / "run_meta.md",
        cfg,
        session_id,
        run_id,
        user_sim_cfg,
        generated_role,
        persist_type,
        exec_max_turns,
        create_settings_json,
        run_settings,
        create_trace,
        results,
        workspace_export=workspace_export,
    )
    write_dialogue_md(result_dir / "dialogue.md", results)

    (run_data_dir / "turn_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    llm_eval = evaluate_with_llm(
        results,
        expected_facts=expected_facts,
        prompts_dir=AUTO_TEST_DIR / "prompts",
        llm_cfg=llm_eval_cfg,
    )
    if llm_eval.get("skipped") or llm_eval.get("error"):
        raise RuntimeError(f"LLM 评估失败: {llm_eval.get('reason') or llm_eval.get('error')}")

    (run_data_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "evaluation_mode": "llm_only",
                "llm_evaluation": llm_eval,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_evaluation_md(result_dir / "evaluation.md", rule_eval=None, llm_eval=llm_eval)

    (run_data_dir / "README.md").write_text(
        "# run_data 结构说明\n\n"
        "- `turn_results.json`: 每轮结构化结果。\n"
        "- `evaluation.json`: LLM-only 评估结果。\n"
        "- `../workspace/`: 用户可见工作区导出（含 `_manifest.json` 与文件落盘结果）。\n",
        encoding="utf-8",
    )

    ok_turns = sum(1 for r in results if r.get("run_end") and not r.get("run_error"))
    llm_summary = ""
    response_json = llm_eval.get("response_json")
    if isinstance(response_json, dict):
        llm_summary = (
            f"pass={response_json.get('pass')} "
            f"score_0_100={response_json.get('score_0_100')}"
        )
    print(f"[INFO] evaluation llm_only {llm_summary}".strip())
    print(f"[INFO] done: success_turns={ok_turns}/{len(results)}")
    print(f"[INFO] outputs: {result_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
