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

# 鍏佽鐩存帴杩愯鏈枃浠舵椂浠?src/ 涓嬪鍏ヨ瘎浼版ā鍧椼€?
AUTO_TEST_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = AUTO_TEST_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.dialogue_evaluator import LLMEvalConfig, evaluate_with_llm, write_evaluation_md
from probe import ProbeLLMJudgeConfig, evaluate_probes, write_probe_evaluation_md
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
    proxy_http: str
    proxy_https: str
    proxy_no_proxy: str
    token: str
    uid: str
    email: str
    source_path: Path
    selected_env: str
    llm_eval: dict[str, Any]
    user_simulator: dict[str, Any]
    probe_eval: dict[str, Any]


@dataclass
class TraceInfo:
    request_id: str
    traceparent: str
    tracestate: str
    x_trace_id: str
    backend_trace_id: str


@dataclass
class ProbeEvalConfig:
    enabled: bool
    dataset_path: Path
    fail_on_error: bool
    max_fail_details: int
    llm_judge: ProbeLLMJudgeConfig
    deterministic_weight: float
    llm_weight: float


REQUIRED_NOTEBOOK_CLEAR_TEXT = "璇蜂綘鎶婂綋鍓嶈蹇嗘枃浠堕噸缃负绯荤粺鍒濆妯℃澘"
DEFAULT_FIRST_USER_TEXT = REQUIRED_NOTEBOOK_CLEAR_TEXT
PROMPTS_DIR = AUTO_TEST_DIR / "prompts"
FRAMEWORK_PROMPTS_DIR = PROMPTS_DIR / "framework"
TARGET_PROMPTS_DIR = PROMPTS_DIR / "targets"
DEFAULT_TARGET_NAME = "advoo"
DEFAULT_USER_SIM_SYSTEM_PROMPT_FILE = FRAMEWORK_PROMPTS_DIR / "simulator" / "system" / "user_simulator_system.prompt"
DEFAULT_USER_SIM_SCENARIO_PROMPT_FILE = FRAMEWORK_PROMPTS_DIR / "simulator" / "system" / "scenario_template.prompt"
DEFAULT_USER_SIM_ROLE_SYSTEM_PROMPT_FILE = FRAMEWORK_PROMPTS_DIR / "simulator" / "role_generation" / "role_system.prompt"
DEFAULT_USER_SIM_ROLE_USER_PROMPT_FILE = FRAMEWORK_PROMPTS_DIR / "simulator" / "role_generation" / "role_user.prompt"
DEFAULT_PROBE_JUDGE_SYSTEM_PROMPT_FILE = FRAMEWORK_PROMPTS_DIR / "evaluator" / "probe" / "probe_judge_system.prompt"
DEFAULT_PROBE_JUDGE_USER_PROMPT_FILE = FRAMEWORK_PROMPTS_DIR / "evaluator" / "probe" / "probe_judge_user.prompt"
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


def _load_optional_text(path: Path, fallback: str = "") -> str:
    if not path.exists():
        return fallback
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return text or fallback


def _normalize_target_name(raw: Any) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]", "", str(raw or "").strip().lower())
    return name or DEFAULT_TARGET_NAME


def _resolve_target_dir(target_name: str) -> Path:
    return TARGET_PROMPTS_DIR / _normalize_target_name(target_name)


def _render_prompt_vars(template: str, vars_map: dict[str, str]) -> str:
    out = str(template or "")
    for key, value in vars_map.items():
        out = out.replace(f"{{{{{key}}}}}", str(value))
    return out


def _trim_prompt_text(text: str, max_chars: int = 6000) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 20] + "\n...(truncated)..."


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
    parser.add_argument(
        "--env",
        type=str,
        default="",
        help="Target runtime environment (prod/test). Default is prod.",
    )
    return parser.parse_args(argv)


def resolve_config_path() -> Path:
    for path in CFG_PATH_CANDIDATES:
        if path.exists():
            return path
    return CFG_PATH_CANDIDATES[0]


def _normalize_env_name(raw: Any, default: str = "prod") -> str:
    text = str(raw or "").strip().lower()
    if text in {"prod", "production"}:
        return "prod"
    if text in {"test", "testing", "staging", "stage"}:
        return "test"
    return default


def _merge_dict(primary: Any, fallback: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(fallback, dict):
        out.update(fallback)
    if isinstance(primary, dict):
        out.update(primary)
    return out


def _resolve_active_env_name(obj: dict[str, Any], runtime_env: str) -> str:
    if runtime_env:
        return _normalize_env_name(runtime_env, "prod")
    env_from_var = str(os.getenv("AUTO_TEST_ENV", "")).strip()
    if env_from_var:
        return _normalize_env_name(env_from_var, "prod")
    env_from_cfg = str(obj.get("active_env") or "").strip()
    if env_from_cfg:
        return _normalize_env_name(env_from_cfg, "prod")
    return "prod"


def _load_json_config(path: Path, runtime_env: str) -> AuthConfig:
    obj = _safe_json_loads(path.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"閰嶇疆鏂囦欢涓嶆槸鍚堟硶 JSON: {path.as_posix()}")

    active_env = _resolve_active_env_name(obj, runtime_env)
    environments = obj.get("environments")
    env_obj: dict[str, Any] = {}
    if isinstance(environments, dict) and environments:
        chosen = environments.get(active_env)
        if not isinstance(chosen, dict):
            raise RuntimeError(
                f"environment '{active_env}' not found in config.environments, path={path.as_posix()}"
            )
        env_obj = chosen

    auth = _merge_dict(env_obj.get("auth"), obj.get("auth"))
    base_url = str(env_obj.get("base_url") or obj.get("base_url") or "").strip().rstrip("/")
    dotai_base_url = str(env_obj.get("dotai_base_url") or obj.get("dotai_base_url") or "").strip().rstrip("/")
    proxy = _merge_dict(env_obj.get("proxy"), obj.get("proxy"))
    proxy_http = str(
        proxy.get("http")
        or proxy.get("http_proxy")
        or env_obj.get("http_proxy")
        or obj.get("http_proxy")
        or ""
    ).strip()
    proxy_https = str(
        proxy.get("https")
        or proxy.get("https_proxy")
        or env_obj.get("https_proxy")
        or obj.get("https_proxy")
        or ""
    ).strip()
    proxy_no_proxy = str(
        proxy.get("no_proxy")
        or env_obj.get("no_proxy")
        or obj.get("no_proxy")
        or ""
    ).strip()
    token = str(auth.get("token", "") or env_obj.get("token", "") or obj.get("token", "")).strip()
    uid = str(auth.get("uid", "") or env_obj.get("uid", "") or obj.get("uid", "")).strip()
    email = str(auth.get("email", "") or env_obj.get("email", "") or obj.get("email", "")).strip()
    llm_eval = obj.get("llm_eval")
    if not isinstance(llm_eval, dict):
        llm_eval = {}
    user_simulator = obj.get("user_simulator")
    if not isinstance(user_simulator, dict):
        user_simulator = {}
    probe_eval = obj.get("probe_eval")
    if not isinstance(probe_eval, dict):
        probe_eval = {}

    if not base_url or not token or not uid or not email:
        raise RuntimeError(f"缂哄皯蹇呭～瀛楁: base_url/token/uid/email, path={path.as_posix()}")

    return AuthConfig(
        base_url=base_url,
        dotai_base_url=dotai_base_url,
        proxy_http=proxy_http,
        proxy_https=proxy_https,
        proxy_no_proxy=proxy_no_proxy,
        token=token,
        uid=uid,
        email=email,
        source_path=path,
        selected_env=active_env,
        llm_eval=llm_eval,
        user_simulator=user_simulator,
        probe_eval=probe_eval,
    )


def _load_text_config(path: Path, runtime_env: str) -> AuthConfig:
    text = path.read_text(encoding="utf-8", errors="ignore")
    active_env = _normalize_env_name(runtime_env or os.getenv("AUTO_TEST_ENV", "") or "prod", "prod")
    base_url = _pick_text_value(text, "base_url").rstrip("/")
    dotai_base_url = _pick_text_value(text, "dotai_base_url").rstrip("/")
    proxy_http = _pick_text_value(text, "http_proxy")
    proxy_https = _pick_text_value(text, "https_proxy")
    proxy_no_proxy = _pick_text_value(text, "no_proxy")
    token = _pick_text_value(text, "token")
    uid = _pick_text_value(text, "uid")
    email = _pick_text_value(text, "email")
    if not base_url or not token or not uid or not email:
        raise RuntimeError(f"缂哄皯蹇呭～瀛楁: base_url/token/uid/email, path={path.as_posix()}")
    return AuthConfig(
        base_url=base_url,
        dotai_base_url=dotai_base_url,
        proxy_http=proxy_http,
        proxy_https=proxy_https,
        proxy_no_proxy=proxy_no_proxy,
        token=token,
        uid=uid,
        email=email,
        source_path=path,
        selected_env=active_env,
        llm_eval={},
        user_simulator={},
        probe_eval={},
    )


def load_config(path: Path, runtime_env: str) -> AuthConfig:
    if not path.exists():
        raise RuntimeError(f"閰嶇疆鏂囦欢涓嶅瓨鍦? {path.as_posix()}")
    if path.suffix.lower() == ".json":
        return _load_json_config(path, runtime_env)
    return _load_text_config(path, runtime_env)


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


def apply_proxy_from_config(cfg: AuthConfig) -> dict[str, str]:
    # Env vars keep higher priority; config acts as default fallback.
    applied: dict[str, str] = {}
    http_proxy = str(cfg.proxy_http or "").strip()
    https_proxy = str(cfg.proxy_https or "").strip()
    no_proxy = str(cfg.proxy_no_proxy or "").strip()

    if http_proxy and not os.getenv("HTTP_PROXY"):
        os.environ["HTTP_PROXY"] = http_proxy
        applied["HTTP_PROXY"] = http_proxy
    if http_proxy and not os.getenv("http_proxy"):
        os.environ["http_proxy"] = http_proxy
        applied["http_proxy"] = http_proxy

    if https_proxy and not os.getenv("HTTPS_PROXY"):
        os.environ["HTTPS_PROXY"] = https_proxy
        applied["HTTPS_PROXY"] = https_proxy
    if https_proxy and not os.getenv("https_proxy"):
        os.environ["https_proxy"] = https_proxy
        applied["https_proxy"] = https_proxy

    if no_proxy and not os.getenv("NO_PROXY"):
        os.environ["NO_PROXY"] = no_proxy
        applied["NO_PROXY"] = no_proxy
    if no_proxy and not os.getenv("no_proxy"):
        os.environ["no_proxy"] = no_proxy
        applied["no_proxy"] = no_proxy

    return applied


def mask_token(token: str) -> str:
    raw = token.strip()
    if raw.lower().startswith("bearer "):
        raw = raw.split(" ", 1)[1].strip()
    if len(raw) <= 16:
        return "*" * len(raw)
    return f"{raw[:8]}...{raw[-8:]}"


def build_llm_eval_config(cfg: AuthConfig) -> LLMEvalConfig:
    llm_cfg = cfg.llm_eval if isinstance(cfg.llm_eval, dict) else {}
    foundation_cfg = llm_cfg.get("foundation")
    if not isinstance(foundation_cfg, dict):
        foundation_cfg = {}
    profile_cfg = llm_cfg.get("profile")
    if not isinstance(profile_cfg, dict):
        profile_cfg = {}
    profiles_cfg = llm_cfg.get("profiles")
    if not isinstance(profiles_cfg, dict):
        profiles_cfg = {}

    active_profile_name = str(
        profile_cfg.get("active")
        or os.getenv("AUTO_TEST_LLM_EVAL_PROFILE_ACTIVE", "memory_compression")
    ).strip() or "memory_compression"
    primary_mode = str(
        llm_cfg.get("primary_mode")
        or os.getenv("AUTO_TEST_LLM_EVAL_PRIMARY_MODE", "llm_v1")
    ).strip().lower()
    if primary_mode not in {"llm_v1", "foundation_v2", "final_v2"}:
        primary_mode = "llm_v1"
    active_profile_cfg = profiles_cfg.get(active_profile_name)
    if not isinstance(active_profile_cfg, dict):
        active_profile_cfg = {}

    def _coerce_weight_map(raw: Any) -> dict[str, float]:
        if not isinstance(raw, dict):
            return {}
        out: dict[str, float] = {}
        for key, value in raw.items():
            name = str(key or "").strip()
            if not name:
                continue
            out[name] = max(0.0, _cfg_float(value, 0.0))
        return out

    def _coerce_profile_name_list(raw: Any) -> list[str]:
        names: list[str] = []
        if isinstance(raw, str):
            parts = raw.split(",")
            for part in parts:
                name = str(part or "").strip()
                if name:
                    names.append(name)
        elif isinstance(raw, (list, tuple)):
            for item in raw:
                name = str(item or "").strip()
                if name:
                    names.append(name)
        seen: set[str] = set()
        normalized: list[str] = []
        for name in names:
            lower = name.lower()
            if lower in seen:
                continue
            seen.add(lower)
            normalized.append(name)
        return normalized

    def _coerce_profile_route_map(raw: Any) -> dict[str, list[str]]:
        if not isinstance(raw, dict):
            return {}
        out: dict[str, list[str]] = {}
        for key, value in raw.items():
            mode = str(key or "").strip().lower()
            if not mode:
                continue
            names = _coerce_profile_name_list(value)
            if names:
                out[mode] = names
        return out

    profile_enabled_global = _cfg_bool(
        profile_cfg.get("enabled"),
        _env_bool("AUTO_TEST_LLM_EVAL_PROFILE_ENABLED", True),
    )
    active_profile_enabled = _cfg_bool(
        active_profile_cfg.get("enabled"),
        _env_bool(
            "AUTO_TEST_LLM_EVAL_PROFILE_ENABLED",
            profile_enabled_global,
        ),
    )
    active_profiles = _coerce_profile_name_list(profile_cfg.get("active_profiles"))
    env_active_profiles = str(os.getenv("AUTO_TEST_LLM_EVAL_PROFILE_ACTIVE_PROFILES", "")).strip()
    if env_active_profiles:
        active_profiles = _coerce_profile_name_list(env_active_profiles)
    if not active_profiles:
        active_profiles = [active_profile_name]

    route_map = _coerce_profile_route_map(profile_cfg.get("active_profiles_by_capability_mode"))
    env_route_map = str(os.getenv("AUTO_TEST_LLM_EVAL_PROFILE_ROUTE_MAP_JSON", "")).strip()
    if env_route_map:
        parsed = _safe_json_loads(env_route_map)
        if isinstance(parsed, dict):
            route_map = _coerce_profile_route_map(parsed)

    profile_weights_by_name: dict[str, dict[str, float]] = {}
    profile_enabled_by_name: dict[str, bool] = {}
    profile_merge_weights: dict[str, float] = {}
    for name, raw in profiles_cfg.items():
        profile_name = str(name or "").strip()
        if not profile_name:
            continue
        profile_raw = raw if isinstance(raw, dict) else {}
        profile_enabled_by_name[profile_name] = _cfg_bool(
            profile_raw.get("enabled"),
            profile_enabled_global,
        )
        profile_weights_by_name[profile_name] = _coerce_weight_map(profile_raw.get("weights"))
        profile_merge_weights[profile_name] = max(0.0, _cfg_float(profile_raw.get("merge_weight"), 1.0))

    if active_profile_name not in profile_enabled_by_name:
        profile_enabled_by_name[active_profile_name] = active_profile_enabled
    if active_profile_name not in profile_weights_by_name:
        profile_weights_by_name[active_profile_name] = _coerce_weight_map(active_profile_cfg.get("weights"))
    if active_profile_name not in profile_merge_weights:
        profile_merge_weights[active_profile_name] = 1.0

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
        foundation_enabled=_cfg_bool(
            foundation_cfg.get("enabled"),
            _env_bool("AUTO_TEST_LLM_EVAL_FOUNDATION_ENABLED", True),
        ),
        foundation_weights=_coerce_weight_map(foundation_cfg.get("weights")),
        profile_active=active_profile_name,
        profile_active_profiles=active_profiles,
        profile_active_profiles_by_capability_mode=route_map,
        profile_enabled=active_profile_enabled,
        profile_enabled_by_name=profile_enabled_by_name,
        profile_weight=max(
            0.0,
            min(
                1.0,
                _cfg_float(
                    profile_cfg.get("weight"),
                    _env_float("AUTO_TEST_LLM_EVAL_PROFILE_WEIGHT", 0.35),
                ),
            ),
        ),
        profile_fallback_to_foundation_only=_cfg_bool(
            profile_cfg.get("fallback_to_foundation_only"),
            _env_bool("AUTO_TEST_LLM_EVAL_PROFILE_FALLBACK", True),
        ),
        profile_weights=_coerce_weight_map(active_profile_cfg.get("weights")),
        profile_weights_by_name=profile_weights_by_name,
        profile_merge_weights=profile_merge_weights,
        shadow_pass_threshold_0_100=max(
            0.0,
            min(
                100.0,
                _cfg_float(
                    llm_cfg.get("shadow_pass_threshold_0_100"),
                    _env_float("AUTO_TEST_LLM_EVAL_SHADOW_THRESHOLD_0_100", 70.0),
                ),
            ),
        ),
        primary_mode=primary_mode,
    )


def _normalize_primary_mode(raw: Any) -> str:
    mode = str(raw or "").strip().lower()
    if mode in {"llm_v1", "foundation_v2", "final_v2"}:
        return mode
    return "llm_v1"


def _resolve_profile_router(llm_cfg: LLMEvalConfig, capability_mode: str) -> dict[str, Any]:
    mode = str(capability_mode or "").strip().lower()
    route_map = (
        llm_cfg.profile_active_profiles_by_capability_mode
        if isinstance(llm_cfg.profile_active_profiles_by_capability_mode, dict)
        else {}
    )
    selected = route_map.get(mode)
    source = "active_profiles_by_capability_mode" if isinstance(selected, list) and selected else "active_profiles"
    if not isinstance(selected, list) or not selected:
        selected = llm_cfg.profile_active_profiles if isinstance(llm_cfg.profile_active_profiles, list) else []
    if not selected:
        selected = [llm_cfg.profile_active]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in selected:
        name = str(item or "").strip()
        if not name:
            continue
        lower = name.lower()
        if lower in seen:
            continue
        seen.add(lower)
        normalized.append(name)
    if not normalized:
        normalized = [str(llm_cfg.profile_active or "memory_compression").strip() or "memory_compression"]

    return {
        "source": source,
        "capability_mode": mode,
        "selected_profiles": normalized,
    }


def _to_bool(value: Any, default: bool = False) -> bool:
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


def _to_float(value: Any, default: float = 0.0) -> float:
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


def _build_primary_evaluation(llm_eval: dict[str, Any], llm_cfg: LLMEvalConfig) -> dict[str, Any]:
    mode = _normalize_primary_mode(llm_cfg.primary_mode)
    response_json = llm_eval.get("response_json")
    if not isinstance(response_json, dict):
        response_json = {}
    shadow = llm_eval.get("evaluation_v2_shadow")
    if not isinstance(shadow, dict):
        shadow = {}

    if mode == "foundation_v2":
        foundation = shadow.get("foundation")
        if not isinstance(foundation, dict):
            foundation = {}
        score_0_100 = round(max(0.0, min(100.0, _to_float(foundation.get("score"), 0.0) * 100.0)), 2)
        threshold_0_100 = round(max(0.0, min(100.0, _to_float(llm_cfg.shadow_pass_threshold_0_100, 70.0))), 2)
        return {
            "mode": mode,
            "source": "evaluation_v2_shadow.foundation",
            "pass": score_0_100 >= threshold_0_100,
            "score_0_100": score_0_100,
            "threshold_0_100": threshold_0_100,
        }

    if mode == "final_v2":
        final = shadow.get("final")
        if not isinstance(final, dict):
            final = {}
        score_0_100 = round(max(0.0, min(100.0, _to_float(final.get("score_0_100"), 0.0))), 2)
        threshold_0_100 = round(max(0.0, min(100.0, _to_float(final.get("threshold_0_100"), 70.0))), 2)
        return {
            "mode": mode,
            "source": "evaluation_v2_shadow.final",
            "pass": _to_bool(final.get("pass"), score_0_100 >= threshold_0_100),
            "score_0_100": score_0_100,
            "threshold_0_100": threshold_0_100,
        }

    score_0_100 = round(max(0.0, min(100.0, _to_float(response_json.get("score_0_100"), 0.0))), 2)
    return {
        "mode": "llm_v1",
        "source": "llm_evaluation.response_json",
        "pass": _to_bool(response_json.get("pass"), False),
        "score_0_100": score_0_100,
        "threshold_0_100": None,
    }


def _build_evaluation_compare(
    llm_eval: dict[str, Any],
    llm_cfg: LLMEvalConfig,
    evaluation_primary: dict[str, Any],
) -> dict[str, Any]:
    response_json = llm_eval.get("response_json")
    if not isinstance(response_json, dict):
        response_json = {}
    shadow = llm_eval.get("evaluation_v2_shadow")
    if not isinstance(shadow, dict):
        shadow = {}

    threshold_0_100 = round(max(0.0, min(100.0, _to_float(llm_cfg.shadow_pass_threshold_0_100, 70.0))), 2)

    llm_score = round(max(0.0, min(100.0, _to_float(response_json.get("score_0_100"), 0.0))), 2)
    llm_pass = _to_bool(response_json.get("pass"), False)
    llm_available = bool(response_json)

    foundation = shadow.get("foundation")
    if not isinstance(foundation, dict):
        foundation = {}
    foundation_score = round(max(0.0, min(100.0, _to_float(foundation.get("score"), 0.0) * 100.0)), 2)
    foundation_available = bool(foundation)
    foundation_pass = foundation_score >= threshold_0_100

    final = shadow.get("final")
    if not isinstance(final, dict):
        final = {}
    final_score = round(max(0.0, min(100.0, _to_float(final.get("score_0_100"), 0.0))), 2)
    final_threshold = round(max(0.0, min(100.0, _to_float(final.get("threshold_0_100"), threshold_0_100))), 2)
    final_available = bool(final)
    final_pass = _to_bool(final.get("pass"), final_score >= final_threshold)

    def _delta(left: float, left_ok: bool, right: float, right_ok: bool) -> float | None:
        if not (left_ok and right_ok):
            return None
        return round(left - right, 2)

    return {
        "primary_mode": str(evaluation_primary.get("mode") or _normalize_primary_mode(llm_cfg.primary_mode)),
        "llm_v1": {
            "available": llm_available,
            "pass": llm_pass if llm_available else None,
            "score_0_100": llm_score if llm_available else None,
            "threshold_0_100": None,
            "source": "llm_evaluation.response_json",
        },
        "foundation_v2": {
            "available": foundation_available,
            "pass": foundation_pass if foundation_available else None,
            "score_0_100": foundation_score if foundation_available else None,
            "threshold_0_100": threshold_0_100 if foundation_available else None,
            "source": "evaluation_v2_shadow.foundation",
        },
        "final_v2": {
            "available": final_available,
            "pass": final_pass if final_available else None,
            "score_0_100": final_score if final_available else None,
            "threshold_0_100": final_threshold if final_available else None,
            "source": "evaluation_v2_shadow.final",
        },
        "delta": {
            "foundation_minus_llm_v1": _delta(foundation_score, foundation_available, llm_score, llm_available),
            "final_minus_foundation": _delta(final_score, final_available, foundation_score, foundation_available),
            "final_minus_llm_v1": _delta(final_score, final_available, llm_score, llm_available),
        },
    }


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
    target_name = _normalize_target_name(
        sim_cfg.get("target_name")
        or sim_cfg.get("target")
        or os.getenv("AUTO_TEST_TARGET_NAME", DEFAULT_TARGET_NAME)
    )
    target_dir = _resolve_target_dir(target_name)
    target_intent_pool = _trim_prompt_text(
        _load_optional_text(
            target_dir / "scenarios" / "intent_pool.yaml",
            "- 闇€姹傛緞娓匼n- 鏂囨鍒涗綔\n- 鍥剧墖鐢熸垚\n- 鏂规杩介棶\n- 鍙戝竷鏁寸悊\n- 鎬荤粨楠屾敹",
        )
    )
    target_policies = _trim_prompt_text(
        _load_optional_text(
            target_dir / "scenarios" / "policies.yaml",
            "capabilities:\n"
            "  - text_generation\n"
            "  - image_generation\n"
            "  - planning\n"
            "constraints:\n"
            "  - follow_user_instruction\n"
            "  - avoid_tool_payload_leak",
        )
    )
    industry_options = str(
        sim_cfg.get("industry_options")
        or "椁愰ギ銆侀浂鍞€佹暀鑲层€佸尰鐤椼€佸埗閫犮€佽法澧冪數鍟嗐€佹湰鍦扮敓娲汇€丼aaS"
    ).strip()
    identity_options = str(
        sim_cfg.get("identity_options")
        or "鑰佹澘銆佽繍钀ヨ礋璐ｄ汉銆佸競鍦鸿礋璐ｄ汉銆佷骇鍝佺粡鐞嗐€佸尯鍩熶唬鐞嗐€侀噰璐礋璐ｄ汉"
    ).strip()
    personas_dir = target_dir / "personas"
    personas_file = personas_dir / f"{target_name}_users.json"
    if not personas_file.exists():
        candidates = sorted(personas_dir.glob("*.json")) if personas_dir.exists() else []
        personas_file = candidates[0] if candidates else personas_file
    target_personas = _trim_prompt_text(_load_optional_text(personas_file, "[]"))
    target_rubrics = _trim_prompt_text(_load_optional_text(target_dir / "rubrics" / "probe_rubrics.yaml", ""))

    max_turns = max(
        1,
        min(
            128,
            _cfg_int(
                sim_cfg.get("max_turns"),
                _env_int("AUTO_TEST_MAX_TURNS", _env_int("AUTO_TEST_USER_SIM_MAX_TURNS", 5)),
            ),
        ),
    )
    first_user_message = _ensure_required_note_clear_text(
        str(sim_cfg.get("first_user_message") or "").strip() or DEFAULT_FIRST_USER_TEXT
    )

    default_system_prompt = (
        "You are the user simulator in an automated test. "
        "Generate the next user input based on the role and conversation history. "
        "Return JSON only with fields: user_text, stop, reason. "
        "user_text must be natural language and must not include explanations."
    )
    default_scenario_prompt = (
        "You are role-playing a realistic customer for the target assistant.\n"
        "Target name: {{TARGET_NAME}}\n\n"
        "Target policies:\n"
        "{{TARGET_POLICIES}}\n\n"
        "Target intent pool:\n"
        "{{TARGET_INTENT_POOL}}\n\n"
        "Probe rubrics (optional):\n"
        "{{TARGET_RUBRICS}}\n\n"
        "Randomization constraints:\n"
        "1. Pick one main intent each turn; avoid fixed ordering.\n"
        "2. Each turn should request a concrete deliverable or a clear follow-up.\n"
        "3. Vary tone and length naturally.\n"
        "4. Avoid repeating the previous user sentence pattern.\n\n"
        "Turn limit: {{MAX_TURNS}}. Set stop=true only when the goal is complete."
    )
    default_role_system_prompt = (
        "You are a test-data designer that generates a role-playable random customer persona. "
        "Return JSON only, without explanations."
    )
    default_role_user_prompt = (
        "璇风敓鎴愪竴涓敤浜庡杞璇濇祴璇曠殑闅忔満瀹㈡埛瑙掕壊銆俓n"
        "鐩爣鍚嶇О锛歿{TARGET_NAME}}\n"
        "鐩爣鑳藉姏杈圭晫锛堜緵鍙傝€冿級锛歕n{{TARGET_POLICIES}}\n\n"
        "鍙€夌敤鎴风敾鍍忔牱鏈紙鍙€熼壌浣嗙姝㈢収鎶勶級锛歕n{{TARGET_PERSONAS}}\n\n"
        "绾︽潫淇℃伅锛歕n"
        "- 鏈疆鏈€澶у璇濇暟锛歿{MAX_TURNS}}\n"
        "- 棣栬疆鍥哄畾鐢ㄦ埛鍔ㄤ綔锛歿{REQUIRED_NOTEBOOK_CLEAR_TEXT}}\n\n"
        "瑕佹眰锛歕n"
        "1. 瑙掕壊瑕佹湁鐪熷疄涓氬姟鑳屾櫙涓庢矡閫氶鏍笺€俓n"
        "2. 蹇呴』鍖呭惈鍐茬獊绾︽潫锛堜緥濡傞绠椼€佹椂鏁堛€佸悎瑙勩€佽皟鎬у啿绐侊級銆俓n"
        "3. 灏介噺閬垮厤涓庡父瑙佹ā鏉垮悓璐ㄥ寲銆俓n\n"
        "杈撳嚭 JSON 瀛楁锛歕n"
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
    scenario_prompt_file = str(sim_cfg.get("scenario_prompt_file") or "").strip()
    scenario_prompt_path = _resolve_prompt_path(scenario_prompt_file, DEFAULT_USER_SIM_SCENARIO_PROMPT_FILE)
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
    target_vars = {
        "TARGET_NAME": target_name,
        "TARGET_INTENT_POOL": target_intent_pool,
        "TARGET_POLICIES": target_policies,
        "TARGET_PERSONAS": target_personas,
        "TARGET_RUBRICS": target_rubrics,
        "INDUSTRY_OPTIONS": industry_options,
        "IDENTITY_OPTIONS": identity_options,
        "MAX_TURNS": str(max_turns),
        "REQUIRED_NOTEBOOK_CLEAR_TEXT": REQUIRED_NOTEBOOK_CLEAR_TEXT,
    }
    scenario_template = str(sim_cfg.get("scenario_prompt") or "").strip()
    if not scenario_template:
        scenario_template = _load_prompt_template(
            scenario_prompt_path if scenario_prompt_file else DEFAULT_USER_SIM_SCENARIO_PROMPT_FILE,
            default_scenario_prompt,
        )
    scenario_prompt = _render_prompt_vars(
        scenario_template,
        target_vars,
    )
    role_system_prompt = str(sim_cfg.get("role_system_prompt") or "").strip() or _load_prompt_template(
        role_system_prompt_path,
        default_role_system_prompt,
    )
    role_user_template = str(sim_cfg.get("role_user_prompt") or "").strip() or _load_prompt_template(
        role_user_prompt_path,
        default_role_user_prompt,
    )
    role_user_prompt = _render_prompt_vars(role_user_template, target_vars)

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
        max_turns=max_turns,
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
        raise RuntimeError(f"create_session 涓氬姟閿欒: {payload}")

    session_id = ((payload.get("data") or {}).get("sessionId") or "").strip()
    if not session_id:
        raise RuntimeError(f"create_session 缂哄皯 sessionId: {payload}")
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
        raise RuntimeError("AUTO_TEST_PERSIST_TYPE 浠呮敮鎸?0(db) / 1(cache) / 2(one-shot)")
    return persist_type


def build_exec_max_turns() -> int:
    max_turns = _env_int("AUTO_TEST_EXEC_MAX_TURNS", 8)
    if max_turns < 1 or max_turns > 128:
        raise RuntimeError("AUTO_TEST_EXEC_MAX_TURNS 闇€鍦?1~128 涔嬮棿")
    return max_turns


def build_create_settings_json(run_settings: dict[str, Any]) -> str | None:
    import os

    # 浼樺厛浣跨敤鏄惧紡閰嶇疆锛涚敤浜庡榻愬墠绔€滀細璇濆垱寤哄嵆淇濆瓨 settings鈥濈殑璺緞銆?
    raw = os.getenv("AUTO_TEST_CREATE_SETTINGS_JSON", "").strip()
    if raw:
        custom = _safe_json_loads(raw)
        if not isinstance(custom, dict):
            raise RuntimeError("AUTO_TEST_CREATE_SETTINGS_JSON 蹇呴』鏄?JSON 瀵硅薄")
        return json.dumps(custom, ensure_ascii=False, sort_keys=True)

    # 榛樿鎶?runSettings 鍚屾鍒?create_session.settings锛岄伩鍏?skillIds 鍙紶 execute_session
    # 浣嗘湭杩涘叆鍒濆绯荤粺 prompt 鐨勬儏鍐点€?
    if _env_bool("AUTO_TEST_CREATE_SETTINGS_FROM_RUN_SETTINGS", True):
        return json.dumps(run_settings, ensure_ascii=False, sort_keys=True)
    return None


def build_expected_facts() -> dict[str, str]:
    # LLM-only mode no longer binds to fixed expected-facts assertions.
    return {}


def build_probe_eval_config(cfg: AuthConfig) -> ProbeEvalConfig:
    probe_cfg = cfg.probe_eval if isinstance(cfg.probe_eval, dict) else {}
    llm_cfg = probe_cfg.get("llm_judge")
    if not isinstance(llm_cfg, dict):
        llm_cfg = {}
    score_merge = probe_cfg.get("score_merge")
    if not isinstance(score_merge, dict):
        score_merge = {}
    default_probe_system_rel = DEFAULT_PROBE_JUDGE_SYSTEM_PROMPT_FILE.relative_to(PROMPTS_DIR).as_posix()
    default_probe_user_rel = DEFAULT_PROBE_JUDGE_USER_PROMPT_FILE.relative_to(PROMPTS_DIR).as_posix()

    enabled = _env_bool("AUTO_TEST_ENABLE_PROBE_EVAL", _cfg_bool(probe_cfg.get("enabled"), False))
    default_rel = Path("datasets") / "probes" / "clinic_memory_v1.json"
    raw_dataset = str(os.getenv("AUTO_TEST_PROBE_DATASET_PATH", "")).strip() or str(probe_cfg.get("dataset_path") or "").strip()
    dataset_path = (AUTO_TEST_DIR / default_rel).resolve()
    if raw_dataset:
        custom = Path(raw_dataset)
        dataset_path = custom if custom.is_absolute() else (AUTO_TEST_DIR / custom).resolve()

    llm_judge_cfg = ProbeLLMJudgeConfig(
        enabled=_env_bool("AUTO_TEST_ENABLE_PROBE_LLM_JUDGE", _cfg_bool(llm_cfg.get("enabled"), False)),
        base_url=str(
            os.getenv("AUTO_TEST_PROBE_LLM_URL", "")
            or llm_cfg.get("base_url")
            or llm_cfg.get("url")
            or ""
        ).strip(),
        model=str(os.getenv("AUTO_TEST_PROBE_LLM_MODEL", "") or llm_cfg.get("model") or "").strip(),
        api_key=str(os.getenv("AUTO_TEST_PROBE_LLM_API_KEY", "") or llm_cfg.get("api_key") or llm_cfg.get("key") or "").strip(),
        timeout_sec=max(10, min(300, _env_int("AUTO_TEST_PROBE_LLM_TIMEOUT_SEC", _cfg_int(llm_cfg.get("timeout_sec"), 45)))),
        repeats=max(1, min(9, _env_int("AUTO_TEST_PROBE_LLM_REPEATS", _cfg_int(llm_cfg.get("repeats"), 3)))),
        max_retries=max(0, min(6, _env_int("AUTO_TEST_PROBE_LLM_MAX_RETRIES", _cfg_int(llm_cfg.get("max_retries"), 2)))),
        fail_open=_env_bool("AUTO_TEST_PROBE_LLM_FAIL_OPEN", _cfg_bool(llm_cfg.get("fail_open"), False)),
        system_prompt_path=str(
            os.getenv("AUTO_TEST_PROBE_LLM_SYSTEM_PROMPT_FILE", "")
            or llm_cfg.get("system_prompt_file")
            or default_probe_system_rel
        ).strip(),
        user_prompt_path=str(
            os.getenv("AUTO_TEST_PROBE_LLM_USER_PROMPT_FILE", "")
            or llm_cfg.get("user_prompt_file")
            or default_probe_user_rel
        ).strip(),
    )
    deterministic_weight = _env_float(
        "AUTO_TEST_PROBE_DETERMINISTIC_WEIGHT",
        _cfg_float(score_merge.get("deterministic_weight"), 0.65),
    )
    llm_weight = _env_float(
        "AUTO_TEST_PROBE_LLM_FINAL_WEIGHT",
        _cfg_float(score_merge.get("llm_weight"), 0.35),
    )
    return ProbeEvalConfig(
        enabled=enabled,
        dataset_path=dataset_path,
        fail_on_error=_env_bool(
            "AUTO_TEST_PROBE_FAIL_ON_DATASET_ERROR",
            _cfg_bool(probe_cfg.get("fail_on_error"), True),
        ),
        max_fail_details=max(
            1,
            min(
                100,
                _env_int("AUTO_TEST_PROBE_MAX_FAIL_DETAILS", _cfg_int(probe_cfg.get("max_fail_details"), 20)),
            ),
        ),
        llm_judge=llm_judge_cfg,
        deterministic_weight=max(0.0, min(1.0, deterministic_weight)),
        llm_weight=max(0.0, min(1.0, llm_weight)),
    )


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
    probe_eval: dict[str, Any] | None = None,
) -> None:
    ok_turns = sum(1 for r in results if r.get("run_end") and not r.get("run_error"))
    lines = [
        "# Dialogue Test Metadata",
        "",
        f"- run_id: `{run_id}`",
        f"- session_id: `{session_id}`",
        f"- base_url: `{cfg.base_url}`",
        f"- proxy_http: `{cfg.proxy_http or ''}`",
        f"- proxy_https: `{cfg.proxy_https or ''}`",
        f"- proxy_no_proxy: `{cfg.proxy_no_proxy or ''}`",
        f"- uid: `{cfg.uid}`",
        f"- email: `{cfg.email}`",
        f"- token(masked): `{mask_token(cfg.token)}`",
        f"- runtime_env: `{cfg.selected_env}`",
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
            "## 杩借釜瀛楁璇存槑",
            "",
            "- Trace IDs come from backend response headers, not a client-side generated trace id.",
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
    if isinstance(probe_eval, dict):
        probe_cfg = probe_eval.get("config")
        if not isinstance(probe_cfg, dict):
            probe_cfg = {}
        summary = probe_eval.get("summary")
        if isinstance(summary, dict):
            lines.extend(
                [
                    "",
                    "## Probe Evaluation",
                    "",
                    f"- dataset_id: `{summary.get('dataset_id', '')}`",
                    f"- dataset_version: `{summary.get('dataset_version', '')}`",
                    f"- total_probes: `{summary.get('total_probes', 0)}`",
                    f"- passed_probes: `{summary.get('passed_probes', 0)}`",
                    f"- failed_probes: `{summary.get('failed_probes', 0)}`",
                    f"- skipped_probes: `{summary.get('skipped_probes', 0)}`",
                    f"- deterministic_score: `{summary.get('deterministic_score', 0)}`",
                    f"- llm_subjective_score: `{summary.get('llm_subjective_score', 0)}`",
                    f"- final_weighted_score: `{summary.get('final_weighted_score', summary.get('weighted_score', 0))}`",
                    f"- llm_probe_count: `{summary.get('llm_probe_count', 0)}`",
                    f"- llm_probe_failed: `{summary.get('llm_probe_failed', 0)}`",
                    f"- llm_probe_skipped: `{summary.get('llm_probe_skipped', 0)}`",
                    f"- critical_failed: `{summary.get('critical_failed', [])}`",
                    f"- score_merge: `det={probe_cfg.get('deterministic_weight', 0)}, llm={probe_cfg.get('llm_weight', 0)}`",
                    f"- llm_judge_enabled: `{probe_cfg.get('llm_judge_enabled', False)}`",
                    f"- llm_judge_model: `{probe_cfg.get('llm_judge_model', '')}`",
                    f"- llm_judge_repeats: `{probe_cfg.get('llm_judge_repeats', 0)}`",
                    f"- llm_judge_max_retries: `{probe_cfg.get('llm_judge_max_retries', 0)}`",
                    f"- llm_judge_fail_open: `{probe_cfg.get('llm_judge_fail_open', False)}`",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dialogue_md(path: Path, results: list[dict[str, Any]]) -> None:
    lines = ["# 鐢ㄦ埛瑙嗚瀵硅瘽璁板綍", ""]
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
    cfg = load_config(cfg_path, args.env)
    applied_proxy = apply_proxy_from_config(cfg)
    dotai_base_url = resolve_dotai_base_url(cfg)
    print(f"[INFO] config_path={cfg.source_path.as_posix()}")
    print(f"[INFO] runtime_env={cfg.selected_env}")

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
    if applied_proxy:
        print(f"[INFO] proxy_from_config={json.dumps(applied_proxy, ensure_ascii=False)}")
    else:
        print("[INFO] proxy_from_config=(none)")
    print("[INFO] creating session...")

    expected_facts = build_expected_facts()
    run_settings = build_run_settings()
    llm_eval_cfg = build_llm_eval_config(cfg)
    probe_eval_cfg = build_probe_eval_config(cfg)
    user_sim_cfg = build_user_simulator_config(cfg, llm_eval_cfg)
    profile_router = _resolve_profile_router(llm_eval_cfg, user_sim_cfg.capability_mode)
    llm_eval_cfg.profile_active_profiles = profile_router.get("selected_profiles", llm_eval_cfg.profile_active_profiles)
    llm_eval_cfg.profile_router_context = profile_router
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
        print(
            "[INFO] llm_eval_v2_shadow "
            f"primary_mode={llm_eval_cfg.primary_mode} "
            f"foundation_enabled={llm_eval_cfg.foundation_enabled} "
            f"profile={llm_eval_cfg.profile_active} "
            f"profiles={llm_eval_cfg.profile_active_profiles} "
            f"profile_route_source={profile_router.get('source')} "
            f"profile_enabled={llm_eval_cfg.profile_enabled} "
            f"profile_weight={llm_eval_cfg.profile_weight}"
        )
    print(f"[INFO] probe_eval_enabled={probe_eval_cfg.enabled}")
    print(f"[INFO] probe_eval_dataset={probe_eval_cfg.dataset_path.as_posix()}")
    print(
        "[INFO] probe_eval_score_merge="
        f"deterministic:{probe_eval_cfg.deterministic_weight},llm:{probe_eval_cfg.llm_weight}"
    )
    print(f"[INFO] probe_llm_judge_enabled={probe_eval_cfg.llm_judge.enabled}")
    if probe_eval_cfg.llm_judge.enabled:
        print(f"[INFO] probe_llm_judge_url={probe_eval_cfg.llm_judge.base_url}")
        print(f"[INFO] probe_llm_judge_model={probe_eval_cfg.llm_judge.model}")
        print(
            "[INFO] probe_llm_judge_repeats="
            f"{probe_eval_cfg.llm_judge.repeats},retries={probe_eval_cfg.llm_judge.max_retries},"
            f"fail_open={probe_eval_cfg.llm_judge.fail_open}"
        )

    if args.max_turns > 0:
        user_sim_cfg.max_turns = max(1, min(128, args.max_turns))
        print(f"[INFO] override max_turns from CLI: {user_sim_cfg.max_turns}")

    if not user_sim_cfg.enabled:
        raise RuntimeError("LLM-only mode requires config.user_simulator.enabled=true.")
    if not user_sim_cfg.base_url or not user_sim_cfg.model or not user_sim_cfg.api_key:
        raise RuntimeError("Incomplete user_simulator config: base_url/model/api_key are required.")
    if not llm_eval_cfg.enabled:
        raise RuntimeError("LLM-only mode requires config.llm_eval.enabled=true.")
    if not llm_eval_cfg.base_url or not llm_eval_cfg.model or not llm_eval_cfg.api_key:
        raise RuntimeError("Incomplete llm_eval config: base_url/model/api_key are required.")

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
                    raise RuntimeError(f"user_simulator turn={idx} 鐢熸垚澶辫触: {exc}") from exc

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

    (run_data_dir / "turn_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    probe_eval_payload: dict[str, Any] | None = None
    if probe_eval_cfg.enabled:
        try:
            probe_eval_payload = evaluate_probes(
                dataset_path=probe_eval_cfg.dataset_path,
                turn_results_path=run_data_dir / "turn_results.json",
                workspace_manifest_path=result_dir / "workspace" / "_manifest.json",
                raw_events_path=raw_events_path,
                llm_judge_cfg=probe_eval_cfg.llm_judge,
                prompts_dir=PROMPTS_DIR,
                deterministic_weight=probe_eval_cfg.deterministic_weight,
                llm_weight=probe_eval_cfg.llm_weight,
            )
            (run_data_dir / "probe_results.json").write_text(
                json.dumps(probe_eval_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            write_probe_evaluation_md(
                result_dir / "probe_evaluation.md",
                probe_eval_payload,
                max_fail_details=probe_eval_cfg.max_fail_details,
            )
            summary = probe_eval_payload.get("summary")
            if isinstance(summary, dict):
                print(
                    "[INFO] probe_eval "
                    f"total={summary.get('total_probes', 0)} "
                    f"passed={summary.get('passed_probes', 0)} "
                    f"failed={summary.get('failed_probes', 0)} "
                    f"score={summary.get('final_weighted_score', summary.get('weighted_score', 0))} "
                    f"det={summary.get('deterministic_score', 0)} "
                    f"llm={summary.get('llm_subjective_score', 0)}"
                )
        except Exception as exc:
            if probe_eval_cfg.fail_on_error:
                raise RuntimeError(f"probe evaluation failed: {exc}") from exc
            probe_eval_payload = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "skipped": True,
                "error": str(exc),
                "dataset_path": probe_eval_cfg.dataset_path.as_posix(),
            }
            (run_data_dir / "probe_results.json").write_text(
                json.dumps(probe_eval_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[WARN] probe evaluation skipped due to error: {exc}")

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
        probe_eval=probe_eval_payload,
    )
    write_dialogue_md(result_dir / "dialogue.md", results)

    llm_eval = evaluate_with_llm(
        results,
        expected_facts=expected_facts,
        prompts_dir=AUTO_TEST_DIR / "prompts",
        llm_cfg=llm_eval_cfg,
    )
    if llm_eval.get("skipped") or llm_eval.get("error"):
        raise RuntimeError(f"LLM 璇勪及澶辫触: {llm_eval.get('reason') or llm_eval.get('error')}")

    evaluation_payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_mode": "llm_only",
        "llm_evaluation": llm_eval,
    }
    evaluation_v2_shadow = llm_eval.get("evaluation_v2_shadow")
    if isinstance(evaluation_v2_shadow, dict):
        evaluation_payload["evaluation_v2_shadow"] = evaluation_v2_shadow
        if _normalize_primary_mode(llm_eval_cfg.primary_mode) != "llm_v1":
            evaluation_payload["evaluation_v2"] = evaluation_v2_shadow
    evaluation_primary = _build_primary_evaluation(llm_eval, llm_eval_cfg)
    evaluation_payload["evaluation_primary"] = evaluation_primary
    evaluation_compare = _build_evaluation_compare(llm_eval, llm_eval_cfg, evaluation_primary)
    evaluation_payload["evaluation_compare"] = evaluation_compare

    (run_data_dir / "evaluation.json").write_text(
        json.dumps(
            evaluation_payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_evaluation_md(
        result_dir / "evaluation.md",
        rule_eval=None,
        llm_eval=llm_eval,
        evaluation_primary=evaluation_primary,
        evaluation_compare=evaluation_compare,
    )

    (run_data_dir / "README.md").write_text(
        "# run_data 缁撴瀯璇存槑\n\n"
        "- `turn_results.json`: 姣忚疆缁撴瀯鍖栫粨鏋溿€俓n"
        "- `evaluation.json`: LLM-only 璇勪及缁撴灉锛堝惈 `evaluation_v2_shadow` 褰卞瓙缁撴瀯锛夈€俓n"
        "- `evaluation.json.evaluation_primary`: 褰撳墠涓诲垽瀹氱粨鏋滐紙鍙厤缃?`llm_v1/foundation_v2/final_v2`锛夈€俓n"
        "- `evaluation.json.evaluation_compare`: `llm_v1/foundation_v2/final_v2` A/B compare and score deltas.\n"
        "- `evaluation.json.evaluation_v2_shadow.profile_combined`: Phase D multi-profile merged score view.\n"
        "- `probe_results.json`: 鎺㈤拡璇勪及缁撴瀯鍖栫粨鏋滐紙鍚敤 `AUTO_TEST_ENABLE_PROBE_EVAL=true` 鏃剁敓鎴愶紱鏀寔 deterministic + llm judge锛夈€俓n"
        "- `../workspace/`: 鐢ㄦ埛鍙宸ヤ綔鍖哄鍑猴紙鍚?`_manifest.json` 涓庢枃浠惰惤鐩樼粨鏋滐級銆俓n",
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
    shadow_summary = ""
    if isinstance(evaluation_v2_shadow, dict):
        final = evaluation_v2_shadow.get("final")
        if isinstance(final, dict):
            shadow_summary = (
                f"shadow_score_0_100={final.get('score_0_100')} "
                f"shadow_pass={final.get('pass')}"
            )
    print(f"[INFO] evaluation llm_only {llm_summary}".strip())
    print(
        "[INFO] evaluation_primary "
        f"mode={evaluation_primary.get('mode')} "
        f"pass={evaluation_primary.get('pass')} "
        f"score_0_100={evaluation_primary.get('score_0_100')}"
    )
    if shadow_summary:
        print(f"[INFO] evaluation_v2_shadow {shadow_summary}".strip())
    compare_delta = evaluation_compare.get("delta") if isinstance(evaluation_compare, dict) else None
    if isinstance(compare_delta, dict):
        print(
            "[INFO] evaluation_compare "
            f"foundation_minus_llm_v1={compare_delta.get('foundation_minus_llm_v1')} "
            f"final_minus_foundation={compare_delta.get('final_minus_foundation')} "
            f"final_minus_llm_v1={compare_delta.get('final_minus_llm_v1')}"
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
