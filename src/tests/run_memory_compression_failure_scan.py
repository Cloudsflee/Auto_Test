from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import statistics
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

AUTO_TEST_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = AUTO_TEST_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests.memory_failure_llm import (  # noqa: E402
    CAUSE_COMPRESSION_RELATED,
    CAUSE_INSTRUCTION_OVERRIDE,
    CAUSE_INSUFFICIENT_EVIDENCE,
    CAUSE_TASK_SWITCH,
    CAUSE_UNKNOWN,
    JUDGE_AMBIGUOUS,
    JUDGE_FORGOTTEN,
    JUDGE_REMEMBERED,
    LLMEndpointConfig,
    build_default_probe_anchor,
    build_fallback_probe_question,
    generate_probe_question,
    judge_loss_suspected,
    verify_loss_confirmed,
)
from tests.run_5turn_session_test import (  # noqa: E402
    DEFAULT_FIRST_USER_TEXT,
    apply_proxy_from_config,
    build_auth_context,
    build_create_settings_json,
    build_exec_max_turns,
    build_llm_eval_config,
    build_persist_type,
    build_run_settings,
    build_user_simulator_config,
    cancel_session,
    create_session,
    execute_turn,
    load_config,
    normalize_authz,
    now_stamp,
    preflight_cleanup_test_processes,
    resolve_config_path,
    resolve_dotai_base_url,
)
from tests.user_simulator_engine import (  # noqa: E402
    GeneratedRole,
    UserSimulatorState,
    generate_role_for_session,
    generate_user_turn_with_simulator,
)
from tests.workspace_pipeline import export_workspace_view  # noqa: E402


@dataclass
class ScanArgs:
    env: str
    sessions: int
    warmup_sessions: int
    hard_max_turns: int
    probe_interval: int
    focus_window: int
    parallel_sessions: int
    importance_tier: str = "should_keep"
    turn_timeout_sec: int = 420
    session_wall_timeout_sec: int = 0
    min_post_samples_per_tier: int = 3
    probe_mode: str = "hidden"
    probe_cooldown_turns: int = 4
    probe_similarity_threshold: float = 0.9
    probe_recent_window: int = 4
    probe_regen_max_attempts: int = 2
    skip_preflight_cleanup: bool = False


@dataclass
class SharedRuntime:
    cfg: Any
    headers: dict[str, str]
    run_settings: dict[str, Any]
    persist_type: int
    exec_max_turns: int
    create_settings_json: str | None
    user_sim_cfg: Any
    dotai_base_url: str
    sessions_root: Path
    args: ScanArgs
    probe_llm_cfg: LLMEndpointConfig


PROBE_KIND_MEMORY_POINT = "memory_point_probe"
COMPACTION_TOOL_NAMES = {"memory_file_compaction", "summary_compaction"}
COMPACTION_TOOL_SCOPE_MAP = {
    "memory_file_compaction": "memory_file",
    "summary_compaction": "session_summary",
}
COMPACTION_REQUIRED_SCOPES = ("memory_file", "session_summary")
COMPACTION_SUCCESS_MARKERS = ("压缩完成", "壓縮完成", "compacted")
COMPACTION_FAILURE_MARKERS = ("压缩失败", "壓縮失敗", "compaction failed", "SESSION_MEMORY_COMPACTION_FAILED")
IMPORTANCE_TIERS = {"must_keep", "should_keep", "may_drop"}
PROBE_MODES = {"hidden", "explicit"}


def _as_bool(value: Any, default: bool = False) -> bool:
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


def _as_float(value: Any, default: float = 0.0) -> float:
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


def _as_int(value: Any, default: int = 0) -> int:
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
            return int(float(raw))
        except Exception:
            return default
    return default


def _normalize_importance_tier(value: Any) -> str:
    tier = str(value or "").strip().lower()
    if tier in IMPORTANCE_TIERS:
        return tier
    return "should_keep"


def _normalize_probe_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in PROBE_MODES:
        return mode
    return "hidden"


def parse_args(argv: list[str]) -> ScanArgs:
    parser = argparse.ArgumentParser(description="Scan memory-compression first-failure turn across sessions.")
    parser.add_argument("--env", type=str, default="", help="Runtime environment (prod/test).")
    parser.add_argument("--sessions", type=int, default=12, help="Session count.")
    parser.add_argument("--warmup-sessions", type=int, default=3, help="Warmup sessions excluded from chart.")
    parser.add_argument("--hard-max-turns", type=int, default=40, help="Safety cap per session; 0 = no cap.")
    parser.add_argument("--probe-interval", type=int, default=3, help="Coarse probe interval.")
    parser.add_argument("--focus-window", type=int, default=3, help="Focus window around estimated failure.")
    parser.add_argument("--parallel-sessions", type=int, default=1, help="Concurrent sessions (warmup always sequential).")
    parser.add_argument(
        "--turn-timeout-sec",
        type=int,
        default=420,
        help="Per-turn timeout seconds (applied via AUTO_TEST_TURN_TIMEOUT_SEC).",
    )
    parser.add_argument(
        "--session-wall-timeout-sec",
        type=int,
        default=0,
        help="Per-session wall-clock timeout seconds. <=0 disables this guard.",
    )
    parser.add_argument(
        "--importance-tier",
        type=str,
        default="should_keep",
        help="Gating tier for this run: must_keep / should_keep / may_drop.",
    )
    parser.add_argument(
        "--min-post-samples-per-tier",
        type=int,
        default=3,
        help="Minimum post-compaction samples required per tier for effect metrics.",
    )
    parser.add_argument(
        "--probe-mode",
        type=str,
        default="hidden",
        choices=sorted(PROBE_MODES),
        help="Probe style: hidden=embed probe in natural business demand; explicit=direct memory probe.",
    )
    parser.add_argument(
        "--probe-cooldown-turns",
        type=int,
        default=4,
        help="Minimum turn gap before reusing same probe signature.",
    )
    parser.add_argument(
        "--probe-similarity-threshold",
        type=float,
        default=0.9,
        help="Skip candidate probe if text similarity with recent probes exceeds this threshold.",
    )
    parser.add_argument(
        "--probe-recent-window",
        type=int,
        default=4,
        help="How many recent probe texts to check for dedup similarity.",
    )
    parser.add_argument(
        "--probe-regen-max-attempts",
        type=int,
        default=2,
        help="Max re-generation attempts when candidate probe is rejected by cooldown/similarity guard.",
    )
    parser.add_argument(
        "--skip-preflight-cleanup",
        action="store_true",
        help="Deprecated no-op. Preflight process cleanup is disabled.",
    )
    parsed = parser.parse_args(argv)
    return ScanArgs(
        env=str(parsed.env or "").strip(),
        sessions=max(1, min(300, int(parsed.sessions))),
        warmup_sessions=max(0, min(299, int(parsed.warmup_sessions))),
        hard_max_turns=max(0, min(40, int(parsed.hard_max_turns))),
        probe_interval=max(1, min(60, int(parsed.probe_interval))),
        focus_window=max(1, min(60, int(parsed.focus_window))),
        parallel_sessions=max(1, min(32, int(parsed.parallel_sessions))),
        importance_tier=_normalize_importance_tier(parsed.importance_tier),
        turn_timeout_sec=max(120, min(1800, int(parsed.turn_timeout_sec))),
        session_wall_timeout_sec=max(0, min(14400, int(parsed.session_wall_timeout_sec))),
        min_post_samples_per_tier=max(1, min(200, int(parsed.min_post_samples_per_tier))),
        probe_mode=_normalize_probe_mode(parsed.probe_mode),
        probe_cooldown_turns=max(0, min(20, int(parsed.probe_cooldown_turns))),
        probe_similarity_threshold=max(0.6, min(0.999, float(parsed.probe_similarity_threshold))),
        probe_recent_window=max(1, min(12, int(parsed.probe_recent_window))),
        probe_regen_max_attempts=max(0, min(8, int(parsed.probe_regen_max_attempts))),
        skip_preflight_cleanup=bool(parsed.skip_preflight_cleanup),
    )


def _estimate_focus_turn(failure_turns: list[int]) -> int | None:
    if not failure_turns:
        return None
    return max(4, int(round(statistics.median(failure_turns))))


def _build_scan_mode_label(focus_turn: int | None, focus_window: int) -> str:
    return "coarse" if focus_turn is None else f"focused@{focus_turn}+-{focus_window}"


def _should_probe_turn(turn_idx: int, coarse_interval: int, focus_turn: int | None, focus_window: int) -> bool:
    if turn_idx < 4:
        return False
    base_gap = turn_idx - 4
    if focus_turn is None:
        return base_gap % coarse_interval == 0
    left = max(4, focus_turn - focus_window)
    right = focus_turn + focus_window
    if turn_idx < left:
        return base_gap % coarse_interval == 0
    if left <= turn_idx <= right:
        return True
    after_interval = max(1, coarse_interval // 2)
    return (turn_idx - right) % after_interval == 0


def _build_filler_prompt(session_idx: int, turn_idx: int, session_id: str) -> str:
    md_path = f"/workspace/advoo/{session_id}/memory_scan_s{session_idx:02d}.md"
    img_path = f"/workspace/advoo/{session_id}/memory_scan_s{session_idx:02d}_t{turn_idx:02d}.jpg"
    text_first = [
        (
            "这版先按之前方向给我一段口语文案，控制在80字内，"
            f"并存成 md 到 `{md_path}`；同时配一张同主题图片到 `{img_path}`，文案优先。"
        ),
        (
            "先给我一版可直接发布的短文案并落盘，"
            f"md 放 `{md_path}`；再补一张配图放 `{img_path}`，这轮文案细节要更完整。"
        ),
        (
            "我先看文案版本，你给我2个自然标题+1段正文写进 "
            f"`{md_path}`，再生成一张呼应封面图存 `{img_path}`。"
        ),
    ]
    image_first = [
        (
            "这轮我先看视觉，先出一张图到 "
            f"`{img_path}`，再补一段简短文案并写入 `{md_path}`，图片优先。"
        ),
        (
            "先给我一张更有氛围感的主视觉图放 "
            f"`{img_path}`，然后把对应发布文案整理到 `{md_path}`。"
        ),
        (
            "先出图后出字：配图存 "
            f"`{img_path}`，文案存 `{md_path}`，两项都要但这轮视觉占比更高。"
        ),
    ]
    balanced = [
        (
            "这轮文图一起给：一段口语文案写入 "
            f"`{md_path}`，再来一张同主题配图存 `{img_path}`，两边比重五五开。"
        ),
        (
            "我要一个文图成套版本，文案落到 "
            f"`{md_path}`，配图落到 `{img_path}`，语气自然别太模板化。"
        ),
    ]
    roll = random.random()
    if roll < 0.4:
        bucket = text_first
    elif roll < 0.8:
        bucket = image_first
    else:
        bucket = balanced
    return random.choice(bucket)


def _normalize_probe_text_for_similarity(text: Any) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return ""
    tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", raw)
    return "".join(tokens)


def _text_similarity_ratio(text_a: Any, text_b: Any) -> float:
    a = _normalize_probe_text_for_similarity(text_a)
    b = _normalize_probe_text_for_similarity(text_b)
    if (not a) or (not b):
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def _max_probe_similarity(candidate_text: str, history_texts: list[str], window: int) -> float:
    if not candidate_text or not history_texts:
        return 0.0
    if window <= 0:
        window = 1
    tail = history_texts[-window:]
    return max((_text_similarity_ratio(candidate_text, item) for item in tail), default=0.0)


def _judge_probe_candidate(
    *,
    candidate_text: str,
    candidate_signature: str,
    turn_idx: int,
    recent_probe_texts: list[str],
    probe_last_turn_by_signature: dict[str, int],
    similarity_threshold: float,
    recent_window: int,
    cooldown_turns: int,
) -> tuple[bool, str, float]:
    if not candidate_text:
        return False, "candidate_empty", 0.0
    signature = str(candidate_signature or "").strip()
    if cooldown_turns > 0 and signature:
        last_turn = probe_last_turn_by_signature.get(signature)
        if isinstance(last_turn, int) and last_turn > 0:
            turn_gap = max(0, turn_idx - last_turn)
            if turn_gap < cooldown_turns:
                return False, f"cooldown_active:{turn_gap}<{cooldown_turns}", 0.0
    max_similarity = _max_probe_similarity(candidate_text, recent_probe_texts, recent_window)
    if max_similarity >= similarity_threshold:
        return False, f"similarity_high:{round(max_similarity, 4)}>={round(similarity_threshold, 4)}", max_similarity
    return True, "", max_similarity


def _generate_filler_user_turn(
    *,
    shared: SharedRuntime,
    sim_state: UserSimulatorState,
    turn_results: list[dict[str, Any]],
    generated_role: GeneratedRole,
    turn_idx: int,
    latest_workspace_snapshot: dict[str, Any] | None,
    session_idx: int,
    session_id: str,
) -> tuple[str, str, UserSimulatorState]:
    next_state = sim_state
    try:
        sim_text, should_stop, next_state = generate_user_turn_with_simulator(
            sim_cfg=shared.user_sim_cfg,
            sim_state=sim_state,
            results=turn_results,
            role=generated_role,
            turn_idx=turn_idx,
            workspace_snapshot=latest_workspace_snapshot,
        )
        user_text = str(sim_text or "").strip()
        if should_stop or not user_text:
            return _build_filler_prompt(session_idx, turn_idx, session_id), "simulator_fallback_filler", next_state
        return user_text, "simulator_filler", next_state
    except Exception as exc:
        print(
            "[WARN] simulator_turn_fallback "
            f"session={session_idx} turn={turn_idx} reason={exc.__class__.__name__}"
        )
        return _build_filler_prompt(session_idx, turn_idx, session_id), "simulator_error_fallback", next_state


def _workspace_artifact_counts(payload: dict[str, Any]) -> tuple[int, int]:
    counts = payload.get("workspace_counts")
    if not isinstance(counts, dict):
        return 0, 0
    text_files = int(counts.get("exported_text_files") or 0)
    image_files = int(counts.get("exported_image_files") or 0)
    return text_files, image_files


def _iter_raw_events(raw_events_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not raw_events_path.exists():
        return rows
    for line in raw_events_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _looks_like_compaction_text(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    for marker in COMPACTION_SUCCESS_MARKERS:
        if marker.lower() in raw:
            return True
    return False


def _looks_like_compaction_failure_text(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    for marker in COMPACTION_FAILURE_MARKERS:
        if marker.lower() in raw:
            return True
    return False


def _compaction_scope_from_tool_name(tool_name: str) -> str:
    return COMPACTION_TOOL_SCOPE_MAP.get(str(tool_name or "").strip(), "unknown")


def _event_order_tuple(row: dict[str, Any], fallback_index: int) -> tuple[int, int]:
    seq = _as_int(row.get("event_seq"), -1)
    if seq < 0:
        return (10**9, fallback_index)
    sub_seq = _as_int(row.get("event_sub_seq"), fallback_index)
    return (seq, sub_seq)


def _detect_compaction_events(
    raw_events_path: Path,
    turn_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _iter_raw_events(raw_events_path)
    events: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    text_done_turns: set[int] = set()
    text_failure_turns: set[int] = set()
    observed_turns: set[int] = set()
    by_turn: dict[int, dict[str, Any]] = {}

    for idx, row in enumerate(rows):
        event_raw = row.get("event_raw")
        if not isinstance(event_raw, dict):
            continue
        event_type = str(row.get("event_type") or "").strip().upper()
        turn = _as_int(row.get("turn"), 0)
        if turn <= 0:
            continue
        observed_turns.add(turn)
        state = by_turn.setdefault(
            turn,
            {
                "tool_start": {},
                "tool_result": {},
                "run_end_orders": [],
                "text_done_orders": [],
                "text_done_scopes": [],
                "failure_orders": [],
            },
        )
        order_tuple = _event_order_tuple(row, idx)
        name = str(event_raw.get("toolCallName") or event_raw.get("tool_call_name") or "").strip()
        tool_call_id = str(event_raw.get("toolCallId") or event_raw.get("tool_call_id") or "").strip()
        content = str(event_raw.get("content") or "")
        run_err_msg = str(event_raw.get("message") or "")

        if event_type == "TOOL_CALL_START" and name in COMPACTION_TOOL_NAMES:
            scope = _compaction_scope_from_tool_name(name)
            state["tool_start"][name] = {
                "scope": scope,
                "tool_call_id": tool_call_id,
                "order": order_tuple,
            }
            key = (turn, scope, "tool_start")
            if key not in seen:
                seen.add(key)
                events.append(
                    {
                        "turn": turn,
                        "event": "compaction_start",
                        "scope": scope,
                        "source": "raw_event",
                        "message": name,
                    }
                )
            continue

        if event_type == "TOOL_RESULT" and name in COMPACTION_TOOL_NAMES:
            scope = _compaction_scope_from_tool_name(name)
            state["tool_result"][name] = {
                "scope": scope,
                "tool_call_id": tool_call_id,
                "order": order_tuple,
            }
            key = (turn, scope, "tool_result_done")
            if key not in seen:
                seen.add(key)
                events.append(
                    {
                        "turn": turn,
                        "event": "compaction_done",
                        "scope": scope,
                        "source": "raw_event",
                        "message": name,
                    }
                )
            result_text = event_raw.get("result")
            if isinstance(result_text, (dict, list)):
                result_text = json.dumps(result_text, ensure_ascii=False)
            if _looks_like_compaction_failure_text(str(result_text or "")):
                fail_key = (turn, scope, "tool_result_failed")
                if fail_key not in seen:
                    seen.add(fail_key)
                    text_failure_turns.add(turn)
                    state["failure_orders"].append(order_tuple)
                    events.append(
                        {
                            "turn": turn,
                            "event": "compaction_failed",
                            "scope": scope,
                            "source": "raw_event",
                            "message": str(result_text or "").strip(),
                        }
                    )
            continue

        if event_type == "RUN_END":
            state["run_end_orders"].append(order_tuple)
            continue

        if event_type == "TEXT" and _looks_like_compaction_text(content):
            scope = "unknown"
            if "会话" in content or "session" in content.lower():
                scope = "session_summary"
            elif "文件" in content or "file" in content.lower() or "memory" in content.lower():
                scope = "memory_file"
            key = (turn, scope, "text_done")
            if key not in seen:
                seen.add(key)
                text_done_turns.add(turn)
                state["text_done_orders"].append(order_tuple)
                state["text_done_scopes"].append(scope)
                events.append(
                    {
                        "turn": turn,
                        "event": "compaction_done",
                        "scope": scope,
                        "source": "raw_event",
                        "message": content.strip(),
                    }
                )
            continue

        if event_type == "RUN_ERROR" and _looks_like_compaction_failure_text(run_err_msg):
            key = (turn, "unknown", "run_error")
            if key not in seen:
                seen.add(key)
                text_failure_turns.add(turn)
                state["failure_orders"].append(order_tuple)
                events.append(
                    {
                        "turn": turn,
                        "event": "compaction_failed",
                        "scope": "unknown",
                        "source": "raw_event",
                        "message": run_err_msg.strip(),
                    }
                )

    for tr in turn_results:
        turn = int(tr.get("turn") or 0)
        run_error = str(tr.get("run_error") or "").strip()
        assistant_text = str(tr.get("assistant_text") or "").strip()
        if run_error and _looks_like_compaction_failure_text(run_error):
            key = (turn, "unknown", "turn_result_error")
            if key not in seen:
                seen.add(key)
                text_failure_turns.add(turn)
                events.append(
                    {
                        "turn": turn,
                        "event": "compaction_failed",
                        "scope": "unknown",
                        "source": "turn_result",
                        "message": run_error,
                    }
                )
        if assistant_text and _looks_like_compaction_text(assistant_text):
            key = (turn, "unknown", "turn_result_done")
            if key not in seen:
                seen.add(key)
                text_done_turns.add(turn)
                events.append(
                    {
                        "turn": turn,
                        "event": "compaction_done",
                        "scope": "unknown",
                        "source": "turn_result",
                        "message": assistant_text,
                    }
                )

    verified_completion_turns: list[int] = []
    tool_lifecycle_verified_turns: list[int] = []
    indicator_text_verified_turns: list[int] = []
    verified_completion_methods_by_turn: dict[int, str] = {}
    completion_turn_diagnostics: list[dict[str, Any]] = []
    required_tool_names = sorted(COMPACTION_TOOL_NAMES)
    required_scope_names = sorted(COMPACTION_REQUIRED_SCOPES)
    for turn in sorted(by_turn.keys()):
        state = by_turn.get(turn) if isinstance(by_turn.get(turn), dict) else {}
        start_map = state.get("tool_start") if isinstance(state.get("tool_start"), dict) else {}
        result_map = state.get("tool_result") if isinstance(state.get("tool_result"), dict) else {}
        run_end_orders = state.get("run_end_orders") if isinstance(state.get("run_end_orders"), list) else []
        text_done_orders = state.get("text_done_orders") if isinstance(state.get("text_done_orders"), list) else []
        text_done_scopes = state.get("text_done_scopes") if isinstance(state.get("text_done_scopes"), list) else []
        failure_orders = state.get("failure_orders") if isinstance(state.get("failure_orders"), list) else []
        started_tools = sorted(start_map.keys())
        result_tools = sorted(result_map.keys())
        started_scopes = sorted({_compaction_scope_from_tool_name(name) for name in started_tools})
        result_scopes = sorted({_compaction_scope_from_tool_name(name) for name in result_tools})
        indicator_scopes = sorted({str(scope or "") for scope in text_done_scopes if str(scope or "")})
        missing_start_tools = [name for name in required_tool_names if name not in start_map]
        missing_result_tools = [name for name in required_tool_names if name not in result_map]
        missing_result_scopes = [scope for scope in required_scope_names if scope not in result_scopes]
        latest_result_order: tuple[int, int] | None = None
        for row in result_map.values():
            if not isinstance(row, dict):
                continue
            row_order = row.get("order")
            if not (isinstance(row_order, tuple) and len(row_order) == 2):
                continue
            if (latest_result_order is None) or (row_order > latest_result_order):
                latest_result_order = row_order
        has_run_end_after_compaction = False
        if latest_result_order is not None:
            has_run_end_after_compaction = any(
                isinstance(order, tuple) and len(order) == 2 and order > latest_result_order for order in run_end_orders
            )
        tool_lifecycle_verified = (
            (not missing_start_tools)
            and (not missing_result_tools)
            and (not missing_result_scopes)
            and has_run_end_after_compaction
        )
        latest_indicator_order: tuple[int, int] | None = None
        for order in text_done_orders:
            if not (isinstance(order, tuple) and len(order) == 2):
                continue
            if (latest_indicator_order is None) or (order > latest_indicator_order):
                latest_indicator_order = order
        has_run_end_after_indicator = False
        if latest_indicator_order is not None:
            has_run_end_after_indicator = any(
                isinstance(order, tuple) and len(order) == 2 and order > latest_indicator_order for order in run_end_orders
            )
        indicator_text_verified = bool(text_done_orders) and has_run_end_after_indicator and (not failure_orders)
        completion_verified = tool_lifecycle_verified or indicator_text_verified
        completion_method = (
            "tool_lifecycle"
            if tool_lifecycle_verified
            else ("indicator_text_with_run_end" if indicator_text_verified else "")
        )
        completion_turn_diagnostics.append(
            {
                "turn": turn,
                "tool_starts": started_tools,
                "tool_results": result_tools,
                "started_scopes": started_scopes,
                "result_scopes": result_scopes,
                "indicator_scopes": indicator_scopes,
                "missing_start_tools": missing_start_tools,
                "missing_result_tools": missing_result_tools,
                "missing_result_scopes": missing_result_scopes,
                "has_run_end_after_compaction": has_run_end_after_compaction,
                "has_run_end_after_indicator": has_run_end_after_indicator,
                "has_failure_signal": bool(failure_orders),
                "tool_lifecycle_verified": tool_lifecycle_verified,
                "indicator_text_verified": indicator_text_verified,
                "completion_method": completion_method,
                "completion_verified": completion_verified,
            }
        )
        if completion_verified:
            verified_completion_turns.append(turn)
            if tool_lifecycle_verified:
                tool_lifecycle_verified_turns.append(turn)
            if indicator_text_verified:
                indicator_text_verified_turns.append(turn)
            verified_completion_methods_by_turn[turn] = completion_method
            key = (turn, "full", "completion_verified")
            if key not in seen:
                seen.add(key)
                message = (
                    "memory_file_compaction+summary_compaction+run_end"
                    if tool_lifecycle_verified
                    else "compaction_indicator_text+run_end"
                )
                events.append(
                    {
                        "turn": turn,
                        "event": "compaction_completion_verified",
                        "scope": "full",
                        "source": "raw_event",
                        "message": message,
                    }
                )
        elif started_tools or result_tools or text_done_orders:
            key = (turn, "full", "completion_unverified")
            if key not in seen:
                seen.add(key)
                events.append(
                    {
                        "turn": turn,
                        "event": "compaction_completion_unverified",
                        "scope": "full",
                        "source": "raw_event",
                        "message": "missing required compaction lifecycle evidence",
                    }
                )

    first_verified_completion_turn = min(verified_completion_turns) if verified_completion_turns else None
    diagnostics = {
        "detection_version": "tool_lifecycle_with_indicator_v2",
        "required_tool_names": required_tool_names,
        "required_scopes": required_scope_names,
        "observed_turns": sorted(observed_turns),
        "completion_verified": bool(verified_completion_turns),
        "completion_status": "verified" if verified_completion_turns else "not_verified",
        "first_verified_completion_turn": first_verified_completion_turn,
        "verified_completion_turns": verified_completion_turns,
        "tool_lifecycle_verified_turns": tool_lifecycle_verified_turns,
        "indicator_text_verified_turns": indicator_text_verified_turns,
        "verified_completion_methods_by_turn": {
            str(k): v for k, v in sorted(verified_completion_methods_by_turn.items(), key=lambda item: item[0])
        },
        "legacy_text_done_turns": sorted(text_done_turns),
        "legacy_text_failure_turns": sorted(text_failure_turns),
        "turn_diagnostics": completion_turn_diagnostics,
    }

    events.sort(key=lambda x: (int(x.get("turn") or 0), str(x.get("event") or ""), str(x.get("scope") or "")))
    return events, diagnostics


def _compaction_phase_for_turn(turn: int, first_compaction_done_turn: int | None) -> str:
    if not isinstance(first_compaction_done_turn, int):
        return "pre_compaction"
    return "post_compaction" if turn > first_compaction_done_turn else "pre_compaction"


def _classify_cause_from_verify_reason(reason: str) -> str:
    text = str(reason or "").strip().lower()
    if not text:
        return CAUSE_INSUFFICIENT_EVIDENCE
    if any(k in text for k in ["compression", "压缩", "壓縮", "summary", "compaction"]):
        return CAUSE_COMPRESSION_RELATED
    if any(k in text for k in ["task_switch", "换题", "切换", "改口径", "偏航"]):
        return CAUSE_TASK_SWITCH
    if any(k in text for k in ["override", "覆盖", "忽略", "instruction"]):
        return CAUSE_INSTRUCTION_OVERRIDE
    if any(k in text for k in ["insufficient", "证据不足", "不确定", "ambiguous"]):
        return CAUSE_INSUFFICIENT_EVIDENCE
    return CAUSE_UNKNOWN


def _extract_anchor_points_structured(anchor: dict[str, Any]) -> list[dict[str, str]]:
    structured = anchor.get("anchor_points_structured")
    if isinstance(structured, list):
        out: list[dict[str, str]] = []
        for idx, item in enumerate(structured, start=1):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            point_id = str(item.get("point_id") or f"p{idx}").strip() or f"p{idx}"
            tier_raw = str(item.get("tier") or item.get("retention_tier") or "").strip().lower()
            if tier_raw not in {"must_keep", "should_keep", "may_drop"}:
                tier_raw = "should_keep"
            out.append({"point_id": point_id, "text": text, "tier": tier_raw})
        if out:
            return out
    plain = anchor.get("anchor_points")
    if isinstance(plain, list):
        out_plain: list[dict[str, str]] = []
        for idx, item in enumerate(plain, start=1):
            text = str(item or "").strip()
            if not text:
                continue
            if idx == 1:
                tier = "must_keep"
            elif idx == 2:
                tier = "should_keep"
            else:
                tier = "may_drop"
            out_plain.append({"point_id": f"p{idx}", "text": text, "tier": tier})
        if out_plain:
            return out_plain
    return [
        {"point_id": "p1", "text": "campaign theme", "tier": "must_keep"},
        {"point_id": "p2", "text": "visual style", "tier": "should_keep"},
        {"point_id": "p3", "text": "cta wording", "tier": "may_drop"},
    ]


def _render_contract_text_template(template: str, anchor: dict[str, Any]) -> str:
    text = str(template or "")
    if not text:
        return ""
    points = _extract_anchor_points_structured(anchor)
    for idx, item in enumerate(points, start=1):
        text = text.replace(f"{{{{P{idx}}}}}", str(item.get("text") or ""))
    text = text.replace("{{ANCHOR_NAME}}", str(anchor.get("anchor_name") or ""))
    return text.strip()


def _anchor_points_with_tier(anchor: dict[str, Any]) -> list[dict[str, Any]]:
    points = _extract_anchor_points_structured(anchor)
    return [
        {
            "point_id": str(item.get("point_id") or ""),
            "text": str(item.get("text") or ""),
            "retention_tier": str(item.get("tier") or "should_keep"),
        }
        for item in points
    ]


def _score_tier_from_check(check: dict[str, Any]) -> dict[str, str]:
    miss_confirmed = bool(check.get("miss_confirmed"))
    verify = check.get("verify_judge") if isinstance(check.get("verify_judge"), dict) else {}
    reason = str(verify.get("reason") or "")
    judge_state = str(verify.get("judge_state") or "").strip().lower()
    cause_hint = str(verify.get("cause_hint") or "").strip().lower()
    if judge_state not in {JUDGE_REMEMBERED, JUDGE_AMBIGUOUS, JUDGE_FORGOTTEN}:
        if miss_confirmed:
            state = JUDGE_FORGOTTEN
        else:
            state = JUDGE_REMEMBERED
    else:
        state = judge_state
    if cause_hint:
        cause = cause_hint
    elif reason:
        cause = _classify_cause_from_verify_reason(reason)
    else:
        cause = CAUSE_UNKNOWN if state == JUDGE_FORGOTTEN else "none"
    return {"judge_state": state, "cause_hint": cause}


def _build_probe_timeline(
    probe_checks: list[dict[str, Any]],
    first_compaction_done_turn: int | None,
    anchor: dict[str, Any],
) -> list[dict[str, Any]]:
    anchor_points = _anchor_points_with_tier(anchor)
    timeline: list[dict[str, Any]] = []
    for check in probe_checks:
        turn = int(check.get("turn") or 0)
        phase = _compaction_phase_for_turn(turn, first_compaction_done_turn)
        scored = _score_tier_from_check(check)
        for ap in anchor_points:
            timeline.append(
                {
                    "turn": turn,
                    "kind": str(check.get("kind") or ""),
                    "probe_question": str(check.get("probe_question") or ""),
                    "retention_tier": str(ap.get("retention_tier") or "should_keep"),
                    "anchor_point_id": str(ap.get("point_id") or ""),
                    "anchor_point_text": str(ap.get("text") or ""),
                    "judge_state": scored["judge_state"],
                    "cause_hint": scored["cause_hint"],
                    "compaction_phase": phase,
                    "miss_confirmed": bool(check.get("miss_confirmed")),
                    "verify_reason": str((check.get("verify_judge") or {}).get("reason") or ""),
                }
            )
    return timeline


def _load_contract_anchor_profile() -> dict[str, Any]:
    dataset_path = AUTO_TEST_DIR / "datasets" / "probes" / "compression_retention_contract_v1.json"
    if not dataset_path.exists():
        return {}
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        return {}
    for item in profiles:
        if not isinstance(item, dict):
            continue
        anchor = item.get("anchor")
        if not isinstance(anchor, dict):
            continue
        points = anchor.get("points")
        if not isinstance(points, list) or not points:
            continue
        return {
            "profile_id": str(item.get("probe_id") or "").strip(),
            "anchor_name": str(anchor.get("anchor_name") or "").strip(),
            "anchor_points_structured": points,
            "memory_channel_expectation": str(item.get("memory_channel_expectation") or "").strip(),
            "plant_text_template": str(item.get("plant_text_template") or "").strip(),
            "probe_text_templates": item.get("probe_text_templates") if isinstance(item.get("probe_text_templates"), list) else [],
            "judge_rubric": item.get("judge_rubric") if isinstance(item.get("judge_rubric"), dict) else {},
            "failure_policy": item.get("failure_policy") if isinstance(item.get("failure_policy"), dict) else {},
        }
    return {}


def _merge_anchor_with_contract(default_anchor: dict[str, Any], contract_profile: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(default_anchor, dict):
        default_anchor = {}
    if not isinstance(contract_profile, dict) or not contract_profile:
        return default_anchor
    merged = dict(default_anchor)
    anchor_name = str(contract_profile.get("anchor_name") or "").strip()
    if anchor_name:
        merged["anchor_name"] = anchor_name
    structured = contract_profile.get("anchor_points_structured")
    if isinstance(structured, list) and structured:
        cleaned: list[dict[str, Any]] = []
        plain: list[str] = []
        for idx, item in enumerate(structured, start=1):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            tier = str(item.get("tier") or "").strip().lower()
            if tier not in {"must_keep", "should_keep", "may_drop"}:
                tier = "should_keep"
            point_id = str(item.get("point_id") or f"p{idx}").strip() or f"p{idx}"
            cleaned.append({"point_id": point_id, "text": text, "tier": tier})
            plain.append(text)
        if cleaned:
            merged["anchor_points_structured"] = cleaned
            merged["anchor_points"] = plain
    plant_template = str(contract_profile.get("plant_text_template") or "").strip()
    if plant_template:
        rendered_plant = _render_contract_text_template(plant_template, merged)
        if rendered_plant:
            merged["plant_message"] = rendered_plant
    probe_templates = contract_profile.get("probe_text_templates")
    if isinstance(probe_templates, list):
        rendered_probe_templates: list[str] = []
        for item in probe_templates:
            text = _render_contract_text_template(str(item or ""), merged)
            if text:
                rendered_probe_templates.append(text)
        if rendered_probe_templates:
            merged["probe_text_templates"] = rendered_probe_templates
    memory_channel_expectation = str(contract_profile.get("memory_channel_expectation") or "").strip()
    if memory_channel_expectation:
        merged["memory_channel_expectation"] = memory_channel_expectation
    judge_rubric = contract_profile.get("judge_rubric")
    if isinstance(judge_rubric, dict):
        merged["judge_rubric"] = judge_rubric
    if "failure_policy" in contract_profile:
        merged["failure_policy"] = contract_profile.get("failure_policy")
    if contract_profile.get("profile_id"):
        merged["contract_profile_id"] = contract_profile.get("profile_id")
    return merged


def _parse_failure_policy(raw: Any) -> dict[str, Any]:
    policy = raw if isinstance(raw, dict) else {}
    return {
        "must_keep_post_compaction_max_fail_rate": max(
            0.0,
            min(1.0, _as_float(policy.get("must_keep_post_compaction_max_fail_rate"), 0.0)),
        ),
        "should_keep_post_compaction_max_fail_rate": max(
            0.0,
            min(1.0, _as_float(policy.get("should_keep_post_compaction_max_fail_rate"), 0.2)),
        ),
        "may_drop_used_for_gating": _as_bool(policy.get("may_drop_used_for_gating"), False),
        "consecutive_probe_fail_threshold": max(1, min(8, _as_int(policy.get("consecutive_probe_fail_threshold"), 2))),
    }


def _extract_probe_tier(anchor: dict[str, Any], probe_index: int, default_tier: str) -> str:
    normalized_default = _normalize_importance_tier(default_tier)
    points = _extract_anchor_points_structured(anchor)
    if not points:
        return normalized_default
    idx = max(0, probe_index - 1) % len(points)
    row = points[idx]
    tier = str(row.get("tier") or "").strip().lower()
    if tier in IMPORTANCE_TIERS:
        return tier
    return normalized_default


def _pick_anchor_point_for_probe(anchor: dict[str, Any], probe_tier: str, probe_index: int) -> dict[str, str]:
    points = _extract_anchor_points_structured(anchor)
    if not points:
        return {"point_id": "p0", "text": "之前确认的方向", "tier": _normalize_importance_tier(probe_tier)}
    normalized_tier = _normalize_importance_tier(probe_tier)
    matched = [row for row in points if str(row.get("tier") or "").strip().lower() == normalized_tier]
    pool = matched if matched else points
    idx = max(0, probe_index - 1) % len(pool)
    row = pool[idx]
    return {
        "point_id": str(row.get("point_id") or f"p{idx + 1}").strip() or f"p{idx + 1}",
        "text": str(row.get("text") or "").strip() or "之前确认的方向",
        "tier": str(row.get("tier") or normalized_tier).strip().lower() or normalized_tier,
    }


def _build_hidden_probe_question(
    *,
    anchor: dict[str, Any],
    probe_tier: str,
    probe_index: int,
    turn_idx: int,
    variant_offset: int = 0,
) -> dict[str, Any]:
    point = _pick_anchor_point_for_probe(anchor, probe_tier, probe_index)
    point_text = str(point.get("text") or "之前确认的方向").strip()
    point_id = str(point.get("point_id") or "p0").strip() or "p0"
    anchor_name = str(anchor.get("anchor_name") or "需求方向").strip() or "需求方向"

    templates: list[tuple[str, str]] = [
        (
            "copy_refine",
            f"这版先别重做，沿用“{point_text}”这个方向，给我一条更口语的主文案（60字内），并配一张同主题图，文图一起交付。",
        ),
        (
            "title_pack",
            f"按“{point_text}”这个点，补3个更像真实用户会点开的标题，再给一张配图方向，文图都要自然一点。",
        ),
        (
            "ab_variant",
            f"把“{point_text}”融进去，做A/B两版文案（A更直接，B更有情绪感），并补一张可配这两版的主视觉。",
        ),
        (
            "final_touch",
            f"我准备定稿了，继续沿用“{point_text}”，给我一条可直接发布的短版文案，并同步给一张终稿配图。",
        ),
        (
            "style_lock",
            f"别改主线，按“{point_text}”继续推进，把文案口吻统一成更贴近{anchor_name}，同时把配图风格也统一。",
        ),
    ]

    slot = (max(0, probe_index - 1) + max(0, variant_offset) + max(0, turn_idx - 1)) % len(templates)
    intent, user_text = templates[slot]
    return {
        "probe_user_text": user_text,
        "probe_goal": f"hidden_probe::{intent}",
        "source": "hidden_template",
        "probe_tier": _normalize_importance_tier(probe_tier),
        "probe_mode": "hidden",
        "probe_signature": f"hidden:{intent}:{point_id}",
        "anchor_point_id": point_id,
        "anchor_point_text": point_text,
    }


def _is_gating_failure_candidate(
    *,
    check: dict[str, Any],
    tier: str,
    first_compaction_done_turn: int | None,
    compaction_completion_verified: bool,
    include_pre_compaction: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    normalized_tier = _normalize_importance_tier(tier)
    if normalized_tier not in IMPORTANCE_TIERS:
        return False, ["invalid_tier"]
    if normalized_tier == "may_drop":
        return False, ["may_drop_not_gated"]

    turn = _as_int(check.get("turn"), 0)
    verify = check.get("verify_judge") if isinstance(check.get("verify_judge"), dict) else {}
    judge_state = str(verify.get("judge_state") or "").strip().lower()
    cause_hint = str(verify.get("cause_hint") or "").strip().lower()
    miss_confirmed = _as_bool(check.get("miss_confirmed"), False)

    if not include_pre_compaction:
        if compaction_completion_verified and isinstance(first_compaction_done_turn, int):
            if turn <= first_compaction_done_turn:
                return False, ["pre_or_on_compaction_turn"]
        else:
            return False, ["compaction_completion_not_verified"]

    if judge_state != JUDGE_FORGOTTEN:
        return False, ["judge_not_forgotten"]
    if cause_hint != CAUSE_COMPRESSION_RELATED:
        return False, [f"cause_not_compression_related:{cause_hint or 'empty'}"]
    if not miss_confirmed:
        return False, ["miss_not_confirmed"]
    reasons.append("forgotten+compression_related+post_compaction")
    return True, reasons


def _build_session_gating_summary(
    *,
    probe_checks: list[dict[str, Any]],
    first_compaction_done_turn: int | None,
    compaction_completion_verified: bool,
    configured_tier: str,
    failure_policy: dict[str, Any],
) -> dict[str, Any]:
    normalized_tier = _normalize_importance_tier(configured_tier)
    include_pre_compaction = _as_bool(failure_policy.get("include_pre_compaction_for_gating"), False)
    consecutive_threshold = max(1, min(8, _as_int(failure_policy.get("consecutive_probe_fail_threshold"), 2)))
    max_fail_rate_by_tier = {
        "must_keep": max(0.0, min(1.0, _as_float(failure_policy.get("must_keep_post_compaction_max_fail_rate"), 0.0))),
        "should_keep": max(0.0, min(1.0, _as_float(failure_policy.get("should_keep_post_compaction_max_fail_rate"), 0.2))),
        "may_drop": 1.0 if not _as_bool(failure_policy.get("may_drop_used_for_gating"), False) else 0.0,
    }

    total = 0
    fail = 0
    fail_turns: list[int] = []
    fail_reasons: dict[str, int] = Counter()
    consecutive = 0
    max_consecutive = 0
    triggered = False
    trigger_turn: int | None = None
    trigger_reason = ""

    for idx, check in enumerate(probe_checks, start=1):
        if str(check.get("kind") or "") != "probe_check":
            continue
        tier = _normalize_importance_tier(check.get("probe_tier") or normalized_tier)
        if tier != normalized_tier:
            continue
        total += 1
        is_fail, reasons = _is_gating_failure_candidate(
            check=check,
            tier=tier,
            first_compaction_done_turn=first_compaction_done_turn,
            compaction_completion_verified=compaction_completion_verified,
            include_pre_compaction=include_pre_compaction,
        )
        if is_fail:
            fail += 1
            turn = _as_int(check.get("turn"), 0)
            fail_turns.append(turn)
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
            for reason in reasons:
                fail_reasons[reason] += 1
            if not triggered and consecutive >= consecutive_threshold:
                triggered = True
                trigger_turn = turn
                trigger_reason = f"consecutive_fail_threshold_reached:{consecutive_threshold}"
                break
        else:
            consecutive = 0
            for reason in reasons:
                fail_reasons[reason] += 1

    fail_rate = (fail / total) if total > 0 else 0.0
    allowed_rate = max_fail_rate_by_tier.get(normalized_tier, 0.0)
    threshold_triggered = False
    if not triggered and total > 0 and fail_rate > allowed_rate:
        threshold_triggered = True
        triggered = True
        trigger_turn = fail_turns[0] if fail_turns else None
        trigger_reason = f"fail_rate_exceeded:{round(fail_rate, 4)}>{round(allowed_rate, 4)}"

    return {
        "configured_tier": normalized_tier,
        "first_compaction_done_turn": first_compaction_done_turn,
        "compaction_completion_verified": compaction_completion_verified,
        "include_pre_compaction_for_gating": include_pre_compaction,
        "consecutive_probe_fail_threshold": consecutive_threshold,
        "total_checks_in_tier": total,
        "failure_checks_in_tier": fail,
        "failure_rate_in_tier": round(fail_rate, 4),
        "max_allowed_failure_rate_in_tier": round(allowed_rate, 4),
        "max_consecutive_failures": max_consecutive,
        "failure_turns": fail_turns,
        "reason_counter": dict(fail_reasons),
        "triggered": triggered,
        "trigger_turn": trigger_turn,
        "trigger_reason": trigger_reason,
        "trigger_by_rate_threshold": threshold_triggered,
    }


def _pick_contract_probe_template(anchor: dict[str, Any], probe_index: int, variant_offset: int = 0) -> tuple[str, int]:
    templates = anchor.get("probe_text_templates")
    if not isinstance(templates, list):
        return "", -1
    cleaned = [str(item or "").strip() for item in templates if str(item or "").strip()]
    if not cleaned:
        return "", -1
    idx = (max(0, probe_index - 1) + max(0, variant_offset)) % len(cleaned)
    return cleaned[idx], idx


def _build_probe_question_candidate(
    *,
    shared: SharedRuntime,
    anchor: dict[str, Any],
    turn_idx: int,
    probe_index: int,
    probe_tier: str,
    recent_history: str,
    variant_offset: int,
) -> dict[str, Any]:
    mode = _normalize_probe_mode(shared.args.probe_mode)
    normalized_tier = _normalize_importance_tier(probe_tier)
    if mode == "hidden":
        candidate = _build_hidden_probe_question(
            anchor=anchor,
            probe_tier=normalized_tier,
            probe_index=probe_index,
            turn_idx=turn_idx,
            variant_offset=variant_offset,
        )
        candidate["probe_mode"] = mode
        return candidate

    contract_probe_text, template_idx = _pick_contract_probe_template(anchor, probe_index, variant_offset=variant_offset)
    if contract_probe_text:
        return {
            "probe_user_text": contract_probe_text,
            "probe_goal": "contract_probe_template",
            "source": "contract_template",
            "probe_tier": normalized_tier,
            "probe_mode": mode,
            "probe_signature": f"explicit:contract_template:{template_idx}",
            "template_index": template_idx,
        }

    try:
        generated = generate_probe_question(
            cfg=shared.probe_llm_cfg,
            anchor=anchor,
            turn_idx=turn_idx,
            recent_history=recent_history,
        )
        user_text = str(generated.get("probe_user_text") or "").strip()
        if not user_text:
            raise RuntimeError("empty_probe_question")
        return {
            "probe_user_text": user_text,
            "probe_goal": str(generated.get("probe_goal") or "").strip(),
            "source": "llm_generated",
            "probe_tier": normalized_tier,
            "probe_mode": mode,
            "probe_signature": f"explicit:llm_generated:{normalized_tier}",
            "raw": generated.get("raw"),
        }
    except Exception as exc:
        user_text = build_fallback_probe_question(turn_idx)
        return {
            "probe_user_text": user_text,
            "probe_goal": "fallback_question_due_to_generation_error",
            "source": "fallback",
            "probe_tier": normalized_tier,
            "probe_mode": mode,
            "probe_signature": f"explicit:fallback:{normalized_tier}",
            "error": f"{exc.__class__.__name__}:{exc}",
        }


def _build_compression_effect_summary_from_sessions(
    session_payloads: list[dict[str, Any]],
    *,
    min_post_samples_per_tier: int,
) -> dict[str, Any]:
    sessions_total_input = len(session_payloads)
    sessions_included_verified = 0
    excluded_session_indices: list[int] = []
    tier_total = {"must_keep": 0, "should_keep": 0, "may_drop": 0}
    tier_forgotten_total = {"must_keep": 0, "should_keep": 0, "may_drop": 0}
    tier_post_total = {"must_keep": 0, "should_keep": 0, "may_drop": 0}
    tier_post_forgotten = {"must_keep": 0, "should_keep": 0, "may_drop": 0}
    cause_counter: Counter[str] = Counter()
    false_alarm = 0
    first_post_compaction_failures: list[int] = []

    for session in session_payloads:
        session_index = _as_int(session.get("session_index"), 0)
        compaction_completion = (
            session.get("compaction_completion") if isinstance(session.get("compaction_completion"), dict) else {}
        )
        completion_verified = _as_bool(compaction_completion.get("completion_verified"), False)
        if not completion_verified:
            excluded_session_indices.append(session_index)
            continue
        sessions_included_verified += 1
        st = session.get("stats")
        if not isinstance(st, dict):
            continue
        timeline = st.get("probe_timeline")
        if not isinstance(timeline, list):
            continue
        first_post_turn: int | None = None
        for item in timeline:
            if not isinstance(item, dict):
                continue
            tier = str(item.get("retention_tier") or "")
            if tier not in tier_total:
                continue
            state = str(item.get("judge_state") or "")
            cause = str(item.get("cause_hint") or "")
            phase = str(item.get("compaction_phase") or "")
            tier_total[tier] += 1
            if state == "forgotten":
                tier_forgotten_total[tier] += 1
                if cause:
                    cause_counter[cause] += 1
                    if cause in {CAUSE_TASK_SWITCH, CAUSE_INSTRUCTION_OVERRIDE, CAUSE_INSUFFICIENT_EVIDENCE}:
                        false_alarm += 1
                if phase == "post_compaction":
                    tier_post_forgotten[tier] += 1
                    turn = int(item.get("turn") or 0)
                    if (first_post_turn is None) or (0 < turn < first_post_turn):
                        first_post_turn = turn
            if phase == "post_compaction":
                tier_post_total[tier] += 1
        if isinstance(first_post_turn, int) and first_post_turn > 0:
            first_post_compaction_failures.append(first_post_turn)

    def _rate(num: int, den: int) -> float:
        if den <= 0:
            return 0.0
        return round(num / den, 4)

    retention_rate_by_tier = {
        tier: _rate(tier_total[tier] - tier_forgotten_total[tier], tier_total[tier]) for tier in tier_total
    }
    post_compaction_delta_by_tier = {
        tier: _rate(tier_post_forgotten[tier], tier_post_total[tier]) for tier in tier_total
    }
    total_forgotten = sum(tier_forgotten_total.values())
    false_alarm_rate = _rate(false_alarm, total_forgotten)
    first_post_compaction_failure_turn = _build_stats(first_post_compaction_failures)
    required_samples = max(1, int(min_post_samples_per_tier))
    sample_gate_by_tier: dict[str, Any] = {}
    for tier, post_count in tier_post_total.items():
        sufficient = post_count >= required_samples
        sample_gate_by_tier[tier] = {
            "status": "sufficient" if sufficient else "insufficient",
            "post_sample_count": post_count,
            "min_required": required_samples,
            "shortfall": max(0, required_samples - post_count),
        }
    insufficient_tiers = [tier for tier, row in sample_gate_by_tier.items() if row.get("status") == "insufficient"]
    sample_gate_overall = "sufficient" if not insufficient_tiers else "insufficient"
    excluded_session_indices = sorted({idx for idx in excluded_session_indices if idx > 0})
    compaction_completion_rate = _rate(sessions_included_verified, sessions_total_input)

    return {
        "sessions_total": sessions_included_verified,
        "sessions_total_input": sessions_total_input,
        "sessions_included_verified_compaction": sessions_included_verified,
        "sessions_excluded_unverified_compaction": len(excluded_session_indices),
        "excluded_session_indices_unverified_compaction": excluded_session_indices,
        "compaction_completion_rate": compaction_completion_rate,
        "min_post_samples_per_tier": required_samples,
        "post_compaction_sample_gate": {
            "overall_status": sample_gate_overall,
            "insufficient_tiers": insufficient_tiers,
            "status_by_tier": sample_gate_by_tier,
        },
        "retention_rate_by_tier": retention_rate_by_tier,
        "post_compaction_delta_by_tier": post_compaction_delta_by_tier,
        "false_alarm_rate": false_alarm_rate,
        "first_post_compaction_failure_turn": first_post_compaction_failure_turn,
        "cause_distribution": {k: v for k, v in sorted(cause_counter.items())},
        "tier_sample_count": tier_total,
        "tier_forgotten_count": tier_forgotten_total,
        "tier_post_sample_count": tier_post_total,
        "tier_post_forgotten_count": tier_post_forgotten,
    }


def _summarize_text(text: str, max_len: int = 160) -> str:
    raw = str(text or "").strip().replace("\n", " ")
    return raw if len(raw) <= max_len else raw[: max_len - 3] + "..."


def _render_recent_history(turn_results: list[dict[str, Any]], max_turns: int = 5) -> str:
    if not turn_results:
        return "(empty)"
    lines: list[str] = []
    for item in turn_results[-max_turns:]:
        turn_no = int(item.get("turn", 0))
        lines.append(f"Turn {turn_no}")
        lines.append(f"User: {str(item.get('user_text') or '').strip()}")
        lines.append(f"Assistant: {str(item.get('assistant_text') or '').strip()}")
        lines.append("")
    return "\n".join(lines).strip()


def _build_conversation_window(turn_results: list[dict[str, Any]], current_turn_idx: int, max_turns: int = 8) -> list[dict[str, Any]]:
    window: list[dict[str, Any]] = []
    for item in turn_results[-max_turns:]:
        turn = int(item.get("turn", 0))
        window.append(
            {
                "turn": turn,
                "turn_kind": str(item.get("turn_kind") or ""),
                "user_text": str(item.get("user_text") or ""),
                "assistant_text": str(item.get("assistant_text") or ""),
            }
        )
    if not window or int(window[-1].get("turn", 0)) != int(current_turn_idx):
        window.append(
            {
                "turn": int(current_turn_idx),
                "turn_kind": "probe_check",
                "user_text": "",
                "assistant_text": "",
            }
        )
    return window


def _write_dialogue_md(path: Path, turn_results: list[dict[str, Any]]) -> None:
    lines = ["# 用户视角对话记录", ""]
    for item in turn_results:
        turn = int(item.get("turn", 0))
        kind = str(item.get("turn_kind") or "")
        user_text = str(item.get("user_text") or "").strip() or "(empty)"
        assistant_text = str(item.get("assistant_text") or "").strip() or "(empty)"
        run_error = str(item.get("run_error") or "").strip()
        lines.extend(
            [
                f"## Turn {turn}",
                "",
                f"- turn_kind: `{kind}`",
                f"- run_error: `{run_error}`",
                "",
                f"**User**: {user_text}",
                "",
                f"**Assistant**: {assistant_text}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_distribution_svg(path: Path, values: list[int], title: str) -> dict[str, Any]:
    counts = Counter(values)
    if not counts:
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="420">'
            '<rect x="0" y="0" width="1000" height="420" fill="#f7f7f9"/>'
            f'<text x="40" y="60" font-size="24" fill="#222">{title}</text>'
            '<text x="40" y="110" font-size="18" fill="#555">No failure samples after warmup sessions.</text>'
            "</svg>"
        )
        path.write_text(svg + "\n", encoding="utf-8")
    else:
        xs = sorted(counts.keys())
        ys = [counts[x] for x in xs]
        width = 1000
        height = 520
        margin_l = 80
        margin_r = 40
        margin_t = 80
        margin_b = 90
        chart_w = width - margin_l - margin_r
        chart_h = height - margin_t - margin_b
        ymax = max(ys)
        bar_gap = 18
        bar_w = max(18, int((chart_w - bar_gap * (len(xs) + 1)) / max(1, len(xs))))
        parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect x="0" y="0" width="100%" height="100%" fill="#f8fafc"/>',
            f'<text x="40" y="42" font-size="22" fill="#111">{title}</text>',
            '<line x1="80" y1="430" x2="960" y2="430" stroke="#94a3b8" stroke-width="1.2"/>',
            '<line x1="80" y1="80" x2="80" y2="430" stroke="#94a3b8" stroke-width="1.2"/>',
        ]
        for idx, (xv, yv) in enumerate(zip(xs, ys)):
            x = margin_l + bar_gap + idx * (bar_w + bar_gap)
            h = int((yv / ymax) * (chart_h - 10)) if ymax > 0 else 0
            y = margin_t + chart_h - h
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="#3b82f6" rx="4" ry="4"/>')
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 8}" font-size="14" text-anchor="middle" fill="#1f2937">{yv}</text>'
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{margin_t + chart_h + 24}" font-size="13" text-anchor="middle" fill="#334155">{xv}</text>'
            )
        parts.extend(
            [
                '<text x="500" y="500" font-size="14" text-anchor="middle" fill="#475569">effective failure turn (same as raw)</text>',
                "</svg>",
            ]
        )
        path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return {
        "chart_type": "svg",
        "chart_path": path.as_posix(),
        "fallback_svg_path": path.as_posix(),
        "error": "",
    }


def _build_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "distribution": {}}
    counts = Counter(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 3),
        "median": statistics.median(values),
        "distribution": {str(k): v for k, v in sorted(counts.items())},
    }


def _classify_run_error_bucket(run_error: str) -> str:
    text = str(run_error or "").strip().lower()
    if not text:
        return "none"
    if ("401" in text and "unauthorized" in text) or "auth_401_unauthorized" in text:
        return "auth_401"
    if "127.0.0.1" in text and "7890" in text and ("timed out" in text or "timeout" in text):
        return "proxy_timeout"
    if "turn_timeout_" in text:
        return "turn_timeout"
    if "stream_closed_before_run_end" in text:
        return "stream_closed"
    if "502" in text or "bad gateway" in text:
        return "backend_502"
    if "503" in text:
        return "backend_503"
    if "504" in text:
        return "backend_504"
    if "connectionerror" in text or "request_exception" in text or "execute_turn_exception" in text:
        return "transport_error"
    return "other"


def _build_session_health_metrics(
    turn_results: list[dict[str, Any]],
    compaction_events: list[dict[str, Any]],
    compaction_completion: dict[str, Any],
) -> dict[str, Any]:
    turns_total = len(turn_results)
    run_error_turns: list[int] = []
    timeout_turns: list[int] = []
    stream_closed_turns: list[int] = []
    callback_round_turns: list[int] = []
    multi_request_turns: list[int] = []
    run_end_inferred_turns: list[int] = []
    terminal_missing_turns: list[int] = []
    terminal_reconciled_turns: list[int] = []
    auto_cancel_attempt_turns: list[int] = []
    auto_cancel_success_turns: list[int] = []
    auto_cancel_failed_turns: list[int] = []
    run_error_bucket_counts: Counter[str] = Counter()
    run_error_bucket_turns: dict[str, list[int]] = {}

    for tr in turn_results:
        turn = _as_int(tr.get("turn"), 0)
        run_error = str(tr.get("run_error") or "")
        if run_error:
            run_error_turns.append(turn)
            bucket = _classify_run_error_bucket(run_error)
            run_error_bucket_counts[bucket] += 1
            turn_list = run_error_bucket_turns.get(bucket)
            if not isinstance(turn_list, list):
                turn_list = []
                run_error_bucket_turns[bucket] = turn_list
            if turn > 0:
                turn_list.append(turn)
            if "TURN_TIMEOUT_" in run_error:
                timeout_turns.append(turn)
            if "STREAM_CLOSED_BEFORE_RUN_END" in run_error:
                stream_closed_turns.append(turn)
        if _as_int(tr.get("callback_rounds"), 0) > 0:
            callback_round_turns.append(turn)
        request_ids = tr.get("request_ids")
        if isinstance(request_ids, list) and len(request_ids) > 1:
            multi_request_turns.append(turn)
        if _as_bool(tr.get("run_end_inferred"), False):
            run_end_inferred_turns.append(turn)
        if _as_bool(tr.get("terminal_missing"), False):
            terminal_missing_turns.append(turn)
        terminal_reconcile = tr.get("terminal_reconcile")
        if isinstance(terminal_reconcile, dict) and _as_bool(terminal_reconcile.get("terminal_found"), False):
            terminal_reconciled_turns.append(turn)
        auto_cancel = tr.get("auto_cancel")
        if isinstance(auto_cancel, dict) and _as_bool(auto_cancel.get("attempted"), False):
            auto_cancel_attempt_turns.append(turn)
            if _as_bool(auto_cancel.get("ok"), False):
                auto_cancel_success_turns.append(turn)
            else:
                auto_cancel_failed_turns.append(turn)

    compaction_events_observed = any(
        str(e.get("event") or "")
        in {"compaction_start", "compaction_done", "compaction_completion_verified", "compaction_completion_unverified"}
        for e in compaction_events
        if isinstance(e, dict)
    )
    first_verified_turn = _as_int(compaction_completion.get("first_verified_completion_turn"), 0)

    return {
        "turns_total": turns_total,
        "run_error_turns": sorted({t for t in run_error_turns if t > 0}),
        "run_error_bucket_counts": {k: v for k, v in sorted(run_error_bucket_counts.items())},
        "run_error_bucket_turns": {
            k: sorted({t for t in turns if t > 0})
            for k, turns in sorted(run_error_bucket_turns.items())
        },
        "timeout_turns": sorted({t for t in timeout_turns if t > 0}),
        "stream_closed_turns": sorted({t for t in stream_closed_turns if t > 0}),
        "callback_round_turns": sorted({t for t in callback_round_turns if t > 0}),
        "multi_request_turns": sorted({t for t in multi_request_turns if t > 0}),
        "run_end_inferred_turns": sorted({t for t in run_end_inferred_turns if t > 0}),
        "terminal_missing_turns": sorted({t for t in terminal_missing_turns if t > 0}),
        "terminal_reconciled_turns": sorted({t for t in terminal_reconciled_turns if t > 0}),
        "auto_cancel_attempt_turns": sorted({t for t in auto_cancel_attempt_turns if t > 0}),
        "auto_cancel_success_turns": sorted({t for t in auto_cancel_success_turns if t > 0}),
        "auto_cancel_failed_turns": sorted({t for t in auto_cancel_failed_turns if t > 0}),
        "has_run_error_turn": bool(run_error_turns),
        "has_timeout_turn": bool(timeout_turns),
        "has_stream_closed_turn": bool(stream_closed_turns),
        "has_callback_round_turn": bool(callback_round_turns),
        "has_multi_request_turn": bool(multi_request_turns),
        "has_run_end_inferred_turn": bool(run_end_inferred_turns),
        "has_terminal_missing_turn": bool(terminal_missing_turns),
        "has_terminal_reconciled_turn": bool(terminal_reconciled_turns),
        "has_auto_cancel_attempt_turn": bool(auto_cancel_attempt_turns),
        "has_auto_cancel_failed_turn": bool(auto_cancel_failed_turns),
        "compaction_events_observed": compaction_events_observed,
        "compaction_completion_verified": _as_bool(compaction_completion.get("completion_verified"), False),
        "compaction_first_verified_turn": first_verified_turn if first_verified_turn > 0 else None,
    }


def _build_pipeline_health_summary(session_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    sessions_total = len(session_payloads)
    end_reason_counter: Counter[str] = Counter()
    run_error_bucket_counter: Counter[str] = Counter()
    timeout_sessions: list[int] = []
    stream_closed_sessions: list[int] = []
    run_error_sessions: list[int] = []
    terminal_missing_sessions: list[int] = []
    terminal_reconciled_sessions: list[int] = []
    auto_cancel_attempt_sessions: list[int] = []
    auto_cancel_failed_sessions: list[int] = []
    callback_sessions: list[int] = []
    multi_request_sessions: list[int] = []
    compaction_observed_sessions: list[int] = []
    compaction_verified_sessions: list[int] = []
    selfcheck_retry_sessions: list[int] = []

    turns_total = 0
    turns_with_run_error = 0
    turns_with_timeout = 0
    turns_with_stream_closed = 0
    turns_with_callback = 0
    turns_with_multi_request = 0
    turns_with_terminal_missing = 0
    turns_with_terminal_reconciled = 0
    turns_with_auto_cancel_attempt = 0
    turns_with_auto_cancel_failed = 0

    for session in session_payloads:
        session_idx = _as_int(session.get("session_index"), 0)
        end_reason = str(session.get("end_reason") or "unknown")
        end_reason_counter[end_reason] += 1
        if "workspace_selfcheck_empty_after_retry" in end_reason:
            selfcheck_retry_sessions.append(session_idx)

        health = session.get("health") if isinstance(session.get("health"), dict) else {}
        turns_total += _as_int(health.get("turns_total"), 0)
        turns_with_run_error += len(health.get("run_error_turns") or [])
        turns_with_timeout += len(health.get("timeout_turns") or [])
        turns_with_stream_closed += len(health.get("stream_closed_turns") or [])
        turns_with_callback += len(health.get("callback_round_turns") or [])
        turns_with_multi_request += len(health.get("multi_request_turns") or [])
        turns_with_terminal_missing += len(health.get("terminal_missing_turns") or [])
        turns_with_terminal_reconciled += len(health.get("terminal_reconciled_turns") or [])
        turns_with_auto_cancel_attempt += len(health.get("auto_cancel_attempt_turns") or [])
        turns_with_auto_cancel_failed += len(health.get("auto_cancel_failed_turns") or [])
        bucket_counts = health.get("run_error_bucket_counts")
        if isinstance(bucket_counts, dict):
            for bucket, count in bucket_counts.items():
                run_error_bucket_counter[str(bucket or "other")] += _as_int(count, 0)

        if _as_bool(health.get("has_run_error_turn"), False):
            run_error_sessions.append(session_idx)
        if _as_bool(health.get("has_timeout_turn"), False) or ("session_wall_timeout" in end_reason):
            timeout_sessions.append(session_idx)
        if _as_bool(health.get("has_stream_closed_turn"), False):
            stream_closed_sessions.append(session_idx)
        if _as_bool(health.get("has_terminal_missing_turn"), False):
            terminal_missing_sessions.append(session_idx)
        if _as_bool(health.get("has_terminal_reconciled_turn"), False):
            terminal_reconciled_sessions.append(session_idx)
        if _as_bool(health.get("has_auto_cancel_attempt_turn"), False):
            auto_cancel_attempt_sessions.append(session_idx)
        if _as_bool(health.get("has_auto_cancel_failed_turn"), False):
            auto_cancel_failed_sessions.append(session_idx)
        if _as_bool(health.get("has_callback_round_turn"), False):
            callback_sessions.append(session_idx)
        if _as_bool(health.get("has_multi_request_turn"), False):
            multi_request_sessions.append(session_idx)
        if _as_bool(health.get("compaction_events_observed"), False):
            compaction_observed_sessions.append(session_idx)
        if _as_bool(health.get("compaction_completion_verified"), False):
            compaction_verified_sessions.append(session_idx)

    def _rate(num: int, den: int) -> float:
        if den <= 0:
            return 0.0
        return round(num / den, 4)

    return {
        "sessions_total": sessions_total,
        "end_reason_distribution": {k: v for k, v in sorted(end_reason_counter.items())},
        "run_error_bucket_counts": {k: v for k, v in sorted(run_error_bucket_counter.items())},
        "sessions_with_run_error": len(run_error_sessions),
        "sessions_with_timeout": len(timeout_sessions),
        "sessions_with_stream_closed": len(stream_closed_sessions),
        "sessions_with_terminal_missing": len(terminal_missing_sessions),
        "sessions_with_terminal_reconciled": len(terminal_reconciled_sessions),
        "sessions_with_auto_cancel_attempt": len(auto_cancel_attempt_sessions),
        "sessions_with_auto_cancel_failed": len(auto_cancel_failed_sessions),
        "sessions_with_callback_rounds": len(callback_sessions),
        "sessions_with_multi_request_turns": len(multi_request_sessions),
        "sessions_with_compaction_events": len(compaction_observed_sessions),
        "sessions_with_verified_compaction_completion": len(compaction_verified_sessions),
        "session_run_error_rate": _rate(len(run_error_sessions), sessions_total),
        "session_timeout_rate": _rate(len(timeout_sessions), sessions_total),
        "session_stream_closed_rate": _rate(len(stream_closed_sessions), sessions_total),
        "session_terminal_missing_rate": _rate(len(terminal_missing_sessions), sessions_total),
        "session_terminal_reconciled_rate": _rate(len(terminal_reconciled_sessions), sessions_total),
        "session_auto_cancel_attempt_rate": _rate(len(auto_cancel_attempt_sessions), sessions_total),
        "session_auto_cancel_failed_rate": _rate(len(auto_cancel_failed_sessions), sessions_total),
        "session_callback_round_rate": _rate(len(callback_sessions), sessions_total),
        "session_multi_request_rate": _rate(len(multi_request_sessions), sessions_total),
        "compaction_observed_rate": _rate(len(compaction_observed_sessions), sessions_total),
        "compaction_completion_verified_rate": _rate(len(compaction_verified_sessions), sessions_total),
        "selfcheck_retry_sessions": sorted({idx for idx in selfcheck_retry_sessions if idx > 0}),
        "run_error_session_indices": sorted({idx for idx in run_error_sessions if idx > 0}),
        "timeout_session_indices": sorted({idx for idx in timeout_sessions if idx > 0}),
        "stream_closed_session_indices": sorted({idx for idx in stream_closed_sessions if idx > 0}),
        "terminal_missing_session_indices": sorted({idx for idx in terminal_missing_sessions if idx > 0}),
        "terminal_reconciled_session_indices": sorted({idx for idx in terminal_reconciled_sessions if idx > 0}),
        "auto_cancel_attempt_session_indices": sorted({idx for idx in auto_cancel_attempt_sessions if idx > 0}),
        "auto_cancel_failed_session_indices": sorted({idx for idx in auto_cancel_failed_sessions if idx > 0}),
        "turns_total": turns_total,
        "turns_with_run_error": turns_with_run_error,
        "turns_with_timeout": turns_with_timeout,
        "turns_with_stream_closed": turns_with_stream_closed,
        "turns_with_terminal_missing": turns_with_terminal_missing,
        "turns_with_terminal_reconciled": turns_with_terminal_reconciled,
        "turns_with_auto_cancel_attempt": turns_with_auto_cancel_attempt,
        "turns_with_auto_cancel_failed": turns_with_auto_cancel_failed,
        "turns_with_callback_rounds": turns_with_callback,
        "turns_with_multi_request": turns_with_multi_request,
        "turn_run_error_rate": _rate(turns_with_run_error, turns_total),
        "turn_timeout_rate": _rate(turns_with_timeout, turns_total),
        "turn_stream_closed_rate": _rate(turns_with_stream_closed, turns_total),
        "turn_terminal_missing_rate": _rate(turns_with_terminal_missing, turns_total),
        "turn_terminal_reconciled_rate": _rate(turns_with_terminal_reconciled, turns_total),
        "turn_auto_cancel_attempt_rate": _rate(turns_with_auto_cancel_attempt, turns_total),
        "turn_auto_cancel_failed_rate": _rate(turns_with_auto_cancel_failed, turns_total),
        "turn_multi_request_rate": _rate(turns_with_multi_request, turns_total),
    }


def _build_probe_runtime_summary(session_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    sessions_total = len(session_payloads)
    total_scheduled = 0
    total_executed = 0
    total_regen_retries = 0
    total_skipped_cooldown = 0
    total_skipped_similarity = 0
    total_skipped_empty = 0
    total_suppressed = 0
    skipped_reason_counter: Counter[str] = Counter()
    probe_mode_counter: Counter[str] = Counter()

    for session in session_payloads:
        mode = _normalize_probe_mode(session.get("probe_mode"))
        probe_mode_counter[mode] += 1
        runtime = session.get("probe_runtime") if isinstance(session.get("probe_runtime"), dict) else {}
        total_scheduled += _as_int(runtime.get("scheduled_probe_turns"), 0)
        total_executed += _as_int(runtime.get("executed_probe_turns"), 0)
        total_regen_retries += _as_int(runtime.get("regen_retries"), 0)
        total_skipped_cooldown += _as_int(runtime.get("skipped_due_cooldown"), 0)
        total_skipped_similarity += _as_int(runtime.get("skipped_due_similarity"), 0)
        total_skipped_empty += _as_int(runtime.get("skipped_due_empty_candidate"), 0)
        suppressed_turns = runtime.get("suppressed_probe_turns")
        if isinstance(suppressed_turns, list):
            total_suppressed += len(suppressed_turns)
            for row in suppressed_turns:
                if not isinstance(row, dict):
                    skipped_reason_counter["unknown"] += 1
                    continue
                reason = str(row.get("reason") or "unknown")
                root = reason.split(":", 1)[0] if ":" in reason else reason
                skipped_reason_counter[root or "unknown"] += 1

    def _rate(num: int, den: int) -> float:
        if den <= 0:
            return 0.0
        return round(num / den, 4)

    return {
        "sessions_total": sessions_total,
        "probe_mode_distribution": dict(sorted(probe_mode_counter.items())),
        "scheduled_probe_turns_total": total_scheduled,
        "executed_probe_turns_total": total_executed,
        "suppressed_probe_turns_total": total_suppressed,
        "suppressed_probe_turn_rate": _rate(total_suppressed, total_scheduled),
        "executed_probe_turn_rate": _rate(total_executed, total_scheduled),
        "regen_retries_total": total_regen_retries,
        "skipped_due_cooldown": total_skipped_cooldown,
        "skipped_due_similarity": total_skipped_similarity,
        "skipped_due_empty_candidate": total_skipped_empty,
        "suppressed_reason_distribution": dict(sorted(skipped_reason_counter.items())),
    }


def _build_pipeline_health_report_md(path: Path, payload: dict[str, Any]) -> None:
    health = payload.get("pipeline_health_summary")
    if not isinstance(health, dict):
        health = {}
    lines = [
        "# Pipeline Health Summary",
        "",
        "## Session-Level",
        "",
        f"- sessions_total: `{health.get('sessions_total', 0)}`",
        f"- session_run_error_rate: `{health.get('session_run_error_rate', 0.0)}`",
        f"- session_timeout_rate: `{health.get('session_timeout_rate', 0.0)}`",
        f"- session_stream_closed_rate: `{health.get('session_stream_closed_rate', 0.0)}`",
        f"- session_terminal_missing_rate: `{health.get('session_terminal_missing_rate', 0.0)}`",
        f"- session_terminal_reconciled_rate: `{health.get('session_terminal_reconciled_rate', 0.0)}`",
        f"- session_auto_cancel_attempt_rate: `{health.get('session_auto_cancel_attempt_rate', 0.0)}`",
        f"- session_auto_cancel_failed_rate: `{health.get('session_auto_cancel_failed_rate', 0.0)}`",
        f"- session_multi_request_rate: `{health.get('session_multi_request_rate', 0.0)}`",
        f"- compaction_observed_rate: `{health.get('compaction_observed_rate', 0.0)}`",
        f"- compaction_completion_verified_rate: `{health.get('compaction_completion_verified_rate', 0.0)}`",
        "",
        "## Turn-Level",
        "",
        f"- turns_total: `{health.get('turns_total', 0)}`",
        f"- turns_with_run_error: `{health.get('turns_with_run_error', 0)}`",
        f"- turns_with_timeout: `{health.get('turns_with_timeout', 0)}`",
        f"- turns_with_stream_closed: `{health.get('turns_with_stream_closed', 0)}`",
        f"- turns_with_terminal_missing: `{health.get('turns_with_terminal_missing', 0)}`",
        f"- turns_with_terminal_reconciled: `{health.get('turns_with_terminal_reconciled', 0)}`",
        f"- turns_with_auto_cancel_attempt: `{health.get('turns_with_auto_cancel_attempt', 0)}`",
        f"- turns_with_auto_cancel_failed: `{health.get('turns_with_auto_cancel_failed', 0)}`",
        f"- turns_with_callback_rounds: `{health.get('turns_with_callback_rounds', 0)}`",
        f"- turns_with_multi_request: `{health.get('turns_with_multi_request', 0)}`",
        "",
        "## Distribution",
        "",
        f"- end_reason_distribution: `{json.dumps(health.get('end_reason_distribution', {}), ensure_ascii=False)}`",
        f"- run_error_bucket_counts: `{json.dumps(health.get('run_error_bucket_counts', {}), ensure_ascii=False)}`",
        f"- timeout_session_indices: `{json.dumps(health.get('timeout_session_indices', []), ensure_ascii=False)}`",
        f"- stream_closed_session_indices: `{json.dumps(health.get('stream_closed_session_indices', []), ensure_ascii=False)}`",
        f"- terminal_missing_session_indices: `{json.dumps(health.get('terminal_missing_session_indices', []), ensure_ascii=False)}`",
        f"- terminal_reconciled_session_indices: `{json.dumps(health.get('terminal_reconciled_session_indices', []), ensure_ascii=False)}`",
        f"- auto_cancel_failed_session_indices: `{json.dumps(health.get('auto_cancel_failed_session_indices', []), ensure_ascii=False)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_result_markdown(path: Path, payload: dict[str, Any]) -> None:
    cfg = payload.get("config", {})
    aggregate = payload.get("aggregate", {})
    chart = payload.get("chart", {})
    health = aggregate.get("pipeline_health_summary")
    if not isinstance(health, dict):
        health = {}
    effect = aggregate.get("compression_effect_summary")
    if not isinstance(effect, dict):
        effect = {}
    probe_runtime = aggregate.get("probe_runtime_summary")
    if not isinstance(probe_runtime, dict):
        probe_runtime = {}
    sessions = payload.get("sessions", [])
    lines = [
        "# Memory Compression Failure Scan",
        "",
        f"- runtime_env: `{cfg.get('runtime_env', '')}`",
        f"- sessions: `{cfg.get('sessions', 0)}`",
        f"- warmup_sessions_excluded: `{cfg.get('warmup_sessions', 0)}`",
        f"- hard_max_turns: `{cfg.get('hard_max_turns', 0)}`",
        f"- turn_timeout_sec: `{cfg.get('turn_timeout_sec', 0)}`",
        f"- session_wall_timeout_sec: `{cfg.get('session_wall_timeout_sec', 0)}`",
        f"- probe_interval: `{cfg.get('probe_interval', 0)}`",
        f"- probe_mode: `{cfg.get('probe_mode', 'hidden')}`",
        f"- probe_cooldown_turns: `{cfg.get('probe_cooldown_turns', 0)}`",
        f"- probe_similarity_threshold: `{cfg.get('probe_similarity_threshold', 0.0)}`",
        f"- probe_recent_window: `{cfg.get('probe_recent_window', 0)}`",
        f"- probe_regen_max_attempts: `{cfg.get('probe_regen_max_attempts', 0)}`",
        f"- focus_window: `{cfg.get('focus_window', 0)}`",
        f"- parallel_sessions: `{cfg.get('parallel_sessions', 1)}`",
        f"- importance_tier: `{cfg.get('importance_tier', 'should_keep')}`",
        f"- results_root: `{cfg.get('results_root', '')}`",
        "",
        "## Aggregate",
        "",
        f"- sessions_with_failure: `{aggregate.get('sessions_with_failure', 0)}`",
        f"- sessions_without_failure: `{aggregate.get('sessions_without_failure', 0)}`",
        f"- evaluated_sessions_for_chart: `{aggregate.get('evaluated_sessions_for_chart', 0)}`",
        f"- chart_samples: `{aggregate.get('chart_samples', 0)}`",
        f"- chart_type: `{chart.get('chart_type', '')}`",
        f"- chart_path: `{chart.get('chart_path', '')}`",
        f"- pipeline_health.compaction_completion_verified_rate: `{health.get('compaction_completion_verified_rate', 0.0)}`",
        f"- pipeline_health.session_timeout_rate: `{health.get('session_timeout_rate', 0.0)}`",
        f"- pipeline_health.session_terminal_missing_rate: `{health.get('session_terminal_missing_rate', 0.0)}`",
        f"- pipeline_health.session_auto_cancel_failed_rate: `{health.get('session_auto_cancel_failed_rate', 0.0)}`",
        f"- pipeline_health.run_error_bucket_counts: `{json.dumps(health.get('run_error_bucket_counts', {}), ensure_ascii=False)}`",
        f"- probe_runtime.executed_probe_turn_rate: `{probe_runtime.get('executed_probe_turn_rate', 0.0)}`",
        f"- probe_runtime.suppressed_probe_turn_rate: `{probe_runtime.get('suppressed_probe_turn_rate', 0.0)}`",
        f"- probe_runtime.suppressed_reason_distribution: `{json.dumps(probe_runtime.get('suppressed_reason_distribution', {}), ensure_ascii=False)}`",
        f"- compression_effect.sessions_included_verified: `{effect.get('sessions_included_verified_compaction', 0)}`",
        f"- compression_effect.sample_gate: `{json.dumps((effect.get('post_compaction_sample_gate') or {}), ensure_ascii=False)}`",
        "",
        "## Distribution (effective turn = raw turn)",
        "",
        f"- stats: `{json.dumps(aggregate.get('effective_turn_stats', {}), ensure_ascii=False)}`",
        "",
        "## Sessions",
        "",
        "| session | mode | focus_turn | first_failure_raw | first_failure_effective | turns_executed | end_reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sessions:
        lines.append(
            f"| {row.get('session_index')} | {row.get('scan_mode', '')} | "
            f"{row.get('focus_turn_target', '')} | {row.get('first_failure_turn_raw', '')} | "
            f"{row.get('first_failure_turn_effective', '')} | {row.get('turns_executed', 0)} | "
            f"{row.get('end_reason', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_llm_report(path: Path, payload: dict[str, Any]) -> None:
    sessions = payload.get("sessions", [])
    anchor = payload.get("anchor", {})
    lines = [
        "# Memory Failure LLM Report",
        "",
        "## Probe Anchor",
        "",
        f"- anchor_id: `{anchor.get('anchor_id', '')}`",
        f"- anchor_name: `{anchor.get('anchor_name', '')}`",
        "- anchor_points:",
    ]
    for p in anchor.get("anchor_points", []):
        lines.append(f"  - {p}")
    lines.extend(
        [
            "",
            "## Sessions",
            "",
            "| session | probe_mode | first_failure_raw | first_failure_effective | end_reason | probe_checks |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in sessions:
        lines.append(
            f"| {row.get('session_index')} | {row.get('probe_mode', 'hidden')} | {row.get('first_failure_turn_raw', '')} | "
            f"{row.get('first_failure_turn_effective', '')} | {row.get('end_reason', '')} | "
            f"{row.get('probe_checks_count', 0)} |"
        )
        probe_path = row.get("paths", {}).get("probe_checks", "")
        if probe_path:
            lines.append(f"  - probe_checks: `{probe_path}`")
        runtime = row.get("probe_runtime") if isinstance(row.get("probe_runtime"), dict) else {}
        if runtime:
            lines.append(
                "  - probe_runtime: "
                f"`executed={runtime.get('executed_probe_turns', 0)}, "
                f"suppressed={len(runtime.get('suppressed_probe_turns') or [])}, "
                f"retries={runtime.get('regen_retries', 0)}`"
            )
        diagnosis = row.get("failure_diagnosis")
        if isinstance(diagnosis, dict) and diagnosis:
            lines.append(f"  - loss_turn: `{diagnosis.get('turn')}`")
            lines.append(f"  - suspect_reason: `{diagnosis.get('suspect_reason')}`")
            lines.append(f"  - verify_reason: `{diagnosis.get('verify_reason')}`")
            lines.append(f"  - evidence_refs: `{diagnosis.get('evidence_refs')}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_compression_effect_report_md(path: Path, payload: dict[str, Any]) -> None:
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        aggregate = {}
    effect = aggregate.get("compression_effect_summary")
    if not isinstance(effect, dict):
        effect = {}
    contract_profile = payload.get("contract_profile")
    if not isinstance(contract_profile, dict):
        contract_profile = {}
    lines = [
        "# Compression Effect Summary",
        "",
        "## Core Metrics",
        "",
        f"- sessions_total_input: `{effect.get('sessions_total_input', 0)}`",
        f"- sessions_total: `{effect.get('sessions_total', 0)}`",
        f"- sessions_excluded_unverified_compaction: `{effect.get('sessions_excluded_unverified_compaction', 0)}`",
        f"- compaction_completion_rate: `{effect.get('compaction_completion_rate', 0.0)}`",
        f"- retention_rate_by_tier: `{json.dumps(effect.get('retention_rate_by_tier', {}), ensure_ascii=False)}`",
        f"- post_compaction_delta_by_tier: `{json.dumps(effect.get('post_compaction_delta_by_tier', {}), ensure_ascii=False)}`",
        f"- false_alarm_rate: `{effect.get('false_alarm_rate', 0.0)}`",
        f"- first_post_compaction_failure_turn: `{json.dumps(effect.get('first_post_compaction_failure_turn', {}), ensure_ascii=False)}`",
        "",
        "## Post-Compaction Sample Gate",
        "",
        f"- min_post_samples_per_tier: `{effect.get('min_post_samples_per_tier', 0)}`",
        f"- post_compaction_sample_gate: `{json.dumps(effect.get('post_compaction_sample_gate', {}), ensure_ascii=False)}`",
        "",
        "## Gating",
        "",
        f"- configured_importance_tier: `{payload.get('config', {}).get('importance_tier', 'should_keep')}`",
        "",
        "## Diagnosis",
        "",
        f"- cause_distribution: `{json.dumps(effect.get('cause_distribution', {}), ensure_ascii=False)}`",
        f"- tier_sample_count: `{json.dumps(effect.get('tier_sample_count', {}), ensure_ascii=False)}`",
        f"- tier_forgotten_count: `{json.dumps(effect.get('tier_forgotten_count', {}), ensure_ascii=False)}`",
        f"- tier_post_sample_count: `{json.dumps(effect.get('tier_post_sample_count', {}), ensure_ascii=False)}`",
        f"- tier_post_forgotten_count: `{json.dumps(effect.get('tier_post_forgotten_count', {}), ensure_ascii=False)}`",
        "",
        "## Contract",
        "",
        f"- contract_profile_id: `{contract_profile.get('profile_id', '')}`",
        f"- contract_anchor_name: `{contract_profile.get('anchor_name', '')}`",
        f"- memory_channel_expectation: `{contract_profile.get('memory_channel_expectation', '')}`",
        f"- failure_policy: `{json.dumps(contract_profile.get('failure_policy', {}), ensure_ascii=False)}`",
        "",
        "## Notes",
        "",
        "- `must_keep` 丢失应视作红线故障；`should_keep` 用于压缩退化监控；`may_drop` 仅用于减负观察。",
        "- `false_alarm_rate` 为非 compression_related 的遗忘比例，用于抑制业务换题误报。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_single_session(shared: SharedRuntime, session_idx: int, focus_turn: int | None) -> dict[str, Any]:
    session_dir = shared.sessions_root / f"session_{session_idx:02d}"
    run_data_dir = session_dir / "run_data"
    session_dir.mkdir(parents=True, exist_ok=True)
    run_data_dir.mkdir(parents=True, exist_ok=True)

    scan_mode = _build_scan_mode_label(focus_turn, shared.args.focus_window)
    probe_id = f"{PROBE_KIND_MEMORY_POINT}-{session_idx:02d}-{uuid.uuid4().hex[:6]}"
    probe_turns_executed: list[int] = []
    contract_profile = _load_contract_anchor_profile()
    anchor = _merge_anchor_with_contract(build_default_probe_anchor(), contract_profile)
    failure_policy = _parse_failure_policy(anchor.get("failure_policy"))

    print(
        f"[INFO] session_start index={session_idx} mode={scan_mode} "
        f"focus_turn={focus_turn} probe_mode={_normalize_probe_mode(shared.args.probe_mode)}"
    )

    try:
        generated_role: GeneratedRole = generate_role_for_session(shared.user_sim_cfg)
        print(f"[INFO] user_role session={session_idx} role_name={generated_role.role_name}")
    except Exception as exc:
        print(f"[WARN] user_role_fallback session={session_idx} reason={exc.__class__.__name__}")
        generated_role = GeneratedRole(
            role_name=f"memory-scan-user-{session_idx:02d}",
            role_profile=(
                "- role_name: memory-scan-user\n"
                "- identity: operation specialist\n"
                "- goal: iterate copy and image assets"
            ),
            role_json={
                "role_name": f"memory-scan-user-{session_idx:02d}",
                "identity": "operation specialist",
                "current_goal": "iterate copy and image assets",
            },
        )
    sim_state = UserSimulatorState(previous_response_id="")

    turn_results: list[dict[str, Any]] = []
    probe_checks: list[dict[str, Any]] = []
    first_failure_turn_raw: int | None = None
    failure_diagnosis: dict[str, Any] = {}
    create_trace_obj: dict[str, Any] = {}
    end_reason = "unknown"
    session_id = ""
    raw_events_path = session_dir / "raw_events.jsonl"
    latest_workspace_snapshot: dict[str, Any] | None = None
    probe_recent_texts: list[str] = []
    probe_last_turn_by_signature: dict[str, int] = {}
    probe_control_cfg = {
        "probe_mode": _normalize_probe_mode(shared.args.probe_mode),
        "probe_cooldown_turns": shared.args.probe_cooldown_turns,
        "probe_similarity_threshold": shared.args.probe_similarity_threshold,
        "probe_recent_window": shared.args.probe_recent_window,
        "probe_regen_max_attempts": shared.args.probe_regen_max_attempts,
    }
    probe_runtime_stats: dict[str, Any] = {
        "probe_mode": _normalize_probe_mode(shared.args.probe_mode),
        "scheduled_probe_turns": 0,
        "executed_probe_turns": 0,
        "regen_retries": 0,
        "skipped_due_cooldown": 0,
        "skipped_due_similarity": 0,
        "skipped_due_empty_candidate": 0,
        "suppressed_probe_turns": [],
    }

    with raw_events_path.open("w", encoding="utf-8") as raw_fp:
        session_id, create_trace = create_session(
            shared.cfg.base_url,
            shared.headers,
            persist_type=shared.persist_type,
            settings_json=shared.create_settings_json,
        )
        create_trace_obj = {
            "request_id": create_trace.request_id,
            "traceparent": create_trace.traceparent,
            "tracestate": create_trace.tracestate,
            "x_trace_id": create_trace.x_trace_id,
            "backend_trace_id": create_trace.backend_trace_id,
        }
        print(f"[INFO] session_created index={session_idx} session_id={session_id}")

        turn_idx = 1
        session_turn_cap = 40 if shared.args.hard_max_turns <= 0 else min(40, shared.args.hard_max_turns)
        session_started_at = time.time()
        session_wall_timeout_sec = max(0, _as_int(shared.args.session_wall_timeout_sec, 0))
        while True:
            if session_wall_timeout_sec > 0 and (time.time() - session_started_at) > session_wall_timeout_sec:
                end_reason = "session_wall_timeout"
                print(
                    "[WARN] session_wall_timeout "
                    f"session={session_idx} elapsed_sec={round(time.time() - session_started_at, 2)} "
                    f"limit_sec={session_wall_timeout_sec}"
                )
                break
            if turn_idx > session_turn_cap:
                end_reason = "hard_max_turns_reached"
                break

            probe_question_meta: dict[str, Any] = {}
            if turn_idx == 1:
                turn_kind = "session_start"
                user_text = str(shared.user_sim_cfg.first_user_message or "").strip() or DEFAULT_FIRST_USER_TEXT
            elif turn_idx == 2:
                user_text = str(anchor.get("plant_message") or "")
                turn_kind = "plant_probe"
            elif _should_probe_turn(
                turn_idx=turn_idx,
                coarse_interval=shared.args.probe_interval,
                focus_turn=focus_turn,
                focus_window=shared.args.focus_window,
            ):
                probe_runtime_stats["scheduled_probe_turns"] = _as_int(
                    probe_runtime_stats.get("scheduled_probe_turns"), 0
                ) + 1
                probe_check_index = len(probe_turns_executed) + 1
                probe_tier = _extract_probe_tier(anchor, probe_check_index, shared.args.importance_tier)
                candidate_selected: dict[str, Any] | None = None
                reject_reason = "candidate_empty"
                recent_history = _render_recent_history(turn_results)
                for attempt in range(0, shared.args.probe_regen_max_attempts + 1):
                    candidate_meta = _build_probe_question_candidate(
                        shared=shared,
                        anchor=anchor,
                        turn_idx=turn_idx,
                        probe_index=probe_check_index,
                        probe_tier=probe_tier,
                        recent_history=recent_history,
                        variant_offset=attempt,
                    )
                    candidate_text = str(candidate_meta.get("probe_user_text") or "").strip()
                    candidate_signature = str(candidate_meta.get("probe_signature") or "").strip()
                    accepted, reason, max_similarity = _judge_probe_candidate(
                        candidate_text=candidate_text,
                        candidate_signature=candidate_signature,
                        turn_idx=turn_idx,
                        recent_probe_texts=probe_recent_texts,
                        probe_last_turn_by_signature=probe_last_turn_by_signature,
                        similarity_threshold=shared.args.probe_similarity_threshold,
                        recent_window=shared.args.probe_recent_window,
                        cooldown_turns=shared.args.probe_cooldown_turns,
                    )
                    candidate_meta["probe_generation_attempt"] = attempt + 1
                    candidate_meta["probe_similarity_to_recent"] = round(max_similarity, 4)
                    if accepted:
                        candidate_selected = candidate_meta
                        break

                    reject_reason = reason or "candidate_rejected"
                    if reject_reason.startswith("cooldown_active"):
                        probe_runtime_stats["skipped_due_cooldown"] = _as_int(
                            probe_runtime_stats.get("skipped_due_cooldown"), 0
                        ) + 1
                    elif reject_reason.startswith("similarity_high"):
                        probe_runtime_stats["skipped_due_similarity"] = _as_int(
                            probe_runtime_stats.get("skipped_due_similarity"), 0
                        ) + 1
                    else:
                        probe_runtime_stats["skipped_due_empty_candidate"] = _as_int(
                            probe_runtime_stats.get("skipped_due_empty_candidate"), 0
                        ) + 1
                    if attempt < shared.args.probe_regen_max_attempts:
                        probe_runtime_stats["regen_retries"] = _as_int(probe_runtime_stats.get("regen_retries"), 0) + 1

                if candidate_selected:
                    turn_kind = "probe_check"
                    user_text = str(candidate_selected.get("probe_user_text") or "").strip()
                    probe_question_meta = candidate_selected
                    probe_turns_executed.append(turn_idx)
                    probe_runtime_stats["executed_probe_turns"] = _as_int(
                        probe_runtime_stats.get("executed_probe_turns"), 0
                    ) + 1
                    signature = str(candidate_selected.get("probe_signature") or "").strip()
                    if signature:
                        probe_last_turn_by_signature[signature] = turn_idx
                    if user_text:
                        probe_recent_texts.append(user_text)
                        keep_window = max(6, shared.args.probe_recent_window * 3)
                        if len(probe_recent_texts) > keep_window:
                            probe_recent_texts = probe_recent_texts[-keep_window:]
                else:
                    user_text, turn_kind, sim_state = _generate_filler_user_turn(
                        shared=shared,
                        sim_state=sim_state,
                        turn_results=turn_results,
                        generated_role=generated_role,
                        turn_idx=turn_idx,
                        latest_workspace_snapshot=latest_workspace_snapshot,
                        session_idx=session_idx,
                        session_id=session_id,
                    )
                    suppressed_turns = probe_runtime_stats.get("suppressed_probe_turns")
                    if isinstance(suppressed_turns, list):
                        suppressed_turns.append(
                            {
                                "turn": turn_idx,
                                "reason": reject_reason,
                                "probe_tier": probe_tier,
                                "probe_mode": _normalize_probe_mode(shared.args.probe_mode),
                            }
                        )
            else:
                user_text, turn_kind, sim_state = _generate_filler_user_turn(
                    shared=shared,
                    sim_state=sim_state,
                    turn_results=turn_results,
                    generated_role=generated_role,
                    turn_idx=turn_idx,
                    latest_workspace_snapshot=latest_workspace_snapshot,
                    session_idx=session_idx,
                    session_id=session_id,
                )

            try:
                result = execute_turn(
                    base_url=shared.cfg.base_url,
                    headers=shared.headers,
                    session_id=session_id,
                    persist_type=shared.persist_type,
                    exec_max_turns=shared.exec_max_turns,
                    run_settings=shared.run_settings,
                    user_text=user_text,
                    turn_idx=turn_idx,
                    raw_events_fp=raw_fp,
                )
            except Exception as exc:
                result = {
                    "turn": turn_idx,
                    "user_text": user_text,
                    "assistant_text": "",
                    "event_count": 0,
                    "run_end": False,
                    "run_error": f"EXECUTE_TURN_EXCEPTION:{exc.__class__.__name__}:{str(exc)}",
                    "terminal_event_type": "",
                    "terminal_event_seen": False,
                    "terminal_missing": False,
                    "terminal_reconcile": {
                        "attempted": False,
                        "status": "execute_turn_exception",
                        "terminal_found": False,
                    },
                    "workspace_event_paths": [],
                    "workspace_snapshot": {},
                    "duration_sec": 0.0,
                }
            result["turn_kind"] = turn_kind
            turn_results.append(result)
            snapshot = result.get("workspace_snapshot")
            latest_workspace_snapshot = snapshot if isinstance(snapshot, dict) else None

            if turn_kind in {"plant_probe", "probe_check"}:
                assistant_text = str(result.get("assistant_text") or "")
                suspect_obj: dict[str, Any] = {}
                verify_obj: dict[str, Any] = {}
                miss_confirmed = False

                if turn_kind == "plant_probe":
                    suspect_obj = {"loss_suspected": False, "reason_short": "plant_turn_no_loss_check"}
                    verify_obj = {"forgotten": False, "reason": "plant_turn_no_verify"}
                else:
                    probe_question = str(probe_question_meta.get("probe_user_text") or user_text)
                    try:
                        suspect_obj = judge_loss_suspected(
                            cfg=shared.probe_llm_cfg,
                            anchor=anchor,
                            probe_question=probe_question,
                            assistant_reply=assistant_text,
                        )
                    except Exception as exc:
                        suspect_obj = {
                            "loss_suspected": False,
                            "confidence_0_1": 0.0,
                            "remembered_points": [],
                            "missing_points": [],
                            "reason_short": f"suspect_judge_error:{exc.__class__.__name__}:{exc}",
                        }

                    if bool(suspect_obj.get("loss_suspected")):
                        try:
                            verify_obj = verify_loss_confirmed(
                                cfg=shared.probe_llm_cfg,
                                anchor=anchor,
                                probe_question=probe_question,
                                assistant_reply=assistant_text,
                                suspect_reason=str(suspect_obj.get("reason_short") or ""),
                                conversation_window=_build_conversation_window(turn_results, turn_idx),
                            )
                        except Exception as exc:
                            verify_obj = {
                                "forgotten": False,
                                "confidence_0_1": 0.0,
                                "score_0_100": 0.0,
                                "reason": f"verify_judge_error:{exc.__class__.__name__}:{exc}",
                                "evidence_refs": [],
                            }
                    else:
                        verify_obj = {
                            "forgotten": False,
                            "confidence_0_1": 0.0,
                            "score_0_100": 0.0,
                            "reason": "suspect_false_skip_verify",
                            "evidence_refs": [],
                        }
                    miss_confirmed = bool(verify_obj.get("forgotten"))

                probe_checks.append(
                    {
                        "turn": turn_idx,
                        "kind": turn_kind,
                        "probe_question": str(probe_question_meta.get("probe_user_text") or user_text),
                        "probe_goal": str(probe_question_meta.get("probe_goal") or ""),
                        "probe_source": str(probe_question_meta.get("source") or ""),
                        "probe_mode": _normalize_probe_mode(probe_question_meta.get("probe_mode") or shared.args.probe_mode),
                        "probe_signature": str(probe_question_meta.get("probe_signature") or ""),
                        "probe_generation_attempt": _as_int(probe_question_meta.get("probe_generation_attempt"), 1),
                        "probe_similarity_to_recent": round(
                            max(0.0, min(1.0, _as_float(probe_question_meta.get("probe_similarity_to_recent"), 0.0))),
                            4,
                        ),
                        "probe_tier": _normalize_importance_tier(
                            probe_question_meta.get("probe_tier") or shared.args.importance_tier
                        ),
                        "probe_anchor_point_id": str(probe_question_meta.get("anchor_point_id") or ""),
                        "probe_anchor_point_text": str(probe_question_meta.get("anchor_point_text") or ""),
                        "suspect_judge": suspect_obj,
                        "verify_judge": verify_obj,
                        "miss_confirmed": miss_confirmed,
                        "assistant_excerpt": _summarize_text(assistant_text),
                    }
                )

                if turn_kind == "probe_check" and miss_confirmed and not first_failure_turn_raw:
                    first_failure_turn_raw = turn_idx
                    failure_diagnosis = {
                        "turn": turn_idx,
                        "suspect_reason": str(suspect_obj.get("reason_short") or ""),
                        "verify_reason": str(verify_obj.get("reason") or ""),
                        "evidence_refs": verify_obj.get("evidence_refs"),
                    }

            if result.get("run_error"):
                if _as_bool(os.getenv("AUTO_TEST_AUTO_CANCEL_ON_RUN_ERROR", "true"), True):
                    cancel_info = cancel_session(
                        base_url=shared.cfg.base_url,
                        headers=shared.headers,
                        session_id=session_id,
                    )
                    result["auto_cancel"] = cancel_info
                    print(
                        "[INFO] turn_error_auto_cancel "
                        f"session={session_idx} turn={turn_idx} "
                        f"status={cancel_info.get('status')} ok={cancel_info.get('ok')}"
                    )
                end_reason = "run_error"
                print(
                    "[WARN] turn_error "
                    f"session={session_idx} turn={turn_idx} run_error={result.get('run_error')}"
                )
                break

            if session_wall_timeout_sec > 0 and (time.time() - session_started_at) > session_wall_timeout_sec:
                end_reason = "session_wall_timeout"
                print(
                    "[WARN] session_wall_timeout "
                    f"session={session_idx} elapsed_sec={round(time.time() - session_started_at, 2)} "
                    f"limit_sec={session_wall_timeout_sec} after_turn={turn_idx}"
                )
                break

            turn_idx += 1

        if end_reason == "unknown":
            end_reason = "session_loop_ended"

    workspace_export = export_workspace_view(
        base_url=shared.cfg.base_url,
        headers=shared.headers,
        session_id=session_id,
        raw_events_path=raw_events_path,
        result_dir=session_dir,
        dotai_base_url=shared.dotai_base_url,
        user_id=shared.cfg.uid,
    )

    _write_dialogue_md(session_dir / "dialogue.md", turn_results)
    (run_data_dir / "turn_results.json").write_text(
        json.dumps(turn_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_data_dir / "probe_checks.json").write_text(
        json.dumps(
            {
                "probe_id": probe_id,
                "probe_kind": PROBE_KIND_MEMORY_POINT,
                "scan_mode": scan_mode,
                "focus_turn": focus_turn,
                "probe_mode": _normalize_probe_mode(shared.args.probe_mode),
                "probe_control": probe_control_cfg,
                "anchor": anchor,
                "probe_turns_executed": probe_turns_executed,
                "probe_runtime": probe_runtime_stats,
                "checks": probe_checks,
                "first_failure_turn_raw": first_failure_turn_raw,
                "end_reason": end_reason,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    compaction_events, compaction_completion = _detect_compaction_events(raw_events_path, turn_results)
    first_compaction_done_turn_raw = _as_int(compaction_completion.get("first_verified_completion_turn"), 0)
    first_compaction_done_turn = first_compaction_done_turn_raw if first_compaction_done_turn_raw > 0 else None
    compaction_completion_verified = _as_bool(compaction_completion.get("completion_verified"), False)
    session_gating = _build_session_gating_summary(
        probe_checks=probe_checks,
        first_compaction_done_turn=first_compaction_done_turn,
        compaction_completion_verified=compaction_completion_verified,
        configured_tier=shared.args.importance_tier,
        failure_policy=failure_policy,
    )
    if _as_bool(session_gating.get("triggered"), False):
        if end_reason in {"unknown", "session_loop_ended", "hard_max_turns_reached"}:
            end_reason = "probe_policy_triggered"
        trigger_turn = _as_int(session_gating.get("trigger_turn"), 0)
        if trigger_turn > 0 and (
            not isinstance(first_failure_turn_raw, int) or trigger_turn < int(first_failure_turn_raw)
        ):
            first_failure_turn_raw = trigger_turn
        if not failure_diagnosis:
            failure_diagnosis = {
                "turn": session_gating.get("trigger_turn"),
                "suspect_reason": "policy_gating_triggered",
                "verify_reason": str(session_gating.get("trigger_reason") or ""),
                "evidence_refs": [],
            }

    # Rewrite probe_checks payload to include final gating result aligned with compaction timeline.
    (run_data_dir / "probe_checks.json").write_text(
        json.dumps(
            {
                "probe_id": probe_id,
                "probe_kind": PROBE_KIND_MEMORY_POINT,
                "scan_mode": scan_mode,
                "focus_turn": focus_turn,
                "probe_mode": _normalize_probe_mode(shared.args.probe_mode),
                "probe_control": probe_control_cfg,
                "importance_tier": shared.args.importance_tier,
                "anchor": anchor,
                "failure_policy": failure_policy,
                "probe_turns_executed": probe_turns_executed,
                "probe_runtime": probe_runtime_stats,
                "checks": probe_checks,
                "first_failure_turn_raw": first_failure_turn_raw,
                "first_compaction_done_turn": first_compaction_done_turn,
                "compaction_completion": compaction_completion,
                "session_gating": session_gating,
                "end_reason": end_reason,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    probe_timeline = _build_probe_timeline(probe_checks, first_compaction_done_turn, anchor)
    session_health = _build_session_health_metrics(turn_results, compaction_events, compaction_completion)
    (run_data_dir / "compaction_events.json").write_text(
        json.dumps(
            {
                "session_index": session_idx,
                "session_id": session_id,
                "events": compaction_events,
                "compaction_completion": compaction_completion,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_data_dir / "probe_timeline.json").write_text(
        json.dumps(
            {
                "session_index": session_idx,
                "session_id": session_id,
                "timeline": probe_timeline,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    first_failure_turn_effective = first_failure_turn_raw if isinstance(first_failure_turn_raw, int) else None

    session_payload = {
        "session_index": session_idx,
        "session_id": session_id,
        "probe_id": probe_id,
        "scan_mode": scan_mode,
        "focus_turn_target": focus_turn,
        "generated_role_name": generated_role.role_name,
        "probe_mode": _normalize_probe_mode(shared.args.probe_mode),
        "probe_control": probe_control_cfg,
        "probe_runtime": probe_runtime_stats,
        "probe_turns_executed": probe_turns_executed,
        "probe_checks_count": len(probe_checks),
        "first_failure_turn_raw": first_failure_turn_raw,
        "first_failure_turn_effective": first_failure_turn_effective,
        "turns_executed": len(turn_results),
        "end_reason": end_reason,
        "failure_diagnosis": failure_diagnosis,
        "anchor": anchor,
        "importance_tier": shared.args.importance_tier,
        "failure_policy": failure_policy,
        "session_gating": session_gating,
        "compaction_completion": compaction_completion,
        "health": session_health,
        "stats": {
            "compaction_events": compaction_events,
            "probe_timeline": probe_timeline,
            "compaction_completion": compaction_completion,
        },
        "user_simulator": {
            "mode": shared.user_sim_cfg.mode,
            "model": shared.user_sim_cfg.model,
            "capability_mode": shared.user_sim_cfg.capability_mode,
        },
        "create_trace": create_trace_obj,
        "paths": {
            "raw_events": raw_events_path.relative_to(shared.sessions_root.parent).as_posix(),
            "dialogue": (session_dir / "dialogue.md").relative_to(shared.sessions_root.parent).as_posix(),
            "turn_results": (run_data_dir / "turn_results.json").relative_to(shared.sessions_root.parent).as_posix(),
            "probe_checks": (run_data_dir / "probe_checks.json").relative_to(shared.sessions_root.parent).as_posix(),
            "probe_timeline": (run_data_dir / "probe_timeline.json").relative_to(shared.sessions_root.parent).as_posix(),
            "compaction_events": (run_data_dir / "compaction_events.json").relative_to(shared.sessions_root.parent).as_posix(),
            "workspace_manifest": (session_dir / "workspace" / "_manifest.json").relative_to(shared.sessions_root.parent).as_posix(),
        },
        "workspace_counts": (workspace_export.get("counts") if isinstance(workspace_export, dict) else {}),
    }
    (session_dir / "session_meta.json").write_text(
        json.dumps(session_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[INFO] session_done "
        f"index={session_idx} turns={len(turn_results)} "
        f"first_failure_raw={first_failure_turn_raw} end_reason={end_reason} "
        f"probe_executed={probe_runtime_stats.get('executed_probe_turns', 0)} "
        f"probe_suppressed={len(probe_runtime_stats.get('suppressed_probe_turns') or [])}"
    )
    return session_payload


def _run_single_session_with_quick_self_check(
    shared: SharedRuntime,
    session_idx: int,
    focus_turn: int | None,
) -> dict[str, Any]:
    # Quick self-check: if workspace export is unexpectedly empty (both text/image),
    # rerun the session once from scratch with a fresh session id.
    first = _run_single_session(shared, session_idx, focus_turn=focus_turn)
    text_count, image_count = _workspace_artifact_counts(first)
    if text_count > 0 or image_count > 0:
        return first

    print(
        "[WARN] workspace_artifact_selfcheck_empty "
        f"session={session_idx} text={text_count} image={image_count} retry=1"
    )
    retry_session_dir = shared.sessions_root / f"session_{session_idx:02d}"
    if retry_session_dir.exists():
        shutil.rmtree(retry_session_dir, ignore_errors=True)
    second = _run_single_session(shared, session_idx, focus_turn=focus_turn)
    text_count2, image_count2 = _workspace_artifact_counts(second)
    if text_count2 <= 0 and image_count2 <= 0:
        second["end_reason"] = str(second.get("end_reason") or "") + "|workspace_selfcheck_empty_after_retry"
    return second


def main() -> int:
    args = parse_args(sys.argv[1:])
    if args.skip_preflight_cleanup or _as_bool(os.getenv("AUTO_TEST_SKIP_PREFLIGHT_CLEANUP"), False):
        print("[INFO] preflight_cleanup flag/env detected (deprecated no-op)")
    preflight_cleanup_test_processes("run_memory_compression_failure_scan")
    os.environ["AUTO_TEST_TURN_TIMEOUT_SEC"] = str(args.turn_timeout_sec)
    cfg_path = resolve_config_path()
    cfg = load_config(cfg_path, args.env)
    applied_proxy = apply_proxy_from_config(cfg)
    dotai_base_url = resolve_dotai_base_url(cfg)
    auth_ctx = build_auth_context(cfg.token, cfg.uid, cfg.email)

    run_id = f"{now_stamp()}_{uuid.uuid4().hex[:8]}"
    result_root = AUTO_TEST_DIR / "results" / "memory_failure" / run_id
    sessions_root = result_root / "sessions"
    aggregate_root = result_root / "aggregate"
    sessions_root.mkdir(parents=True, exist_ok=True)
    aggregate_root.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": normalize_authz(cfg.token),
        "uid": cfg.uid,
        "email": cfg.email,
        "Content-Type": "application/json",
    }
    run_settings = build_run_settings()
    persist_type = build_persist_type()
    exec_max_turns = build_exec_max_turns()
    create_settings_json = build_create_settings_json(run_settings)
    llm_eval_cfg = build_llm_eval_config(cfg)
    user_sim_cfg = build_user_simulator_config(cfg, llm_eval_cfg)
    if (not user_sim_cfg.enabled) or (not user_sim_cfg.base_url) or (not user_sim_cfg.model) or (not user_sim_cfg.api_key):
        raise RuntimeError("user_simulator not configured: check enabled/base_url/model/api_key.")

    probe_llm_cfg = LLMEndpointConfig(
        base_url=str(user_sim_cfg.base_url or "").strip(),
        model=str(user_sim_cfg.model or "").strip(),
        api_key=str(user_sim_cfg.api_key or "").strip(),
        timeout_sec=int(user_sim_cfg.timeout_sec),
    )
    if (not probe_llm_cfg.base_url) or (not probe_llm_cfg.model) or (not probe_llm_cfg.api_key):
        raise RuntimeError("probe_llm config missing: base_url/model/api_key")

    print(f"[INFO] config_path={cfg.source_path.as_posix()}")
    print(f"[INFO] runtime_env={cfg.selected_env}")
    print(
        "[INFO] auth_context "
        f"uid={cfg.uid} email={cfg.email} "
        f"token_uid={auth_ctx.get('token_uid') or '(none)'} "
        f"token_email={auth_ctx.get('token_email') or '(none)'} "
        f"token_exp_utc={auth_ctx.get('token_exp_utc') or '(unknown)'} "
        f"token_expired={auth_ctx.get('token_expired')} "
        f"uid_match={auth_ctx.get('uid_match')} "
        f"email_match={auth_ctx.get('email_match')}"
    )
    print(f"[INFO] run_id={run_id}")
    print(f"[INFO] results_root={result_root.as_posix()}")
    print(f"[INFO] sessions={args.sessions}")
    print(f"[INFO] warmup_sessions={args.warmup_sessions}")
    print(f"[INFO] hard_max_turns={args.hard_max_turns}")
    print(f"[INFO] turn_timeout_sec={args.turn_timeout_sec}")
    print(f"[INFO] session_wall_timeout_sec={args.session_wall_timeout_sec}")
    print(f"[INFO] http_connect_timeout_sec={max(3, min(60, _as_int(os.getenv('AUTO_TEST_HTTP_CONNECT_TIMEOUT_SEC'), 15)))}")
    print(
        f"[INFO] stream_read_timeout_sec="
        f"{max(4, min(1800, _as_int(os.getenv('AUTO_TEST_STREAM_READ_TIMEOUT_SEC'), args.turn_timeout_sec + 30)))}"
    )
    print(f"[INFO] create_session_retries={max(0, min(3, _as_int(os.getenv('AUTO_TEST_CREATE_SESSION_RETRIES'), 2)))}")
    print(f"[INFO] execute_transient_retries={max(0, min(3, _as_int(os.getenv('AUTO_TEST_EXECUTE_TRANSIENT_RETRIES'), 3)))}")
    print(f"[INFO] query_ui_events_retries={max(0, min(3, _as_int(os.getenv('AUTO_TEST_QUERY_UI_EVENTS_RETRIES'), 3)))}")
    print(
        f"[INFO] terminal_reconcile_grace_sec="
        f"{max(0.0, min(180.0, _as_float(os.getenv('AUTO_TEST_TERMINAL_RECONCILE_GRACE_SEC'), 90.0)))}"
    )
    print(
        f"[INFO] terminal_reconcile_poll_sec="
        f"{max(0.2, min(10.0, _as_float(os.getenv('AUTO_TEST_TERMINAL_RECONCILE_POLL_SEC'), 1.0)))}"
    )
    print(f"[INFO] probe_interval={args.probe_interval}")
    print(f"[INFO] probe_mode={args.probe_mode}")
    print(f"[INFO] probe_cooldown_turns={args.probe_cooldown_turns}")
    print(f"[INFO] probe_similarity_threshold={args.probe_similarity_threshold}")
    print(f"[INFO] probe_recent_window={args.probe_recent_window}")
    print(f"[INFO] probe_regen_max_attempts={args.probe_regen_max_attempts}")
    print(f"[INFO] focus_window={args.focus_window}")
    print(f"[INFO] parallel_sessions={args.parallel_sessions}")
    print(f"[INFO] importance_tier={args.importance_tier}")
    print(f"[INFO] min_post_samples_per_tier={args.min_post_samples_per_tier}")
    print(f"[INFO] persist_type={persist_type}")
    print(f"[INFO] exec_max_turns={exec_max_turns}")
    print(f"[INFO] dotai_base_url={dotai_base_url or '(disabled)'}")
    print(f"[INFO] user_simulator_mode={user_sim_cfg.mode}")
    print(f"[INFO] user_simulator_model={user_sim_cfg.model}")
    if applied_proxy:
        print(f"[INFO] proxy_from_config={json.dumps(applied_proxy, ensure_ascii=False)}")
    else:
        print("[INFO] proxy_from_config=(none)")

    shared = SharedRuntime(
        cfg=cfg,
        headers=headers,
        run_settings=run_settings,
        persist_type=persist_type,
        exec_max_turns=exec_max_turns,
        create_settings_json=create_settings_json,
        user_sim_cfg=user_sim_cfg,
        dotai_base_url=dotai_base_url,
        sessions_root=sessions_root,
        args=args,
        probe_llm_cfg=probe_llm_cfg,
    )
    run_contract_profile = _load_contract_anchor_profile()

    session_payloads: list[dict[str, Any]] = []
    historical_failure_turns: list[int] = []

    warmup_count = min(args.warmup_sessions, args.sessions)
    for idx in range(1, warmup_count + 1):
        payload = _run_single_session_with_quick_self_check(shared, idx, focus_turn=None)
        session_payloads.append(payload)
        ft = payload.get("first_failure_turn_raw")
        if isinstance(ft, int):
            historical_failure_turns.append(ft)

    next_idx = warmup_count + 1
    if next_idx <= args.sessions:
        for idx in range(next_idx, args.sessions + 1):
            focus_turn = _estimate_focus_turn(historical_failure_turns)
            payload = _run_single_session_with_quick_self_check(shared, idx, focus_turn=focus_turn)
            session_payloads.append(payload)
            ft = payload.get("first_failure_turn_raw")
            if isinstance(ft, int):
                historical_failure_turns.append(ft)

    session_payloads_sorted = sorted(session_payloads, key=lambda x: int(x.get("session_index", 0)))

    chart_sessions = [s for s in session_payloads_sorted if int(s["session_index"]) > args.warmup_sessions]
    chart_values = [
        int(s["first_failure_turn_effective"])
        for s in chart_sessions
        if isinstance(s.get("first_failure_turn_effective"), int)
    ]
    chart_sessions_without_failure = sum(1 for s in chart_sessions if s.get("first_failure_turn_effective") is None)
    chart_title = f"Memory Compression Failure Turn Distribution (exclude first {args.warmup_sessions} warmup sessions)"
    chart_meta = _write_distribution_svg(aggregate_root / "failure_turn_distribution.svg", chart_values, chart_title)

    aggregate = {
        "sessions_total": len(session_payloads_sorted),
        "sessions_with_failure": sum(1 for s in session_payloads_sorted if isinstance(s.get("first_failure_turn_raw"), int)),
        "sessions_without_failure": sum(1 for s in session_payloads_sorted if s.get("first_failure_turn_raw") is None),
        "evaluated_sessions_for_chart": len(chart_sessions),
        "chart_samples": len(chart_values),
        "chart_sessions_without_failure": chart_sessions_without_failure,
        "raw_turn_stats_all_sessions": _build_stats(
            [int(s["first_failure_turn_raw"]) for s in session_payloads_sorted if isinstance(s.get("first_failure_turn_raw"), int)]
        ),
        "effective_turn_stats": _build_stats(chart_values),
    }
    pipeline_health_summary = _build_pipeline_health_summary(session_payloads_sorted)
    compression_effect_summary = _build_compression_effect_summary_from_sessions(
        session_payloads_sorted,
        min_post_samples_per_tier=args.min_post_samples_per_tier,
    )
    probe_runtime_summary = _build_probe_runtime_summary(session_payloads_sorted)
    aggregate["pipeline_health_summary"] = pipeline_health_summary
    aggregate["compression_effect_summary"] = compression_effect_summary
    aggregate["probe_runtime_summary"] = probe_runtime_summary

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "runtime_env": cfg.selected_env,
            "config_source": cfg.source_path.as_posix(),
            "base_url": cfg.base_url,
            "dotai_base_url": dotai_base_url,
            "sessions": args.sessions,
            "warmup_sessions": args.warmup_sessions,
            "hard_max_turns": args.hard_max_turns,
            "turn_timeout_sec": args.turn_timeout_sec,
            "session_wall_timeout_sec": args.session_wall_timeout_sec,
            "probe_interval": args.probe_interval,
            "probe_mode": args.probe_mode,
            "probe_cooldown_turns": args.probe_cooldown_turns,
            "probe_similarity_threshold": args.probe_similarity_threshold,
            "probe_recent_window": args.probe_recent_window,
            "probe_regen_max_attempts": args.probe_regen_max_attempts,
            "focus_window": args.focus_window,
            "parallel_sessions": args.parallel_sessions,
            "importance_tier": args.importance_tier,
            "min_post_samples_per_tier": args.min_post_samples_per_tier,
            "persist_type": persist_type,
            "exec_max_turns": exec_max_turns,
            "run_settings": run_settings,
            "create_settings_json": create_settings_json,
            "auth_context": auth_ctx,
            "results_root": result_root.as_posix(),
        },
        "chart": chart_meta,
        "aggregate": aggregate,
        "anchor": _merge_anchor_with_contract(build_default_probe_anchor(), run_contract_profile),
        "contract_profile": run_contract_profile,
        "sessions": session_payloads_sorted,
    }
    contract_profile_md = summary_payload.get("contract_profile")
    if isinstance(contract_profile_md, dict):
        summary_payload["contract_profile"] = {
            "profile_id": contract_profile_md.get("profile_id"),
            "anchor_name": contract_profile_md.get("anchor_name"),
            "memory_channel_expectation": contract_profile_md.get("memory_channel_expectation"),
            "failure_policy": contract_profile_md.get("failure_policy"),
            "judge_rubric": contract_profile_md.get("judge_rubric"),
            "plant_text_template": contract_profile_md.get("plant_text_template"),
            "probe_text_templates": contract_profile_md.get("probe_text_templates"),
        }

    (aggregate_root / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _build_result_markdown(aggregate_root / "summary.md", summary_payload)
    _build_llm_report(aggregate_root / "memory_failure_llm_report.md", summary_payload)
    (aggregate_root / "pipeline_health_summary.json").write_text(
        json.dumps(pipeline_health_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _build_pipeline_health_report_md(
        aggregate_root / "pipeline_health_report.md",
        {"pipeline_health_summary": pipeline_health_summary},
    )
    (aggregate_root / "compression_effect_summary.json").write_text(
        json.dumps(compression_effect_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _build_compression_effect_report_md(aggregate_root / "compression_effect_report.md", summary_payload)

    (result_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "run_id": run_id,
                "results_root": result_root.as_posix(),
                "sessions_root": sessions_root.as_posix(),
                "aggregate_root": aggregate_root.as_posix(),
                "aggregate_summary_json": (aggregate_root / "summary.json").as_posix(),
                "aggregate_summary_md": (aggregate_root / "summary.md").as_posix(),
                "aggregate_llm_report_md": (aggregate_root / "memory_failure_llm_report.md").as_posix(),
                "aggregate_pipeline_health_json": (aggregate_root / "pipeline_health_summary.json").as_posix(),
                "aggregate_pipeline_health_md": (aggregate_root / "pipeline_health_report.md").as_posix(),
                "aggregate_compression_effect_json": (aggregate_root / "compression_effect_summary.json").as_posix(),
                "aggregate_compression_effect_md": (aggregate_root / "compression_effect_report.md").as_posix(),
                "aggregate_distribution_chart": chart_meta.get("chart_path"),
                "session_count": len(session_payloads_sorted),
                "session_dirs": [f"session_{i:02d}" for i in range(1, len(session_payloads_sorted) + 1)],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[INFO] aggregate_summary={(aggregate_root / 'summary.json').as_posix()}")
    print(f"[INFO] aggregate_chart={chart_meta.get('chart_path')}")
    print(f"[INFO] done results_root={result_root.as_posix()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
