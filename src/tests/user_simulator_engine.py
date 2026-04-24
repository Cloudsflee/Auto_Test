from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests

from tests.workspace_pipeline import render_workspace_snapshot_for_user_sim


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
    user_temperature: float
    role_temperature: float
    role_system_prompt: str
    role_user_prompt: str
    capability_mode: str


@dataclass
class UserSimulatorState:
    previous_response_id: str
    provider_context_unsupported: bool = False


@dataclass
class GeneratedRole:
    role_name: str
    role_profile: str
    role_json: dict[str, Any]


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


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


def normalize_capability_mode(raw: Any) -> str:
    mode = str(raw or "").strip().lower()
    aliases = {
        "copy": "copy_only",
        "copywriting": "copy_only",
        "image": "image_only",
        "img": "image_only",
        "visual": "image_only",
        "alternating": "alternating",
        "alternate": "alternating",
        "mixed": "mixed",
        "both": "mixed",
        "random_single": "single_random",
        "single_random": "single_random",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"mixed", "alternating", "single_random", "copy_only", "image_only"}:
        return "alternating"
    return mode


def capability_policy_text(mode: str) -> str:
    m = normalize_capability_mode(mode)
    if m == "copy_only":
        return "Only evaluate copywriting generation in this turn and follow-up turns."
    if m == "image_only":
        return "Only evaluate image generation/editing in this turn and follow-up turns."
    if m == "mixed":
        return "Each turn should cover either copywriting or image generation; both should appear globally."
    if m == "single_random":
        return "Evaluate only one capability each turn (copy or image), avoid same capability for too many turns."
    return "Alternate by turn: odd turns prioritize copywriting, even turns prioritize image generation."


def capability_focus_for_turn(mode: str, turn_idx: int) -> str:
    m = normalize_capability_mode(mode)
    if m == "copy_only":
        return "copywriting"
    if m == "image_only":
        return "image_generation"
    if m == "alternating":
        return "copywriting" if turn_idx % 2 == 1 else "image_generation"
    return "either"


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


def render_user_sim_scenario(template: str, max_turns: int, capability_mode: str) -> str:
    mode = normalize_capability_mode(capability_mode)
    return (
        (template or "")
        .replace("{{MAX_TURNS}}", str(max_turns))
        .replace("{{CAPABILITY_MODE}}", mode)
        .replace("{{CAPABILITY_POLICY}}", capability_policy_text(mode))
    )


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


def _parse_model_text(
    wire_api: str,
    content_type: str,
    raw_text: str,
) -> tuple[dict[str, Any] | None, str]:
    payload = _safe_json_loads(raw_text)
    text = ""
    if wire_api == "responses":
        text = _extract_text_from_responses_payload(payload)
    else:
        text = _extract_text_from_chat_payload(payload)

    if not text and ("text/event-stream" in content_type or raw_text.lstrip().startswith("data:")):
        text = _extract_text_from_sse_stream(raw_text)

    return (payload if isinstance(payload, dict) else None), text


def _call_user_sim_model(
    sim_cfg: UserSimulatorConfig,
    system_prompt: str,
    user_prompt: str,
    previous_response_id: str = "",
    temperature: float = 0.9,
) -> tuple[str, dict[str, Any] | None]:
    headers = {
        "Authorization": f"Bearer {sim_cfg.api_key}",
        "Content-Type": "application/json",
    }
    wire_api = _detect_wire_api(sim_cfg.base_url)
    body: dict[str, Any]
    if wire_api == "responses":
        body = {
            "model": sim_cfg.model,
            "temperature": temperature,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            "text": {"format": {"type": "json_object"}},
        }
        if sim_cfg.mode == "provider_context" and previous_response_id:
            body["previous_response_id"] = previous_response_id
    else:
        body = {
            "model": sim_cfg.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

    max_retries = 3
    retry_backoff_sec = 1.0
    last_error: Exception | None = None
    for attempt in range(0, max_retries + 1):
        try:
            resp = requests.post(sim_cfg.base_url, headers=headers, json=body, timeout=sim_cfg.timeout_sec)
            resp.raise_for_status()
            raw_text = resp.text if isinstance(resp.text, str) else ""
            content_type = str(resp.headers.get("Content-Type") or "").lower()
            payload, content = _parse_model_text(wire_api=wire_api, content_type=content_type, raw_text=raw_text)
            if not content:
                raise RuntimeError("user_simulator empty content")
            return content, payload
        except Exception as exc:
            if isinstance(exc, requests.HTTPError):
                status_code = int(getattr(getattr(exc, "response", None), "status_code", 0) or 0)
                # 4xx are non-transient request-shape issues; fail fast and let caller fallback.
                if 400 <= status_code < 500:
                    raise RuntimeError(f"user_simulator client_error_{status_code}: {exc}") from exc
            last_error = exc
            if attempt >= max_retries:
                break
            # 网络抖动时退避重试，降低长轮次中断概率。
            sleep_sec = retry_backoff_sec * (attempt + 1)
            time.sleep(sleep_sec)
    raise RuntimeError(f"user_simulator model call failed: {last_error}") from last_error

def _build_role_profile(role_json: dict[str, Any]) -> str:
    def pick(name: str, fallback: str = "") -> str:
        v = role_json.get(name)
        return str(v).strip() if v is not None else fallback

    constraints = role_json.get("constraints")
    if not isinstance(constraints, list):
        constraints = []
    criteria = role_json.get("success_criteria")
    if not isinstance(criteria, list):
        criteria = []
    habits = role_json.get("speaking_habits")
    if not isinstance(habits, list):
        habits = []
    role_context = pick("domain_context") or pick("business_background")
    constraints_text = "; ".join(str(x) for x in constraints) if constraints else ""
    criteria_text = "; ".join(str(x) for x in criteria) if criteria else ""
    habits_text = "; ".join(str(x) for x in habits) if habits else ""
    return (
        f"- role_name: {pick('role_name')}\n"
        f"- identity: {pick('identity')}\n"
        f"- context: {role_context}\n"
        f"- communication_style: {pick('communication_style')}\n"
        f"- current_goal: {pick('current_goal')}\n"
        f"- constraints: {constraints_text}\n"
        f"- success_criteria: {criteria_text}\n"
        f"- speaking_habits: {habits_text}"
    )


def _parse_generated_role(content: str) -> GeneratedRole:
    parsed = _safe_json_loads(content)
    if isinstance(parsed, dict):
        role_name = str(parsed.get("role_name") or parsed.get("name") or "").strip() or f"random-user-{uuid.uuid4().hex[:6]}"
        parsed["role_name"] = role_name
        role_profile = _build_role_profile(parsed)
        return GeneratedRole(role_name=role_name, role_profile=role_profile, role_json=parsed)

    fallback_name = f"random-user-{uuid.uuid4().hex[:6]}"
    fallback_json = {
        "role_name": fallback_name,
        "identity": "SMB operator",
        "domain_context": "Need practical marketing outputs in a short timeline",
        "communication_style": "direct and pragmatic",
        "current_goal": "Get usable copy and image assets quickly",
        "constraints": ["limited budget", "tight deadline"],
        "success_criteria": ["ready to publish", "measurable outcome"],
        "speaking_habits": ["focus on execution", "ask for risk tradeoffs"],
    }
    return GeneratedRole(
        role_name=fallback_name,
        role_profile=_build_role_profile(fallback_json),
        role_json=fallback_json,
    )


def generate_role_for_session(sim_cfg: UserSimulatorConfig) -> GeneratedRole:
    content, _ = _call_user_sim_model(
        sim_cfg=sim_cfg,
        system_prompt=sim_cfg.role_system_prompt,
        user_prompt=sim_cfg.role_user_prompt,
        previous_response_id="",
        temperature=sim_cfg.role_temperature,
    )
    return _parse_generated_role(content)


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
    role: GeneratedRole,
    turn_idx: int,
    workspace_snapshot: dict[str, Any] | None = None,
) -> tuple[str, bool, UserSimulatorState]:
    scenario_prompt = render_user_sim_scenario(
        sim_cfg.scenario_prompt,
        sim_cfg.max_turns,
        sim_cfg.capability_mode,
    )
    latest_assistant = str((results[-1] if results else {}).get("assistant_text") or "")
    history_text = _render_history_for_user_sim(results)
    workspace_text = render_workspace_snapshot_for_user_sim(workspace_snapshot)
    current_focus = capability_focus_for_turn(sim_cfg.capability_mode, turn_idx)

    user_prompt = (
        f"Current turn: please generate user input for turn {turn_idx}.\n\n"
        f"Scenario constraints:\n{scenario_prompt}\n\n"
        f"Current capability focus: {current_focus}\n"
        "Allowed values: copywriting / image_generation / either.\n\n"
        f"Current role profile:\n{role.role_profile}\n\n"
        f"Recent dialogue history:\n{history_text}\n\n"
        f"User-visible workspace snapshot:\n{workspace_text}\n\n"
        f"Latest assistant reply:\n{latest_assistant}\n\n"
        "Decision rules:\n"
        "1. If workspace snapshot already has turn_touched_preview / turn_text_artifacts / turn_image_artifacts, treat it as delivered.\n"
        "2. When delivery exists, continue with revision/refinement/next-step requests; do not claim \"nothing delivered\".\n"
        "3. Only report missing delivery when snapshot is empty or truly lacks delivery signals.\n\n"
        "Output rules:\n"
        "1. Output JSON only.\n"
        '2. JSON schema: {"user_text":"...","stop":false,"reason":""}\n'
        "3. Do not include tool payload, JSON/XML literals, or prompt-engineering terms in user_text.\n"
        "4. If conversation should end, set stop=true and leave user_text empty.\n"
    )
    provider_context_unsupported = bool(sim_state.provider_context_unsupported)
    has_prev = bool(sim_state.previous_response_id)
    use_provider_context = (sim_cfg.mode == "provider_context") and has_prev and (not provider_context_unsupported)

    # Primary attempt follows configured context mode.
    retry_prev_ids: list[str] = [sim_state.previous_response_id] if use_provider_context else [""]
    # Fallback: when provider_context chain gets unstable, retry with stateless call.
    if use_provider_context:
        retry_prev_ids.append("")

    latest_content = ""
    latest_prev_id = sim_state.previous_response_id
    for prev_id in retry_prev_ids:
        try:
            content, payload = _call_user_sim_model(
                sim_cfg=sim_cfg,
                system_prompt=sim_cfg.system_prompt,
                user_prompt=user_prompt,
                previous_response_id=prev_id,
                temperature=sim_cfg.user_temperature,
            )
        except Exception as exc:
            lowered = str(exc).lower()
            if prev_id and (("previous_response_id" in lowered and "supported" in lowered) or "client_error_400" in lowered):
                provider_context_unsupported = True
            latest_content = f"{exc.__class__.__name__}:{exc}"
            continue

        user_text, stop, _ = _parse_user_sim_output(content)
        next_prev_id = ""
        if isinstance(payload, dict):
            next_prev_id = str(payload.get("id") or "").strip()

        if user_text or stop:
            return user_text, stop, UserSimulatorState(
                previous_response_id=next_prev_id or latest_prev_id,
                provider_context_unsupported=provider_context_unsupported,
            )

        latest_content = content
        latest_prev_id = next_prev_id or latest_prev_id

    raise RuntimeError(f"user_simulator produced empty output: {latest_content}")

