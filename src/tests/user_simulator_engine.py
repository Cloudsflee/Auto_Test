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
        return "本轮及后续轮次只考查文案撰写能力，不发起图片生成要求。"
    if m == "image_only":
        return "本轮及后续轮次只考查图片生成/编辑能力，不发起纯文案交付要求。"
    if m == "mixed":
        return "每轮都要至少覆盖文案或图片中的一项；在全局上尽量两项都出现。"
    if m == "single_random":
        return "每轮仅考查一项能力（文案或图片），由你随机选择，避免连续 2 轮同一能力。"
    return "按轮次交替考查：奇数轮优先文案，偶数轮优先图片生成。"


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
            last_error = exc
            if attempt >= max_retries:
                break
            # 网络抖动（如 WinError 10054）时退避重试，降低长轮次中断概率。
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
    return (
        f"- 角色名: {pick('role_name')}\n"
        f"- 身份: {pick('identity')}\n"
        f"- 业务背景: {pick('business_background')}\n"
        f"- 沟通风格: {pick('communication_style')}\n"
        f"- 当前目标: {pick('current_goal')}\n"
        f"- 约束条件: {'；'.join(str(x) for x in constraints) if constraints else ''}\n"
        f"- 成功标准: {'；'.join(str(x) for x in criteria) if criteria else ''}\n"
        f"- 表达习惯: {'；'.join(str(x) for x in habits) if habits else ''}"
    )


def _parse_generated_role(content: str) -> GeneratedRole:
    parsed = _safe_json_loads(content)
    if isinstance(parsed, dict):
        role_name = str(parsed.get("role_name") or parsed.get("name") or "").strip() or f"随机客户-{uuid.uuid4().hex[:6]}"
        parsed["role_name"] = role_name
        role_profile = _build_role_profile(parsed)
        return GeneratedRole(role_name=role_name, role_profile=role_profile, role_json=parsed)

    fallback_name = f"随机客户-{uuid.uuid4().hex[:6]}"
    fallback_json = {
        "role_name": fallback_name,
        "identity": "中小企业主",
        "business_background": "近期希望提升线上获客效率",
        "communication_style": "直接务实",
        "current_goal": "尽快得到可执行的文案与图片素材",
        "constraints": ["预算有限", "时间紧"],
        "success_criteria": ["能直接上线", "可衡量效果"],
        "speaking_habits": ["关注落地性", "会追问风险点"],
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
        f"当前请生成第 {turn_idx} 轮用户输入。\n\n"
        f"场景要求：\n{scenario_prompt}\n\n"
        f"本轮能力焦点：{current_focus}\n"
        "可选值说明：copywriting / image_generation / either。\n\n"
        f"当前扮演角色：\n{role.role_profile}\n\n"
        f"最近多轮记录：\n{history_text}\n\n"
        f"Workspace snapshot visible to user:\n{workspace_text}\n\n"
        f"上一轮助手回复：\n{latest_assistant}\n\n"
        "输出要求：\n"
        "1. 只输出 JSON。\n"
        "2. JSON 格式：{\"user_text\":\"...\",\"stop\":false,\"reason\":\"\"}\n"
        "3. user_text 里不要出现工具调用、JSON、XML。\n"
        "4. 如果应该结束，stop=true 且 user_text 置空。"
    )
    # Primary attempt follows configured context mode.
    retry_prev_ids: list[str] = [sim_state.previous_response_id]
    # Fallback: when provider_context chain gets unstable, retry with stateless call.
    if sim_cfg.mode == "provider_context" and sim_state.previous_response_id:
        retry_prev_ids.append("")

    latest_content = ""
    latest_prev_id = sim_state.previous_response_id
    for prev_id in retry_prev_ids:
        content, payload = _call_user_sim_model(
            sim_cfg=sim_cfg,
            system_prompt=sim_cfg.system_prompt,
            user_prompt=user_prompt,
            previous_response_id=prev_id,
            temperature=sim_cfg.user_temperature,
        )

        user_text, stop, _ = _parse_user_sim_output(content)
        next_prev_id = ""
        if isinstance(payload, dict):
            next_prev_id = str(payload.get("id") or "").strip()

        if user_text or stop:
            return user_text, stop, UserSimulatorState(previous_response_id=next_prev_id or latest_prev_id)

        latest_content = content
        latest_prev_id = next_prev_id or latest_prev_id

    raise RuntimeError(f"user_simulator produced empty output: {latest_content}")
