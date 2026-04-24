from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests


CAUSE_COMPRESSION_RELATED = "compression_related"
CAUSE_TASK_SWITCH = "task_switch"
CAUSE_INSTRUCTION_OVERRIDE = "instruction_override"
CAUSE_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
CAUSE_UNKNOWN = "unknown"

JUDGE_REMEMBERED = "remembered"
JUDGE_AMBIGUOUS = "ambiguous"
JUDGE_FORGOTTEN = "forgotten"


@dataclass
class LLMEndpointConfig:
    base_url: str
    model: str
    api_key: str
    timeout_sec: int = 60


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


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
    output = payload.get("output")
    parts: list[str] = []
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
    parts: list[str] = []
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
            parts.append(text)
    return "".join(parts).strip()


def _parse_llm_text(wire_api: str, content_type: str, raw_text: str) -> tuple[dict[str, Any] | None, str]:
    payload = _safe_json_loads(raw_text)
    text = ""
    if wire_api == "responses":
        text = _extract_text_from_responses_payload(payload)
    else:
        text = _extract_text_from_chat_payload(payload)
    if not text and ("text/event-stream" in content_type or raw_text.lstrip().startswith("data:")):
        text = _extract_text_from_sse_stream(raw_text)
    return (payload if isinstance(payload, dict) else None), text


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


def _to_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _short_text(text: str, max_len: int = 220) -> str:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(raw) <= max_len:
        return raw
    return raw[: max_len - 3] + "..."


def call_json_llm(
    cfg: LLMEndpointConfig,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> dict[str, Any]:
    if not cfg.base_url or not cfg.model or not cfg.api_key:
        raise RuntimeError("llm config missing base_url/model/api_key")

    wire_api = _detect_wire_api(cfg.base_url)
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    if wire_api == "responses":
        body = {
            "model": cfg.model,
            "temperature": temperature,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            "text": {"format": {"type": "json_object"}},
        }
    else:
        body = {
            "model": cfg.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
    resp = requests.post(cfg.base_url, headers=headers, json=body, timeout=cfg.timeout_sec)
    resp.raise_for_status()
    raw_text = resp.text if isinstance(resp.text, str) else ""
    content_type = str(resp.headers.get("Content-Type") or "").lower()
    _, text = _parse_llm_text(wire_api, content_type, raw_text)
    parsed = _safe_json_loads(text if isinstance(text, str) else "")
    if not isinstance(parsed, dict):
        raise RuntimeError("llm returned non-json-object")
    return parsed


def build_default_probe_anchor() -> dict[str, Any]:
    points = [
        {
            "point_id": "p1",
            "text": "本轮活动主题叫“春夜开跑周”。",
            "tier": "must_keep",
        },
        {
            "point_id": "p2",
            "text": "视觉偏好是“霓虹绿 + 黑底，偏街头感”。",
            "tier": "should_keep",
        },
        {
            "point_id": "p3",
            "text": "结尾引导句偏好“先到店试喝，再决定下单”。",
            "tier": "may_drop",
        },
    ]
    plant_message = (
        "我们先对齐一下这轮方向，你先记住：\n"
        "主题是“春夜开跑周”；视觉想要“霓虹绿配黑底，偏街头感”；"
        "结尾更想用“先到店试喝，再决定下单”。\n"
        "你先简单复述你记住了什么。"
    )
    return {
        "anchor_id": "campaign_theme_visual_cta_v1",
        "anchor_name": "活动主题-视觉-引导句",
        "anchor_points": [p["text"] for p in points],
        "anchor_points_structured": points,
        "plant_message": plant_message,
    }


PROBE_QUESTION_SYSTEM = (
    "你是自动化测试中的“真实用户模拟器”。\n"
    "目标：围绕已埋入的信息点，提出一条自然、口语化、像真人的追问，"
    "用于测试对方是否仍然记得，不要像考题。\n"
    "只输出 JSON。"
)

PROBE_QUESTION_USER = (
    "当前轮次: {{TURN}}\n"
    "锚点主题: {{ANCHOR_NAME}}\n"
    "锚点要素:\n{{ANCHOR_POINTS}}\n\n"
    "最近上下文:\n{{RECENT_HISTORY}}\n\n"
    "请输出 JSON:\n"
    "{\n"
    '  "probe_user_text": "自然口语化试探问句",\n'
    '  "probe_goal": "本轮想确认的记忆点"\n'
    "}\n"
    "要求:\n"
    "1) 不要逐条复述全部锚点；\n"
    "2) 问法自然，不要像审题；\n"
    "3) 不要出现“你是否记得我说过”这类模板。"
)


SUSPECT_JUDGE_SYSTEM = (
    "你是“初判裁判”，负责快速判断 assistant 当前回复是否疑似遗忘锚点。\n"
    "只输出 JSON。"
)

SUSPECT_JUDGE_USER = (
    "锚点主题: {{ANCHOR_NAME}}\n"
    "锚点要素:\n{{ANCHOR_POINTS}}\n\n"
    "本轮试探问题:\n{{PROBE_QUESTION}}\n\n"
    "assistant 回答:\n{{ASSISTANT_REPLY}}\n\n"
    "请输出 JSON:\n"
    "{\n"
    '  "loss_suspected": true,\n'
    '  "confidence_0_1": 0.0,\n'
    '  "remembered_points": ["..."],\n'
    '  "missing_points": ["..."],\n'
    '  "reason_short": "一句话说明"\n'
    "}\n"
    "注意：不要把“当前任务切换到别的产品线”误判为记忆丢失。\n"
    "你只判断 assistant 是否仍能调用锚点信息本身。"
)


VERIFY_JUDGE_SYSTEM = (
    "你是“复核裁判”，必须严格、保守地判断是否真的发生遗忘。\n"
    "你只输出 JSON，不要输出任何解释性前缀。"
)

VERIFY_JUDGE_USER = (
    "锚点主题: {{ANCHOR_NAME}}\n"
    "锚点要素:\n{{ANCHOR_POINTS}}\n\n"
    "本轮试探问题:\n{{PROBE_QUESTION}}\n"
    "assistant 回答:\n{{ASSISTANT_REPLY}}\n\n"
    "初判理由:\n{{SUSPECT_REASON}}\n\n"
    "最近对话窗口:\n{{CONVERSATION_WINDOW}}\n\n"
    "请输出 JSON:\n"
    "{\n"
    '  "judge_state": "remembered|ambiguous|forgotten",\n'
    '  "cause_hint": "compression_related|task_switch|instruction_override|insufficient_evidence|unknown",\n'
    '  "forgotten": true,\n'
    '  "confidence_0_1": 0.0,\n'
    '  "score_0_100": 0,\n'
    '  "reason": "一句话结论",\n'
    '  "evidence_refs": ["turn:xx", "turn:yy"]\n'
    "}\n"
    "判定约束：\n"
    "1) forgotten=true 仅在锚点已不可用时给出；\n"
    "2) 若只是任务切换/策略转向但仍能正确提及锚点，不算 forgotten；\n"
    "3) 无法充分判断时优先 judge_state=ambiguous 且 cause_hint=insufficient_evidence。"
)


def _render(template: str, vars_map: dict[str, str]) -> str:
    out = str(template or "")
    for key, value in vars_map.items():
        out = out.replace(f"{{{{{key}}}}}", str(value))
    return out


def _extract_anchor_points(anchor: dict[str, Any]) -> list[str]:
    structured = anchor.get("anchor_points_structured")
    if isinstance(structured, list):
        out: list[str] = []
        for item in structured:
            if not isinstance(item, dict):
                continue
            txt = str(item.get("text") or "").strip()
            if txt:
                tier = str(item.get("tier") or "").strip()
                out.append(f"[{tier}] {txt}" if tier else txt)
        if out:
            return out
    plain = _to_str_list(anchor.get("anchor_points"))
    return plain


def _format_anchor_points(anchor_points: list[str]) -> str:
    if not anchor_points:
        return "- (empty)"
    return "\n".join(f"- {p}" for p in anchor_points)


def build_fallback_probe_question(turn_idx: int) -> str:
    return f"第{turn_idx}轮我想继续这波活动，你按我最早那版偏好再给一句短方向。"


def generate_probe_question(
    cfg: LLMEndpointConfig,
    anchor: dict[str, Any],
    turn_idx: int,
    recent_history: str,
) -> dict[str, Any]:
    anchor_name = str(anchor.get("anchor_name") or "锚点").strip()
    anchor_points = _extract_anchor_points(anchor)
    user_prompt = _render(
        PROBE_QUESTION_USER,
        {
            "TURN": str(turn_idx),
            "ANCHOR_NAME": anchor_name,
            "ANCHOR_POINTS": _format_anchor_points(anchor_points),
            "RECENT_HISTORY": _short_text(recent_history, max_len=2000),
        },
    )
    parsed = call_json_llm(cfg, PROBE_QUESTION_SYSTEM, user_prompt, temperature=0.6)
    text = str(parsed.get("probe_user_text") or "").strip()
    if not text:
        raise RuntimeError("probe question empty")
    return {
        "probe_user_text": text,
        "probe_goal": str(parsed.get("probe_goal") or "").strip(),
        "raw": parsed,
    }


def judge_loss_suspected(
    cfg: LLMEndpointConfig,
    anchor: dict[str, Any],
    probe_question: str,
    assistant_reply: str,
) -> dict[str, Any]:
    anchor_name = str(anchor.get("anchor_name") or "锚点").strip()
    anchor_points = _extract_anchor_points(anchor)
    user_prompt = _render(
        SUSPECT_JUDGE_USER,
        {
            "ANCHOR_NAME": anchor_name,
            "ANCHOR_POINTS": _format_anchor_points(anchor_points),
            "PROBE_QUESTION": _short_text(probe_question, max_len=800),
            "ASSISTANT_REPLY": _short_text(assistant_reply, max_len=2000),
        },
    )
    parsed = call_json_llm(cfg, SUSPECT_JUDGE_SYSTEM, user_prompt, temperature=0.0)
    return {
        "loss_suspected": _to_bool(parsed.get("loss_suspected"), False),
        "confidence_0_1": round(max(0.0, min(1.0, _to_float(parsed.get("confidence_0_1"), 0.0))), 4),
        "remembered_points": _to_str_list(parsed.get("remembered_points")),
        "missing_points": _to_str_list(parsed.get("missing_points")),
        "reason_short": str(parsed.get("reason_short") or "").strip(),
        "raw": parsed,
    }


def _normalize_judge_state(raw: Any, forgotten: bool) -> str:
    state = str(raw or "").strip().lower()
    if state in {JUDGE_REMEMBERED, JUDGE_AMBIGUOUS, JUDGE_FORGOTTEN}:
        return state
    return JUDGE_FORGOTTEN if forgotten else JUDGE_AMBIGUOUS


def _normalize_cause_hint(raw: Any) -> str:
    cause = str(raw or "").strip().lower()
    if cause in {
        CAUSE_COMPRESSION_RELATED,
        CAUSE_TASK_SWITCH,
        CAUSE_INSTRUCTION_OVERRIDE,
        CAUSE_INSUFFICIENT_EVIDENCE,
        CAUSE_UNKNOWN,
    }:
        return cause
    return CAUSE_UNKNOWN


def verify_loss_confirmed(
    cfg: LLMEndpointConfig,
    anchor: dict[str, Any],
    probe_question: str,
    assistant_reply: str,
    suspect_reason: str,
    conversation_window: list[dict[str, Any]],
) -> dict[str, Any]:
    anchor_name = str(anchor.get("anchor_name") or "锚点").strip()
    anchor_points = _extract_anchor_points(anchor)
    window_text = json.dumps(conversation_window, ensure_ascii=False, indent=2)
    user_prompt = _render(
        VERIFY_JUDGE_USER,
        {
            "ANCHOR_NAME": anchor_name,
            "ANCHOR_POINTS": _format_anchor_points(anchor_points),
            "PROBE_QUESTION": _short_text(probe_question, max_len=800),
            "ASSISTANT_REPLY": _short_text(assistant_reply, max_len=2000),
            "SUSPECT_REASON": _short_text(suspect_reason, max_len=500),
            "CONVERSATION_WINDOW": _short_text(window_text, max_len=4500),
        },
    )
    parsed = call_json_llm(cfg, VERIFY_JUDGE_SYSTEM, user_prompt, temperature=0.0)
    forgotten = _to_bool(parsed.get("forgotten"), False)
    judge_state = _normalize_judge_state(parsed.get("judge_state"), forgotten)
    cause_hint = _normalize_cause_hint(parsed.get("cause_hint"))
    if judge_state == JUDGE_FORGOTTEN:
        forgotten = True
    return {
        "judge_state": judge_state,
        "cause_hint": cause_hint,
        "forgotten": forgotten,
        "confidence_0_1": round(max(0.0, min(1.0, _to_float(parsed.get("confidence_0_1"), 0.0))), 4),
        "score_0_100": round(max(0.0, min(100.0, _to_float(parsed.get("score_0_100"), 0.0))), 2),
        "reason": str(parsed.get("reason") or "").strip(),
        "evidence_refs": _to_str_list(parsed.get("evidence_refs")),
        "raw": parsed,
    }
