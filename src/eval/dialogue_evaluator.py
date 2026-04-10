from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class LLMEvalConfig:
    enabled: bool
    base_url: str
    model: str
    api_key: str
    timeout_sec: int


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "y", "on")


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", (text or "")).lower()


def _contains_fact(text: str, fact: str) -> bool:
    return _normalize_for_match(fact) in _normalize_for_match(text)


def _get_turn_result(results: list[dict[str, Any]], turn_idx: int) -> dict[str, Any]:
    for item in results:
        if int(item.get("turn", -1)) == turn_idx:
            return item
    return {}


def load_llm_eval_config_from_env() -> LLMEvalConfig:
    return LLMEvalConfig(
        enabled=_env_bool("AUTO_TEST_ENABLE_LLM_EVAL", False),
        base_url=os.getenv("AUTO_TEST_EVAL_LLM_URL", "").strip(),
        model=os.getenv("AUTO_TEST_EVAL_LLM_MODEL", "").strip(),
        api_key=os.getenv("AUTO_TEST_EVAL_LLM_API_KEY", "").strip(),
        timeout_sec=_env_int("AUTO_TEST_EVAL_LLM_TIMEOUT_SEC", 30),
    )


def evaluate_rules(
    results: list[dict[str, Any]],
    expected_facts: dict[str, str],
    pass_threshold: float,
) -> dict[str, Any]:
    # 规则评估采用加权打分，关键项失败将直接影响 overall_pass。
    checks: list[dict[str, Any]] = []

    def add_check(
        check_id: str,
        name: str,
        weight: float,
        passed: bool,
        detail: str,
        critical: bool = False,
    ) -> None:
        checks.append(
            {
                "check_id": check_id,
                "name": name,
                "weight": weight,
                "passed": bool(passed),
                "score": 1.0 if passed else 0.0,
                "critical": bool(critical),
                "detail": detail,
            }
        )

    total_turns = len(results)
    success_turns = sum(1 for r in results if r.get("run_end") and not r.get("run_error"))
    add_check(
        "run_success",
        "所有轮次正常结束",
        0.30,
        success_turns == total_turns and total_turns > 0,
        f"success_turns={success_turns}/{total_turns}",
        critical=True,
    )

    empty_turns = [int(r.get("turn", -1)) for r in results if not str(r.get("assistant_text") or "").strip()]
    add_check(
        "non_empty_response",
        "每轮助手回复非空",
        0.15,
        not empty_turns,
        f"empty_turns={empty_turns}",
        critical=True,
    )

    leak_turns: list[int] = []
    leak_patterns = ("<tool>", "</tool>", "\"tool_type\"", "\"tool_name\"", "\"tool_input\"")
    for r in results:
        text = str(r.get("assistant_text") or "")
        if any(pattern in text for pattern in leak_patterns):
            leak_turns.append(int(r.get("turn", -1)))
    add_check(
        "no_tool_leak",
        "用户可见回复中无工具载荷泄漏",
        0.15,
        not leak_turns,
        f"leak_turns={leak_turns}",
        critical=True,
    )

    turn4 = _get_turn_result(results, 4)
    turn4_text = str(turn4.get("assistant_text") or "")
    missing_t4 = [value for value in expected_facts.values() if not _contains_fact(turn4_text, value)]
    add_check(
        "memory_recall_turn4",
        "第4轮记忆召回（姓名/品牌/卖点）",
        0.20,
        not missing_t4,
        f"missing_facts={missing_t4}",
    )

    turn5 = _get_turn_result(results, 5)
    turn5_text = str(turn5.get("assistant_text") or "")
    missing_t5 = [value for value in expected_facts.values() if not _contains_fact(turn5_text, value)]
    add_check(
        "summary_coverage_turn5",
        "第5轮总结覆盖（姓名/品牌/卖点）",
        0.20,
        not missing_t5,
        f"missing_facts={missing_t5}",
    )

    total_weight = sum(float(item["weight"]) for item in checks) or 1.0
    weighted_score = sum(float(item["weight"]) * float(item["score"]) for item in checks) / total_weight
    critical_failed = [item["check_id"] for item in checks if item.get("critical") and not item.get("passed")]
    overall_pass = weighted_score >= pass_threshold and not critical_failed

    return {
        "mode": "rule",
        "pass_threshold": pass_threshold,
        "overall_score": round(weighted_score, 4),
        "overall_pass": bool(overall_pass),
        "critical_failed": critical_failed,
        "metrics": {
            "total_turns": total_turns,
            "success_turns": success_turns,
        },
        "expected_facts": expected_facts,
        "checks": checks,
    }


def render_conversation_text(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in results:
        lines.extend(
            [
                f"Turn {item.get('turn')}",
                f"User: {item.get('user_text') or ''}",
                f"Assistant: {item.get('assistant_text') or ''}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _load_prompt(prompt_file: Path, fallback: str) -> str:
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8", errors="ignore")
    return fallback


def _build_llm_prompts(
    results: list[dict[str, Any]],
    expected_facts: dict[str, str],
    prompts_dir: Path,
) -> tuple[str, str]:
    fallback_system = (
        "你是多轮对话自动测试的严格评审。"
        "请只返回 JSON，包含字段：pass, score_0_100, memory_score, coherence_score, findings, summary。"
    )
    fallback_user = (
        "请评估以下对话。\n"
        "硬事实：\n{{EXPECTED_FACTS}}\n\n"
        "评分维度：\n"
        "1) 记忆召回正确性\n"
        "2) 用户可见文本无工具载荷泄漏\n"
        "3) 回答连贯且可用\n\n"
        "对话：\n{{CONVERSATION}}"
    )

    system_template = _load_prompt(prompts_dir / "llm_eval_system.prompt", fallback_system)
    user_template = _load_prompt(prompts_dir / "llm_eval_user.prompt", fallback_user)

    expected_lines = "\n".join(f"- {key}={value}" for key, value in expected_facts.items())
    conversation = render_conversation_text(results)

    user_prompt = user_template.replace("{{EXPECTED_FACTS}}", expected_lines).replace("{{CONVERSATION}}", conversation)
    return system_template, user_prompt


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
        joined = "".join(parts).strip()
        if joined:
            return joined

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


def _detect_wire_api(url: str) -> str:
    lower = (url or "").lower()
    if lower.endswith("/responses") or "/responses?" in lower:
        return "responses"
    return "chat_completions"


def _parse_llm_text(
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

    # 某些网关会返回 SSE（data: ...），这里做兜底解析。
    if not text and ("text/event-stream" in content_type or raw_text.lstrip().startswith("data:")):
        text = _extract_text_from_sse_stream(raw_text)

    return (payload if isinstance(payload, dict) else None), text


def evaluate_with_llm(
    results: list[dict[str, Any]],
    expected_facts: dict[str, str],
    prompts_dir: Path,
    llm_cfg: LLMEvalConfig | None = None,
) -> dict[str, Any]:
    cfg = llm_cfg or load_llm_eval_config_from_env()
    if not cfg.enabled:
        return {
            "enabled": False,
            "skipped": True,
            "reason": "AUTO_TEST_ENABLE_LLM_EVAL 未启用",
        }
    if not cfg.base_url or not cfg.model or not cfg.api_key:
        return {
            "enabled": True,
            "skipped": True,
            "reason": "缺少 AUTO_TEST_EVAL_LLM_URL / AUTO_TEST_EVAL_LLM_MODEL / AUTO_TEST_EVAL_LLM_API_KEY",
        }

    system_prompt, user_prompt = _build_llm_prompts(results, expected_facts, prompts_dir)

    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    wire_api = _detect_wire_api(cfg.base_url)
    if wire_api == "responses":
        body = {
            "model": cfg.model,
            "temperature": 0,
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
    else:
        body = {
            "model": cfg.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

    try:
        resp = requests.post(cfg.base_url, headers=headers, json=body, timeout=cfg.timeout_sec)
        status_code = resp.status_code
        resp.raise_for_status()
        raw_text = resp.text if isinstance(resp.text, str) else ""
        content_type = (resp.headers.get("Content-Type") or "").lower()
        payload, content = _parse_llm_text(wire_api=wire_api, content_type=content_type, raw_text=raw_text)
        parsed = _safe_json_loads(content) if isinstance(content, str) else None
        return {
            "enabled": True,
            "skipped": False,
            "status_code": status_code,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "wire_api": wire_api,
            "response_json": parsed if isinstance(parsed, dict) else None,
            "response_text": content if isinstance(content, str) else "",
            "raw_response_preview": raw_text[:1000],
        }
    except Exception as exc:
        return {
            "enabled": True,
            "skipped": False,
            "error": str(exc),
            "model": cfg.model,
            "base_url": cfg.base_url,
        }


def write_evaluation_md(path: Path, rule_eval: dict[str, Any] | None, llm_eval: dict[str, Any]) -> None:
    lines: list[str] = ["# 对话评估报告", ""]
    if isinstance(rule_eval, dict) and rule_eval:
        lines.extend(
            [
                "## Rule Evaluation",
                "",
                f"- overall_pass: `{rule_eval.get('overall_pass')}`",
                f"- overall_score: `{rule_eval.get('overall_score')}`",
                f"- pass_threshold: `{rule_eval.get('pass_threshold')}`",
                "",
                "| check_id | passed | weight | detail |",
                "| --- | --- | --- | --- |",
            ]
        )
        for check in rule_eval.get("checks", []):
            lines.append(
                f"| `{check.get('check_id')}` | `{check.get('passed')}` | `{check.get('weight')}` | {check.get('detail') or ''} |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "## Rule Evaluation",
                "",
                "- disabled: `LLM-only mode`",
                "",
            ]
        )

    lines.extend(["## LLM Evaluation", ""])
    if not llm_eval.get("enabled"):
        lines.append(f"- skipped: `{llm_eval.get('reason', 'disabled')}`")
    elif llm_eval.get("skipped"):
        lines.append(f"- skipped: `{llm_eval.get('reason', 'missing config')}`")
    elif llm_eval.get("error"):
        lines.append(f"- error: `{llm_eval.get('error')}`")
    else:
        response_json = llm_eval.get("response_json")
        if isinstance(response_json, dict):
            lines.append(f"- pass: `{response_json.get('pass')}`")
            lines.append(f"- score_0_100: `{response_json.get('score_0_100')}`")
            lines.append(f"- memory_score: `{response_json.get('memory_score')}`")
            lines.append(f"- coherence_score: `{response_json.get('coherence_score')}`")
            lines.append(f"- summary: `{response_json.get('summary')}`")
        else:
            lines.append("- response_json: `(not valid JSON)`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
