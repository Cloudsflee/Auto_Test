from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import requests

from .models import ProbeLLMJudgeConfig, ProbeSpec


DEFAULT_SYSTEM_PROMPT = (
    "你是严格的探针评估裁判。"
    "只允许输出 JSON 对象，不要输出 Markdown。"
    "评分范围为 0~5，且必须给出可复核证据。"
)

DEFAULT_USER_PROMPT = (
    "请评估以下探针。\n"
    "probe_id: {{PROBE_ID}}\n"
    "probe_type: {{PROBE_TYPE}}\n"
    "description: {{PROBE_DESCRIPTION}}\n"
    "target: {{PROBE_TARGET}}\n"
    "pass_threshold_0_5: {{PASS_THRESHOLD_0_5}}\n"
    "rubric:\n{{PROBE_RUBRIC}}\n\n"
    "context:\n{{PROBE_CONTEXT}}\n\n"
    "输出 JSON 字段：\n"
    "- pass (bool)\n"
    "- overall_0_5 (number)\n"
    "- dimension_scores (object)\n"
    "- reason (string)\n"
    "- evidence_refs (array of string)\n"
)

RUBRIC_HINTS = {
    "accuracy": "事实、路径、技术细节是否正确",
    "context_awareness": "是否准确利用了当前上下文状态",
    "artifact_trail": "是否清楚跟踪到相关文件/制品",
    "completeness": "是否覆盖问题中的关键要求",
    "continuity": "是否支持下一步连续推进",
    "instruction_following": "是否遵循 probe 指令与格式要求",
}


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _load_prompt(path: Path, fallback: str) -> str:
    if not path.exists():
        return fallback
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return text or fallback


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


def _to_string_list(value: Any) -> list[str]:
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


def _build_rubric_text(dimensions: list[str]) -> str:
    lines: list[str] = []
    for key in dimensions:
        desc = RUBRIC_HINTS.get(key, key)
        lines.append(f"- {key}: {desc}（0-5分）")
    return "\n".join(lines)


def _json_trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 19] + "\n...(truncated)..."


def _resolve_prompt_path(raw_path: str, base_dir: Path, default_name: str) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        return base_dir / default_name
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def _build_prompt_pair(
    probe: ProbeSpec,
    probe_context: dict[str, Any],
    cfg: ProbeLLMJudgeConfig,
    prompts_dir: Path | None,
) -> tuple[str, str, str]:
    base_dir = prompts_dir if isinstance(prompts_dir, Path) else Path.cwd()
    system_path = _resolve_prompt_path(
        cfg.system_prompt_path,
        base_dir,
        "framework/evaluator/probe/probe_judge_system.prompt",
    )
    user_path = _resolve_prompt_path(
        cfg.user_prompt_path,
        base_dir,
        "framework/evaluator/probe/probe_judge_user.prompt",
    )

    system_template = _load_prompt(system_path, DEFAULT_SYSTEM_PROMPT)
    user_template = _load_prompt(user_path, DEFAULT_USER_PROMPT)

    llm_spec = probe.llm_judge
    if llm_spec is None:
        raise RuntimeError(f"probe[{probe.probe_id}] missing llm_judge")
    context_json = json.dumps(probe_context, ensure_ascii=False, indent=2)
    context_json = _json_trim_text(context_json, 12000)
    target_json = json.dumps(
        {"source": probe.target.source, "turn": probe.target.turn},
        ensure_ascii=False,
    )
    user_prompt = (
        user_template.replace("{{PROBE_ID}}", probe.probe_id)
        .replace("{{PROBE_TYPE}}", probe.probe_type)
        .replace("{{PROBE_DESCRIPTION}}", probe.description or "")
        .replace("{{PROBE_TARGET}}", target_json)
        .replace("{{PASS_THRESHOLD_0_5}}", f"{llm_spec.pass_threshold_0_5}")
        .replace("{{PROBE_RUBRIC}}", _build_rubric_text(llm_spec.dimensions))
        .replace("{{PROBE_CONTEXT}}", context_json)
    )
    prompt_hash = hashlib.sha256((system_template + "\n" + user_prompt).encode("utf-8", errors="ignore")).hexdigest()[:12]
    return system_template, user_prompt, prompt_hash


def _normalize_dimension_scores(raw: Any, dimensions: list[str]) -> dict[str, float]:
    out = {key: 0.0 for key in dimensions}
    if isinstance(raw, dict):
        for key in dimensions:
            out[key] = max(0.0, min(5.0, _to_float(raw.get(key), 0.0)))
    return out


def _parse_one_result(
    parsed_json: Any,
    dimensions: list[str],
    threshold: float,
) -> dict[str, Any]:
    if not isinstance(parsed_json, dict):
        raise RuntimeError("llm judge response is not a json object")

    dim_scores = _normalize_dimension_scores(parsed_json.get("dimension_scores"), dimensions)
    explicit_overall = _to_float(parsed_json.get("overall_0_5"), -1.0)
    mean_overall = sum(dim_scores.values()) / max(1, len(dim_scores))
    overall = explicit_overall if explicit_overall >= 0 else mean_overall
    overall = max(0.0, min(5.0, overall))
    passed = _to_bool(parsed_json.get("pass"), overall >= threshold)
    reason = str(parsed_json.get("reason") or "").strip()
    evidence_refs = _to_string_list(parsed_json.get("evidence_refs"))
    return {
        "pass": passed,
        "overall_0_5": round(overall, 4),
        "dimension_scores": {k: round(v, 4) for k, v in dim_scores.items()},
        "reason": reason,
        "evidence_refs": evidence_refs,
    }


def _do_single_call(
    cfg: ProbeLLMJudgeConfig,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    wire_api = _detect_wire_api(cfg.base_url)
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    if wire_api == "responses":
        body = {
            "model": cfg.model,
            "temperature": 0,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
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
    resp = requests.post(cfg.base_url, headers=headers, json=body, timeout=cfg.timeout_sec)
    status_code = resp.status_code
    resp.raise_for_status()
    raw_text = resp.text if isinstance(resp.text, str) else ""
    content_type = (resp.headers.get("Content-Type") or "").lower()
    payload, text = _parse_llm_text(wire_api=wire_api, content_type=content_type, raw_text=raw_text)
    parsed = _safe_json_loads(text) if isinstance(text, str) else None
    return {
        "status_code": status_code,
        "wire_api": wire_api,
        "payload": payload,
        "response_text": text if isinstance(text, str) else "",
        "parsed_json": parsed if isinstance(parsed, dict) else None,
        "raw_response_preview": raw_text[:1000],
    }


def _aggregate_attempts(attempts: list[dict[str, Any]], dimensions: list[str], threshold: float) -> dict[str, Any]:
    valid = [item for item in attempts if item.get("error") is None]
    if not valid:
        return {
            "valid_attempts": 0,
            "overall_mean": 0.0,
            "overall_stddev": 0.0,
            "pass_rate": 0.0,
            "dimension_mean": {k: 0.0 for k in dimensions},
            "dimension_stddev": {k: 0.0 for k in dimensions},
            "final_pass": False,
        }

    values = [_to_float(item.get("overall_0_5"), 0.0) for item in valid]
    mean_score = statistics.fmean(values)
    std_score = statistics.pstdev(values) if len(values) > 1 else 0.0
    pass_rate = sum(1 for item in valid if _to_bool(item.get("pass"), False)) / len(valid)

    dim_mean: dict[str, float] = {}
    dim_std: dict[str, float] = {}
    for key in dimensions:
        dim_values = [_to_float((item.get("dimension_scores") or {}).get(key), 0.0) for item in valid]
        dim_mean[key] = statistics.fmean(dim_values)
        dim_std[key] = statistics.pstdev(dim_values) if len(dim_values) > 1 else 0.0

    final_pass = (pass_rate >= 0.5) and (mean_score >= threshold)
    return {
        "valid_attempts": len(valid),
        "overall_mean": round(mean_score, 4),
        "overall_stddev": round(std_score, 4),
        "pass_rate": round(pass_rate, 4),
        "dimension_mean": {k: round(v, 4) for k, v in dim_mean.items()},
        "dimension_stddev": {k: round(v, 4) for k, v in dim_std.items()},
        "final_pass": final_pass,
    }


def run_probe_llm_judge(
    probe: ProbeSpec,
    probe_context: dict[str, Any],
    cfg: ProbeLLMJudgeConfig,
    prompts_dir: Path | None = None,
) -> dict[str, Any]:
    llm_spec = probe.llm_judge
    if llm_spec is None:
        return {
            "enabled": False,
            "skipped": True,
            "error": "missing llm_judge spec",
        }
    if not cfg.enabled:
        return {
            "enabled": False,
            "skipped": True,
            "error": "probe llm judge disabled",
        }
    if not cfg.base_url or not cfg.model or not cfg.api_key:
        error = "probe llm judge config missing base_url/model/api_key"
        if cfg.fail_open:
            return {
                "enabled": True,
                "skipped": True,
                "error": error,
            }
        return {
            "enabled": True,
            "skipped": False,
            "error": error,
            "aggregate": {
                "valid_attempts": 0,
                "overall_mean": 0.0,
                "overall_stddev": 0.0,
                "pass_rate": 0.0,
                "final_pass": False,
            },
        }

    system_prompt, user_prompt, prompt_hash = _build_prompt_pair(probe, probe_context, cfg, prompts_dir=prompts_dir)
    attempts: list[dict[str, Any]] = []
    errors: list[str] = []

    repeats = max(1, min(9, int(cfg.repeats)))
    max_retries = max(0, min(6, int(cfg.max_retries)))
    threshold = max(0.0, min(5.0, float(llm_spec.pass_threshold_0_5)))
    dimensions = list(llm_spec.dimensions)

    for idx in range(1, repeats + 1):
        last_error = ""
        success = False
        for retry in range(0, max_retries + 1):
            try:
                response = _do_single_call(cfg=cfg, system_prompt=system_prompt, user_prompt=user_prompt)
                parsed = _parse_one_result(response.get("parsed_json"), dimensions=dimensions, threshold=threshold)
                attempts.append(
                    {
                        "attempt": idx,
                        "retry": retry,
                        "status_code": response.get("status_code"),
                        "wire_api": response.get("wire_api"),
                        "pass": parsed.get("pass"),
                        "overall_0_5": parsed.get("overall_0_5"),
                        "dimension_scores": parsed.get("dimension_scores"),
                        "reason": parsed.get("reason"),
                        "evidence_refs": parsed.get("evidence_refs"),
                        "error": None,
                        "raw_response_preview": response.get("raw_response_preview"),
                    }
                )
                success = True
                break
            except Exception as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
        if not success:
            errors.append(last_error or "unknown llm judge error")
            attempts.append(
                {
                    "attempt": idx,
                    "retry": max_retries,
                    "pass": False,
                    "overall_0_5": 0.0,
                    "dimension_scores": {k: 0.0 for k in dimensions},
                    "reason": "",
                    "evidence_refs": [],
                    "error": last_error or "unknown llm judge error",
                }
            )

    aggregate = _aggregate_attempts(attempts=attempts, dimensions=dimensions, threshold=threshold)
    valid_attempts = int(aggregate.get("valid_attempts") or 0)
    if valid_attempts == 0 and cfg.fail_open:
        return {
            "enabled": True,
            "skipped": True,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "prompt_version_hash": prompt_hash,
            "repeats": repeats,
            "max_retries": max_retries,
            "attempts": attempts,
            "aggregate": aggregate,
            "error": "; ".join(errors) if errors else "all attempts failed",
        }

    return {
        "enabled": True,
        "skipped": False,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "prompt_version_hash": prompt_hash,
        "repeats": repeats,
        "max_retries": max_retries,
        "attempts": attempts,
        "aggregate": aggregate,
        "error": "; ".join(errors) if errors else "",
    }
