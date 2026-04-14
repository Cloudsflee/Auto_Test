from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .loader import load_probe_dataset
from .models import ProbeAssertion, ProbeDataset, ProbeSpec


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _read_json_file(path: Path) -> Any:
    if not path.exists():
        raise RuntimeError(f"required file not found: {path.as_posix()}")
    payload = _safe_json_loads(path.read_text(encoding="utf-8", errors="ignore"))
    if payload is None:
        raise RuntimeError(f"invalid json file: {path.as_posix()}")
    return payload


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _to_paths(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
                continue
            if isinstance(item, dict):
                for key in ("virtual_path", "local_path", "path"):
                    v = item.get(key)
                    if isinstance(v, str) and v.strip():
                        out.append(v.strip())
                        break
        return out
    if isinstance(value, dict):
        return _to_paths(value.get("all_paths"))
    return []


def _get_by_path(obj: Any, path: str) -> Any:
    raw = str(path or "").strip()
    if not raw:
        return obj
    current = obj
    for part in raw.split("."):
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, list):
            try:
                idx = int(part)
            except Exception:
                return None
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
            continue
        return None
    return current


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _extract_target_value(ctx: dict[str, Any], probe: ProbeSpec) -> Any:
    source = probe.target.source
    turn = probe.target.turn
    turn_map = ctx.get("turn_map")
    if not isinstance(turn_map, dict):
        turn_map = {}

    if source == "turn_assistant_text":
        row = turn_map.get(turn, {})
        return str(row.get("assistant_text") or "")
    if source == "turn_user_text":
        row = turn_map.get(turn, {})
        return str(row.get("user_text") or "")
    if source == "turn_object":
        return turn_map.get(turn, {})
    if source == "workspace_manifest":
        return ctx.get("workspace_manifest", {})
    if source == "workspace_paths":
        return ctx.get("workspace_paths", [])
    if source == "global_summary":
        return ctx.get("global_summary", {})
    if source == "turn_results":
        return ctx.get("turn_results", [])
    return None


def _parse_regex_flags(raw_flags: str) -> int:
    flags = 0
    raw = str(raw_flags or "").lower()
    if "i" in raw:
        flags |= re.IGNORECASE
    if "m" in raw:
        flags |= re.MULTILINE
    if "s" in raw:
        flags |= re.DOTALL
    return flags


def _expect_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


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


def _resolve_expect_field(assertion: ProbeAssertion) -> tuple[str, Any]:
    if assertion.field_path:
        return assertion.field_path, assertion.expect
    if isinstance(assertion.expect, dict):
        field = str(assertion.expect.get("field") or assertion.expect.get("field_path") or "").strip()
        value = assertion.expect.get("value")
        return field, value
    return "", assertion.expect


def _eval_assertion(target_value: Any, assertion: ProbeAssertion) -> tuple[bool, Any, str]:
    atype = assertion.assert_type
    detail = ""
    actual: Any = None
    passed = False

    if atype == "non_empty":
        actual = target_value
        if isinstance(target_value, str):
            passed = bool(target_value.strip())
        elif isinstance(target_value, (list, dict)):
            passed = len(target_value) > 0
        else:
            passed = target_value is not None

    elif atype == "contains_any":
        text = _to_text(target_value)
        expected = _expect_list(assertion.expect)
        actual = text
        norm = _normalize_for_match(text)
        passed = any(_normalize_for_match(it) in norm for it in expected)

    elif atype == "contains_all":
        text = _to_text(target_value)
        expected = _expect_list(assertion.expect)
        actual = text
        norm = _normalize_for_match(text)
        passed = all(_normalize_for_match(it) in norm for it in expected)

    elif atype == "not_contains_any":
        text = _to_text(target_value)
        expected = _expect_list(assertion.expect)
        actual = text
        norm = _normalize_for_match(text)
        passed = all(_normalize_for_match(it) not in norm for it in expected)

    elif atype == "regex_match":
        text = _to_text(target_value)
        pattern = str(assertion.expect or "")
        actual = text
        flags = _parse_regex_flags(assertion.flags)
        passed = bool(re.search(pattern, text, flags=flags))

    elif atype == "equals":
        actual = target_value
        passed = target_value == assertion.expect

    elif atype == "file_exists":
        paths = _to_paths(target_value)
        expected_path = str(assertion.expect or "").strip()
        actual = {"paths_count": len(paths)}
        passed = expected_path in paths

    elif atype == "file_ext_count_gte":
        paths = _to_paths(target_value)
        ext_list = assertion.ext
        gte = _as_float(assertion.gte, 0.0)
        if isinstance(assertion.expect, dict):
            exp_ext = assertion.expect.get("ext")
            if isinstance(exp_ext, list):
                ext_list = [str(x).strip().lower().lstrip(".") for x in exp_ext]
            gte = _as_float(assertion.expect.get("gte"), gte)
        if not ext_list:
            ext_list = ["png", "jpg", "jpeg", "webp", "gif", "bmp"]
        count = 0
        for p in paths:
            lower = p.lower()
            if "." not in lower:
                continue
            ext = lower.rsplit(".", 1)[-1].strip().lstrip(".")
            if ext in ext_list:
                count += 1
        actual = {"count": count, "ext": ext_list}
        passed = count >= gte

    elif atype == "json_field_equals":
        field, expected = _resolve_expect_field(assertion)
        actual = _get_by_path(target_value, field)
        passed = actual == expected

    elif atype == "json_field_in":
        field, expected = _resolve_expect_field(assertion)
        expected_list = _expect_list(expected)
        actual = _get_by_path(target_value, field)
        passed = str(actual) in {str(x) for x in expected_list}

    elif atype == "json_field_non_empty":
        field = assertion.field_path or (
            str(assertion.expect.get("field") or assertion.expect.get("field_path") or "").strip()
            if isinstance(assertion.expect, dict)
            else ""
        )
        actual = _get_by_path(target_value, field)
        if isinstance(actual, str):
            passed = bool(actual.strip())
        elif isinstance(actual, (list, dict)):
            passed = len(actual) > 0
        else:
            passed = actual is not None

    elif atype == "bool_field_true":
        field, expected = _resolve_expect_field(assertion)
        if expected is None:
            expected = True
        actual = _get_by_path(target_value, field)
        passed = _as_bool(actual, False) == _as_bool(expected, True)

    elif atype == "numeric_field_gte":
        field = assertion.field_path
        gte = _as_float(assertion.gte, 0.0)
        if isinstance(assertion.expect, dict):
            if not field:
                field = str(assertion.expect.get("field") or assertion.expect.get("field_path") or "").strip()
            gte = _as_float(assertion.expect.get("gte"), gte)
        elif assertion.expect is not None and not field:
            gte = _as_float(assertion.expect, gte)
        actual = _get_by_path(target_value, field) if field else target_value
        passed = _as_float(actual, float("-inf")) >= gte

    elif atype == "list_contains_path_pattern":
        patterns = _expect_list(assertion.expect)
        paths = _to_paths(target_value)
        actual = {"paths_count": len(paths)}
        flags = _parse_regex_flags(assertion.flags)
        passed = all(any(re.search(pat, p, flags=flags) for p in paths) for pat in patterns)

    else:
        detail = f"unsupported assert_type: {atype}"
        actual = target_value
        passed = False

    if assertion.negate:
        passed = not passed
    return passed, actual, detail


def _build_probe_context(
    turn_results_path: Path,
    workspace_manifest_path: Path,
    raw_events_path: Path | None = None,
) -> dict[str, Any]:
    turn_results = _read_json_file(turn_results_path)
    if not isinstance(turn_results, list):
        raise RuntimeError(f"turn_results must be list: {turn_results_path.as_posix()}")
    workspace_manifest = _read_json_file(workspace_manifest_path)
    if not isinstance(workspace_manifest, dict):
        raise RuntimeError(f"workspace_manifest must be object: {workspace_manifest_path.as_posix()}")

    turn_map: dict[int, dict[str, Any]] = {}
    for row in turn_results:
        if not isinstance(row, dict):
            continue
        turn = row.get("turn")
        if isinstance(turn, (int, float)):
            turn_map[int(turn)] = row

    success_turns = sum(1 for row in turn_results if isinstance(row, dict) and row.get("run_end") and not row.get("run_error"))
    any_run_error = any(bool(str((row or {}).get("run_error") or "").strip()) for row in turn_results if isinstance(row, dict))
    counts = workspace_manifest.get("counts")
    if not isinstance(counts, dict):
        counts = {}
    workspace_paths = workspace_manifest.get("all_paths")
    if not isinstance(workspace_paths, list):
        workspace_paths = []

    global_summary = {
        "total_turns": len(turn_map),
        "success_turns": success_turns,
        "any_run_error": any_run_error,
        "workspace_all_paths": len(workspace_paths),
        "workspace_exported_text_files": int(counts.get("exported_text_files") or 0),
        "workspace_exported_image_files": int(counts.get("exported_image_files") or 0),
        "workspace_unresolved_files": int(counts.get("unresolved_files") or 0),
    }

    raw_events_exists = raw_events_path.exists() if isinstance(raw_events_path, Path) else False
    return {
        "turn_results": turn_results,
        "turn_map": turn_map,
        "workspace_manifest": workspace_manifest,
        "workspace_paths": workspace_paths,
        "global_summary": global_summary,
        "raw_events_path": raw_events_path.as_posix() if isinstance(raw_events_path, Path) else "",
        "raw_events_exists": raw_events_exists,
    }


def _build_evidence(value: Any, probe: ProbeSpec) -> dict[str, Any]:
    capture_chars = 180
    if isinstance(probe.evidence_policy, dict):
        raw_limit = probe.evidence_policy.get("capture_chars")
        if isinstance(raw_limit, (int, float)):
            capture_chars = max(80, min(1000, int(raw_limit)))

    if isinstance(value, str):
        return {"snippet": value[:capture_chars]}
    if isinstance(value, list):
        preview = []
        for item in value[:10]:
            if isinstance(item, (str, int, float, bool)):
                preview.append(item)
            elif isinstance(item, dict):
                preview.append({k: item.get(k) for k in list(item.keys())[:4]})
            else:
                preview.append(str(item))
        return {"list_size": len(value), "preview": preview}
    if isinstance(value, dict):
        keys = list(value.keys())[:12]
        preview = {k: value.get(k) for k in keys}
        return {"keys": keys, "preview": preview}
    return {"value": value}


def _aggregate_results(dataset: ProbeDataset, probe_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(probe_results)
    passed = sum(1 for row in probe_results if row.get("passed"))
    failed = total - passed
    weighted_total = sum(float(row.get("weight") or 0.0) for row in probe_results)
    weighted_pass = sum(float(row.get("weight") or 0.0) for row in probe_results if row.get("passed"))
    weighted_score = (weighted_pass / weighted_total) if weighted_total > 0 else 0.0

    by_type: dict[str, dict[str, Any]] = {}
    for row in probe_results:
        key = str(row.get("probe_type") or "unknown")
        item = by_type.setdefault(key, {"total": 0, "passed": 0, "weighted_total": 0.0, "weighted_pass": 0.0})
        item["total"] += 1
        if row.get("passed"):
            item["passed"] += 1
        weight = float(row.get("weight") or 0.0)
        item["weighted_total"] += weight
        if row.get("passed"):
            item["weighted_pass"] += weight
    for key, item in by_type.items():
        wt = float(item.get("weighted_total") or 0.0)
        item["score"] = round((float(item.get("weighted_pass") or 0.0) / wt) if wt > 0 else 0.0, 4)
        item.pop("weighted_total", None)
        item.pop("weighted_pass", None)

    critical_failed = [row.get("probe_id") for row in probe_results if row.get("critical") and not row.get("passed")]

    return {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "total_probes": total,
        "passed_probes": passed,
        "failed_probes": failed,
        "weighted_score": round(weighted_score, 4),
        "critical_failed": critical_failed,
        "by_type": by_type,
    }


def evaluate_probes(
    dataset_path: Path,
    turn_results_path: Path,
    workspace_manifest_path: Path,
    raw_events_path: Path | None = None,
) -> dict[str, Any]:
    dataset = load_probe_dataset(dataset_path)
    ctx = _build_probe_context(turn_results_path, workspace_manifest_path, raw_events_path=raw_events_path)

    probe_results: list[dict[str, Any]] = []
    for probe in dataset.probes:
        target_value = _extract_target_value(ctx, probe)
        assertion_rows: list[dict[str, Any]] = []
        passed_all = True
        failures: list[str] = []
        error_message = ""

        for assertion in probe.assertions:
            try:
                passed, actual, detail = _eval_assertion(target_value, assertion)
                if not passed:
                    passed_all = False
                    failures.append(assertion.assert_type)
                assertion_rows.append(
                    {
                        "assert_type": assertion.assert_type,
                        "passed": bool(passed),
                        "expected": assertion.expect,
                        "actual": actual,
                        "detail": detail,
                        "negate": assertion.negate,
                        "field": assertion.field_path,
                    }
                )
            except Exception as exc:
                passed_all = False
                error_message = f"{exc.__class__.__name__}: {exc}"
                failures.append(assertion.assert_type)
                assertion_rows.append(
                    {
                        "assert_type": assertion.assert_type,
                        "passed": False,
                        "expected": assertion.expect,
                        "actual": None,
                        "detail": error_message,
                        "negate": assertion.negate,
                        "field": assertion.field_path,
                    }
                )

        probe_results.append(
            {
                "probe_id": probe.probe_id,
                "probe_type": probe.probe_type,
                "judge_mode": probe.judge_mode,
                "passed": bool(passed_all),
                "critical": bool(probe.critical),
                "weight": float(probe.weight),
                "priority": probe.priority,
                "description": probe.description,
                "target": {
                    "source": probe.target.source,
                    "turn": probe.target.turn,
                },
                "tags": probe.tags,
                "assertions": assertion_rows,
                "evidence": _build_evidence(target_value, probe),
                "failure_reason": "" if passed_all else f"failed_assertions={','.join(failures)}",
                "error": error_message,
            }
        )

    summary = _aggregate_results(dataset, probe_results)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "path": dataset_path.as_posix(),
            "dataset_id": dataset.dataset_id,
            "dataset_version": dataset.dataset_version,
            "description": dataset.description,
            "owner": dataset.owner,
        },
        "summary": summary,
        "results": probe_results,
    }


def write_probe_evaluation_md(
    path: Path,
    payload: dict[str, Any],
    max_fail_details: int = 20,
) -> None:
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        dataset = {}
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    results = payload.get("results")
    if not isinstance(results, list):
        results = []

    lines: list[str] = [
        "# Probe Evaluation",
        "",
        f"- generated_at: `{payload.get('generated_at') or ''}`",
        f"- dataset_id: `{dataset.get('dataset_id') or ''}`",
        f"- dataset_version: `{dataset.get('dataset_version') or ''}`",
        f"- dataset_path: `{dataset.get('path') or ''}`",
        "",
        "## Summary",
        "",
        f"- total_probes: `{summary.get('total_probes', 0)}`",
        f"- passed_probes: `{summary.get('passed_probes', 0)}`",
        f"- failed_probes: `{summary.get('failed_probes', 0)}`",
        f"- weighted_score: `{summary.get('weighted_score', 0)}`",
        f"- critical_failed: `{summary.get('critical_failed', [])}`",
        "",
        "## By Type",
        "",
        "| type | total | passed | score |",
        "| --- | --- | --- | --- |",
    ]
    by_type = summary.get("by_type")
    if isinstance(by_type, dict):
        for key in sorted(by_type.keys()):
            item = by_type.get(key)
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| `{key}` | `{item.get('total', 0)}` | `{item.get('passed', 0)}` | `{item.get('score', 0)}` |"
            )
    lines.append("")

    failed_rows = [row for row in results if isinstance(row, dict) and not row.get("passed")]
    lines.extend(["## Failed Probes", ""])
    if not failed_rows:
        lines.append("- none")
    else:
        for row in failed_rows[: max(1, max_fail_details)]:
            lines.extend(
                [
                    f"- `{row.get('probe_id')}` ({row.get('probe_type')})",
                    f"  - reason: `{row.get('failure_reason') or row.get('error') or ''}`",
                ]
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

