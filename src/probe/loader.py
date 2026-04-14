from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ProbeAssertion, ProbeDataset, ProbeSpec, ProbeTarget


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _as_float(value: Any, default: float) -> float:
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


def _as_bool(value: Any, default: bool) -> bool:
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


def _normalize_probe_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"recall", "artifact", "continuation", "decision"}:
        return raw
    return "recall"


def _parse_assertion(obj: Any) -> ProbeAssertion:
    if not isinstance(obj, dict):
        obj = {}
    ext_value = obj.get("ext")
    ext_list = [str(x).strip().lower().lstrip(".") for x in ext_value] if isinstance(ext_value, list) else []
    return ProbeAssertion(
        assert_type=str(obj.get("assert_type") or "").strip(),
        expect=obj.get("expect"),
        flags=str(obj.get("flags") or "").strip(),
        negate=_as_bool(obj.get("negate"), False),
        field_path=str(obj.get("field") or obj.get("field_path") or "").strip(),
        ext=ext_list,
        gte=obj.get("gte"),
    )


def _parse_target(obj: Any) -> ProbeTarget:
    if not isinstance(obj, dict):
        obj = {}
    turn = obj.get("turn")
    turn_int = int(turn) if isinstance(turn, (int, float)) else 0
    return ProbeTarget(
        source=str(obj.get("source") or "").strip(),
        turn=max(0, turn_int),
    )


def _parse_probe(obj: Any) -> ProbeSpec:
    if not isinstance(obj, dict):
        obj = {}
    assertions_raw = obj.get("assertions")
    assertions = []
    if isinstance(assertions_raw, list):
        assertions = [_parse_assertion(it) for it in assertions_raw]
    tags = obj.get("tags")
    tags_list = [str(x).strip() for x in tags] if isinstance(tags, list) else []
    evidence_policy = obj.get("evidence_policy")
    if not isinstance(evidence_policy, dict):
        evidence_policy = {}
    return ProbeSpec(
        probe_id=str(obj.get("probe_id") or "").strip(),
        probe_type=_normalize_probe_type(obj.get("probe_type")),
        judge_mode=str(obj.get("judge_mode") or "deterministic").strip().lower(),
        target=_parse_target(obj.get("target")),
        assertions=assertions,
        weight=max(0.0, _as_float(obj.get("weight"), 1.0)),
        critical=_as_bool(obj.get("critical"), False),
        priority=str(obj.get("priority") or "normal").strip(),
        description=str(obj.get("description") or "").strip(),
        tags=tags_list,
        evidence_policy=evidence_policy,
    )


def load_probe_dataset(path: Path) -> ProbeDataset:
    if not path.exists():
        raise RuntimeError(f"probe dataset not found: {path.as_posix()}")
    payload = _safe_json_loads(path.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"probe dataset invalid json object: {path.as_posix()}")

    probes_raw = payload.get("probes")
    if not isinstance(probes_raw, list) or not probes_raw:
        raise RuntimeError(f"probe dataset requires non-empty probes: {path.as_posix()}")

    probes = [_parse_probe(it) for it in probes_raw]
    for idx, probe in enumerate(probes, start=1):
        if not probe.probe_id:
            raise RuntimeError(f"probe[{idx}] missing probe_id")
        if probe.judge_mode != "deterministic":
            raise RuntimeError(f"probe[{probe.probe_id}] unsupported judge_mode: {probe.judge_mode}")
        if not probe.target.source:
            raise RuntimeError(f"probe[{probe.probe_id}] missing target.source")
        if not probe.assertions:
            raise RuntimeError(f"probe[{probe.probe_id}] requires assertions")
        for aidx, assertion in enumerate(probe.assertions, start=1):
            if not assertion.assert_type:
                raise RuntimeError(f"probe[{probe.probe_id}] assertion[{aidx}] missing assert_type")

    return ProbeDataset(
        dataset_id=str(payload.get("dataset_id") or path.stem).strip(),
        dataset_version=str(payload.get("dataset_version") or "1.0.0").strip(),
        description=str(payload.get("description") or "").strip(),
        owner=str(payload.get("owner") or "").strip(),
        probes=probes,
    )
