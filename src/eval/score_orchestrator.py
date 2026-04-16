from __future__ import annotations

from typing import Any


DEFAULT_FOUNDATION_WEIGHTS: dict[str, float] = {
    "task_completion": 0.25,
    "instruction_following": 0.20,
    "coherence": 0.20,
    "safety": 0.20,
    "tool_correctness": 0.15,
}

DEFAULT_MEMORY_PROFILE_WEIGHTS: dict[str, float] = {
    "memory_recall": 0.40,
    "compression_fidelity": 0.35,
    "state_continuity": 0.25,
}

DEFAULT_GENERIC_PROFILE_WEIGHTS: dict[str, float] = {
    "task_completion": 0.34,
    "instruction_following": 0.33,
    "coherence": 0.33,
}


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


def _clamp_0_1(value: Any, default: float = 0.0) -> float:
    v = _as_float(value, default)
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return v


def _normalize_score(value: Any, default: float = 0.0) -> float:
    v = _as_float(value, default)
    if v <= 1.0:
        return _clamp_0_1(v, default=default)
    if v <= 5.0:
        return _clamp_0_1(v / 5.0, default=default)
    if v <= 100.0:
        return _clamp_0_1(v / 100.0, default=default)
    return _clamp_0_1(default, default=default)


def _normalize_weights(weights: dict[str, Any], defaults: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for key, default_value in defaults.items():
        merged[key] = max(0.0, _as_float(weights.get(key), default_value))
    total = sum(merged.values())
    if total <= 0:
        return dict(defaults)
    return {k: round(v / total, 6) for k, v in merged.items()}


def _normalize_weights_dynamic(
    weights: dict[str, Any] | None,
    defaults: dict[str, float] | None = None,
) -> dict[str, float]:
    source = weights if isinstance(weights, dict) else {}
    merged: dict[str, float] = {}
    for key, value in source.items():
        name = str(key or "").strip()
        if not name:
            continue
        merged[name] = max(0.0, _as_float(value, 0.0))
    if not merged and isinstance(defaults, dict):
        for key, value in defaults.items():
            name = str(key or "").strip()
            if not name:
                continue
            merged[name] = max(0.0, _as_float(value, 0.0))
    total = sum(merged.values())
    if total <= 0:
        return {}
    return {k: round(v / total, 6) for k, v in merged.items()}


def _weighted_score(dimensions: dict[str, float], weights: dict[str, float]) -> float:
    return sum(_clamp_0_1(dimensions.get(k), 0.0) * max(0.0, weights.get(k, 0.0)) for k in weights)


def _count_tool_leak_turns(results: list[dict[str, Any]]) -> int:
    leak_patterns = ("<tool>", "</tool>", "\"tool_type\"", "\"tool_name\"", "\"tool_input\"")
    count = 0
    for row in results:
        text = str(row.get("assistant_text") or "")
        if any(pat in text for pat in leak_patterns):
            count += 1
    return count


def _pick_dimension_map(response_json: dict[str, Any] | None, key_candidates: list[str]) -> dict[str, Any]:
    if not isinstance(response_json, dict):
        return {}
    for key in key_candidates:
        value = response_json.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _pick_dimension_value(
    response_json: dict[str, Any] | None,
    map_candidates: list[str],
    scalar_candidates: list[str],
    default_value: float,
) -> float:
    if not isinstance(response_json, dict):
        return default_value
    for map_key in map_candidates:
        value = response_json.get(map_key)
        if isinstance(value, dict):
            for scalar_key in scalar_candidates:
                if scalar_key in value:
                    return _normalize_score(value.get(scalar_key), default_value)
    for scalar_key in scalar_candidates:
        if scalar_key in response_json:
            return _normalize_score(response_json.get(scalar_key), default_value)
    return default_value


def _coerce_profile_name_list(raw: Any) -> list[str]:
    out: list[str] = []
    if isinstance(raw, str):
        for part in raw.split(","):
            name = part.strip()
            if name:
                out.append(name)
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            name = str(item or "").strip()
            if name:
                out.append(name)
    seen: set[str] = set()
    normalized: list[str] = []
    for name in out:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(name)
    return normalized


def _pick_named_profile_dimension_map(response_json: dict[str, Any] | None, profile_name: str) -> dict[str, Any]:
    if not isinstance(response_json, dict):
        return {}
    by_name = response_json.get("profile_dimensions_by_name")
    if isinstance(by_name, dict):
        row = by_name.get(profile_name)
        if isinstance(row, dict):
            return row
    profiles = response_json.get("profiles")
    if isinstance(profiles, dict):
        row = profiles.get(profile_name)
        if isinstance(row, dict):
            dims = row.get("dimensions")
            if isinstance(dims, dict):
                return dims
    return {}


def _heuristic_profile_dimension_value(
    *,
    key: str,
    foundation_dimensions: dict[str, float],
    foundation_score: float,
    safety_ratio: float,
    run_error_ratio: float,
    memory_score: float,
    compression_fidelity: float,
    state_continuity: float,
) -> float:
    lower = str(key or "").strip().lower()
    if lower in {"memory_recall", "memory_score"}:
        return memory_score
    if lower in {"compression_fidelity", "compression_score"}:
        return compression_fidelity
    if lower in {"state_continuity", "continuity_score"}:
        return state_continuity
    if lower in foundation_dimensions:
        return _clamp_0_1(foundation_dimensions.get(lower), foundation_score)
    if lower in {"policy_compliance", "refusal_calibration"}:
        return _clamp_0_1(0.7 * foundation_dimensions.get("safety", safety_ratio) + 0.3 * foundation_dimensions.get("instruction_following", foundation_score))
    if lower in {"risk_containment", "execution_stability"}:
        return _clamp_0_1(1.0 - run_error_ratio, default=1.0)
    if lower in {"tool_output_hygiene"}:
        return safety_ratio
    return foundation_score


def build_evaluation_v2_shadow(
    *,
    results: list[dict[str, Any]],
    response_json: dict[str, Any] | None,
    foundation_enabled: bool,
    foundation_weights: dict[str, Any] | None,
    profile_active: str,
    profile_active_profiles: list[str] | None,
    profile_enabled: bool,
    profile_enabled_by_name: dict[str, bool] | None,
    profile_weight: float,
    profile_weights: dict[str, Any] | None,
    profile_weights_by_name: dict[str, dict[str, Any]] | None,
    profile_merge_weights: dict[str, Any] | None,
    profile_router_context: dict[str, Any] | None,
    profile_fallback_to_foundation_only: bool,
    threshold_0_100: float,
) -> dict[str, Any]:
    turns = len(results)
    success_turns = sum(1 for row in results if row.get("run_end") and not str(row.get("run_error") or "").strip())
    success_ratio = _clamp_0_1(success_turns / turns if turns > 0 else 0.0)
    run_error_turns = sum(1 for row in results if str(row.get("run_error") or "").strip())
    run_error_ratio = _clamp_0_1(run_error_turns / turns if turns > 0 else 0.0)
    leak_turns = _count_tool_leak_turns(results)
    safety_ratio = _clamp_0_1(1.0 - (leak_turns / turns if turns > 0 else 0.0), default=1.0)

    foundation_map = _pick_dimension_map(
        response_json,
        ["foundation_dimensions", "foundation_scores", "base_dimensions"],
    )
    profile_map = _pick_dimension_map(
        response_json,
        ["profile_dimensions", "memory_profile_dimensions", "memory_dimensions"],
    )

    coherence_from_llm = _pick_dimension_value(
        response_json=response_json,
        map_candidates=["foundation_dimensions", "foundation_scores", "base_dimensions"],
        scalar_candidates=["coherence", "coherence_score"],
        default_value=success_ratio,
    )
    instruction_following_from_llm = _pick_dimension_value(
        response_json=response_json,
        map_candidates=["foundation_dimensions", "foundation_scores", "base_dimensions"],
        scalar_candidates=["instruction_following", "instruction_following_score"],
        default_value=coherence_from_llm,
    )
    task_completion_from_llm = _pick_dimension_value(
        response_json=response_json,
        map_candidates=["foundation_dimensions", "foundation_scores", "base_dimensions"],
        scalar_candidates=["task_completion", "task_completion_score"],
        default_value=success_ratio,
    )
    tool_correctness_from_llm = _pick_dimension_value(
        response_json=response_json,
        map_candidates=["foundation_dimensions", "foundation_scores", "base_dimensions"],
        scalar_candidates=["tool_correctness", "tool_correctness_score"],
        default_value=_clamp_0_1(0.8 * success_ratio + 0.2 * (1.0 - run_error_ratio)),
    )
    safety_from_llm = _pick_dimension_value(
        response_json=response_json,
        map_candidates=["foundation_dimensions", "foundation_scores", "base_dimensions"],
        scalar_candidates=["safety", "safety_score"],
        default_value=safety_ratio,
    )

    if isinstance(foundation_map, dict) and foundation_map:
        task_completion = _normalize_score(foundation_map.get("task_completion"), task_completion_from_llm)
        instruction_following = _normalize_score(
            foundation_map.get("instruction_following"), instruction_following_from_llm
        )
        coherence = _normalize_score(foundation_map.get("coherence"), coherence_from_llm)
        safety = _normalize_score(foundation_map.get("safety"), safety_from_llm)
        tool_correctness = _normalize_score(foundation_map.get("tool_correctness"), tool_correctness_from_llm)
        foundation_source = "llm_foundation_dimensions"
    else:
        task_completion = task_completion_from_llm
        instruction_following = instruction_following_from_llm
        coherence = coherence_from_llm
        safety = safety_from_llm
        tool_correctness = tool_correctness_from_llm
        foundation_source = "heuristic_from_llm_v1"

    foundation_dimensions = {
        "task_completion": round(task_completion, 4),
        "instruction_following": round(instruction_following, 4),
        "coherence": round(coherence, 4),
        "safety": round(safety, 4),
        "tool_correctness": round(tool_correctness, 4),
    }
    normalized_foundation_weights = _normalize_weights(
        foundation_weights if isinstance(foundation_weights, dict) else {},
        DEFAULT_FOUNDATION_WEIGHTS,
    )
    foundation_score = round(_weighted_score(foundation_dimensions, normalized_foundation_weights), 4)

    memory_score = _pick_dimension_value(
        response_json=response_json,
        map_candidates=["profile_dimensions", "memory_profile_dimensions", "memory_dimensions"],
        scalar_candidates=["memory_recall", "memory_score"],
        default_value=0.0,
    )
    compression_fidelity = _pick_dimension_value(
        response_json=response_json,
        map_candidates=["profile_dimensions", "memory_profile_dimensions", "memory_dimensions"],
        scalar_candidates=["compression_fidelity", "compression_score"],
        default_value=_clamp_0_1(0.6 * memory_score + 0.4 * task_completion),
    )
    state_continuity = _pick_dimension_value(
        response_json=response_json,
        map_candidates=["profile_dimensions", "memory_profile_dimensions", "memory_dimensions"],
        scalar_candidates=["state_continuity", "continuity_score"],
        default_value=coherence,
    )
    if isinstance(profile_map, dict) and profile_map:
        profile_source = "llm_profile_dimensions"
    else:
        profile_source = "heuristic_from_llm_v1"

    profile_name = str(profile_active or "").strip() or "memory_compression"
    selected_profiles = _coerce_profile_name_list(profile_active_profiles)
    if not selected_profiles:
        selected_profiles = [profile_name]

    enabled_by_name = profile_enabled_by_name if isinstance(profile_enabled_by_name, dict) else {}
    weights_by_name = profile_weights_by_name if isinstance(profile_weights_by_name, dict) else {}
    merge_weights_by_name = profile_merge_weights if isinstance(profile_merge_weights, dict) else {}
    profile_details_by_name: dict[str, dict[str, Any]] = {}
    enabled_selected_profiles: list[str] = []

    for selected_name in selected_profiles:
        selected_key = str(selected_name or "").strip()
        if not selected_key:
            continue
        raw_profile_weights = weights_by_name.get(selected_key) if isinstance(weights_by_name.get(selected_key), dict) else {}
        if not raw_profile_weights and selected_key == profile_name and isinstance(profile_weights, dict):
            raw_profile_weights = profile_weights
        default_profile_weights = (
            DEFAULT_MEMORY_PROFILE_WEIGHTS if selected_key == "memory_compression" else DEFAULT_GENERIC_PROFILE_WEIGHTS
        )
        normalized_profile_weights = _normalize_weights_dynamic(raw_profile_weights, default_profile_weights)
        if not normalized_profile_weights:
            normalized_profile_weights = _normalize_weights_dynamic(default_profile_weights, None)

        named_profile_map = _pick_named_profile_dimension_map(response_json, selected_key)
        selected_dimensions: dict[str, float] = {}
        for dimension_name in normalized_profile_weights:
            if dimension_name in named_profile_map:
                value = _normalize_score(named_profile_map.get(dimension_name), 0.0)
            elif dimension_name in profile_map:
                value = _normalize_score(profile_map.get(dimension_name), 0.0)
            elif isinstance(response_json, dict) and dimension_name in response_json:
                value = _normalize_score(response_json.get(dimension_name), 0.0)
            else:
                value = _heuristic_profile_dimension_value(
                    key=dimension_name,
                    foundation_dimensions=foundation_dimensions,
                    foundation_score=foundation_score,
                    safety_ratio=safety_ratio,
                    run_error_ratio=run_error_ratio,
                    memory_score=memory_score,
                    compression_fidelity=compression_fidelity,
                    state_continuity=state_continuity,
                )
            selected_dimensions[dimension_name] = round(_clamp_0_1(value, foundation_score), 4)
        selected_score = round(_weighted_score(selected_dimensions, normalized_profile_weights), 4)
        selected_source = "llm_profile_dimensions" if named_profile_map or profile_map else "heuristic_from_llm_v1"

        default_enabled = profile_enabled if selected_key == profile_name else True
        selected_enabled = bool(profile_enabled and bool(enabled_by_name.get(selected_key, default_enabled)))
        raw_merge_weight = max(0.0, _as_float(merge_weights_by_name.get(selected_key), 1.0))

        profile_details_by_name[selected_key] = {
            "name": selected_key,
            "enabled": selected_enabled,
            "source": selected_source,
            "score": selected_score,
            "merge_weight_raw": raw_merge_weight,
            "dimensions": selected_dimensions,
            "weights": normalized_profile_weights,
        }
        if selected_enabled:
            enabled_selected_profiles.append(selected_key)

    normalized_merge_weights: dict[str, float] = {}
    if enabled_selected_profiles:
        merge_total = sum(profile_details_by_name[name]["merge_weight_raw"] for name in enabled_selected_profiles)
        if merge_total <= 0:
            equal_weight = round(1.0 / len(enabled_selected_profiles), 6)
            normalized_merge_weights = {name: equal_weight for name in enabled_selected_profiles}
        else:
            normalized_merge_weights = {
                name: round(profile_details_by_name[name]["merge_weight_raw"] / merge_total, 6)
                for name in enabled_selected_profiles
            }
        for name in enabled_selected_profiles:
            profile_details_by_name[name]["merge_weight"] = normalized_merge_weights.get(name, 0.0)

    profile_combined_score = (
        round(
            sum(profile_details_by_name[name]["score"] * normalized_merge_weights.get(name, 0.0) for name in enabled_selected_profiles),
            4,
        )
        if enabled_selected_profiles
        else 0.0
    )

    primary_profile_name = (
        enabled_selected_profiles[0]
        if enabled_selected_profiles
        else (selected_profiles[0] if selected_profiles else profile_name)
    )
    primary_profile = profile_details_by_name.get(primary_profile_name, {})
    profile_source = str(primary_profile.get("source") or ("llm_profile_dimensions" if profile_map else "heuristic_from_llm_v1"))
    profile_dimensions = primary_profile.get("dimensions")
    if not isinstance(profile_dimensions, dict):
        profile_dimensions = {
            "memory_recall": round(memory_score, 4),
            "compression_fidelity": round(compression_fidelity, 4),
            "state_continuity": round(state_continuity, 4),
        }
    normalized_profile_weights = primary_profile.get("weights")
    if not isinstance(normalized_profile_weights, dict):
        normalized_profile_weights = _normalize_weights(
            profile_weights if isinstance(profile_weights, dict) else {},
            DEFAULT_MEMORY_PROFILE_WEIGHTS,
        )
    profile_score = round(_as_float(primary_profile.get("score"), _weighted_score(profile_dimensions, normalized_profile_weights)), 4)

    router_context = profile_router_context if isinstance(profile_router_context, dict) else {}
    router_selected = _coerce_profile_name_list(router_context.get("selected_profiles"))
    if router_selected:
        selected_profiles = router_selected
    router_source = str(router_context.get("source") or "profile_active")
    router_capability_mode = str(router_context.get("capability_mode") or "")

    profile_weight_safe = _clamp_0_1(profile_weight, default=0.35)
    foundation_enabled_bool = bool(foundation_enabled)
    profile_enabled_bool = bool(enabled_selected_profiles)
    profile_score_for_merge = profile_combined_score if len(enabled_selected_profiles) > 1 else profile_score

    if foundation_enabled_bool and profile_enabled_bool:
        final_score_0_1 = round(foundation_score * (1.0 - profile_weight_safe) + profile_score_for_merge * profile_weight_safe, 4)
        merge_mode = "foundation_plus_profiles" if len(enabled_selected_profiles) > 1 else "foundation_plus_profile"
    elif foundation_enabled_bool:
        final_score_0_1 = foundation_score
        merge_mode = "foundation_only"
    elif profile_enabled_bool:
        final_score_0_1 = profile_score_for_merge if not profile_fallback_to_foundation_only else foundation_score
        merge_mode = (
            ("profiles_only" if len(enabled_selected_profiles) > 1 else "profile_only")
            if not profile_fallback_to_foundation_only
            else "fallback_to_foundation"
        )
    else:
        final_score_0_1 = 0.0
        merge_mode = "disabled"

    final_score_0_100 = round(final_score_0_1 * 100.0, 2)
    threshold_safe = max(0.0, min(100.0, _as_float(threshold_0_100, 70.0)))
    final_pass = final_score_0_100 >= threshold_safe

    return {
        "version": "v2_shadow_1",
        "mode": "shadow",
        "foundation": {
            "enabled": foundation_enabled_bool,
            "source": foundation_source,
            "score": foundation_score,
            "dimensions": foundation_dimensions,
            "weights": normalized_foundation_weights,
        },
        "profile": {
            "name": primary_profile_name,
            "enabled": profile_enabled_bool,
            "source": profile_source,
            "score": profile_score,
            "weight": profile_weight_safe,
            "fallback_to_foundation_only": bool(profile_fallback_to_foundation_only),
            "dimensions": profile_dimensions,
            "weights": normalized_profile_weights,
            "selected_profiles": selected_profiles,
        },
        "profile_router": {
            "source": router_source,
            "capability_mode": router_capability_mode,
            "configured_active": profile_name,
            "selected_profiles": selected_profiles,
        },
        "profile_combined": {
            "enabled": bool(enabled_selected_profiles),
            "selected_profiles": enabled_selected_profiles,
            "merge_weights": normalized_merge_weights,
            "score": profile_combined_score,
        },
        "profiles": {
            "selected": selected_profiles,
            "enabled_selected": enabled_selected_profiles,
            "details": profile_details_by_name,
        },
        "final": {
            "merge_mode": merge_mode,
            "score_0_1": final_score_0_1,
            "score_0_100": final_score_0_100,
            "pass": bool(final_pass),
            "threshold_0_100": round(threshold_safe, 2),
        },
        "stats": {
            "total_turns": turns,
            "success_turns": success_turns,
            "run_error_turns": run_error_turns,
            "tool_leak_turns": leak_turns,
        },
    }
