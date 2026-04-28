from __future__ import annotations

import copy
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


AUTO_TEST_DIR = Path(__file__).resolve().parents[2]
SCAN_SCRIPT_PATH = Path(__file__).resolve().with_name("run_memory_compression_failure_scan.py")

# Final session index -> retry session index
DEFAULT_REPLACEMENT_MAP: dict[int, int] = {2: 1, 3: 2, 8: 3, 10: 4}
DEFAULT_EXCLUDED_SESSION_INDICES: set[int] = {10}


def _load_scan_module() -> Any:
    spec = importlib.util.spec_from_file_location("memory_scan_mod_recompute", SCAN_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {SCAN_SCRIPT_PATH.as_posix()}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int(v: Any, default: int = 0) -> int:
    try:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        s = str(v).strip()
        if not s:
            return default
        return int(float(s))
    except Exception:
        return default


def _normalize_paths_for_source(session_payload: dict[str, Any], source_run_root: Path) -> dict[str, Any]:
    cloned = copy.deepcopy(session_payload)
    paths = cloned.get("paths")
    if not isinstance(paths, dict):
        return cloned
    fixed: dict[str, Any] = {}
    for key, value in paths.items():
        raw = str(value or "").strip()
        if not raw:
            fixed[key] = value
            continue
        p = Path(raw)
        if p.is_absolute():
            fixed[key] = p.as_posix()
        else:
            fixed[key] = (source_run_root / p).as_posix()
    cloned["paths"] = fixed
    return cloned


def _build_args(scan: Any, base_config: dict[str, Any], *, sessions: int, warmup_sessions: int) -> Any:
    args = scan.parse_args([])
    args.env = str(base_config.get("runtime_env") or args.env or "test")
    args.sessions = sessions
    args.warmup_sessions = warmup_sessions
    args.hard_max_turns = _as_int(base_config.get("hard_max_turns"), args.hard_max_turns)
    args.probe_interval = _as_int(base_config.get("probe_interval"), args.probe_interval)
    args.focus_window = _as_int(base_config.get("focus_window"), args.focus_window)
    args.parallel_sessions = _as_int(base_config.get("parallel_sessions"), args.parallel_sessions)
    args.importance_tier = str(base_config.get("importance_tier") or args.importance_tier)
    args.turn_timeout_sec = _as_int(base_config.get("turn_timeout_sec"), args.turn_timeout_sec)
    args.session_wall_timeout_sec = _as_int(base_config.get("session_wall_timeout_sec"), args.session_wall_timeout_sec)
    args.min_post_samples_per_tier = _as_int(base_config.get("min_post_samples_per_tier"), args.min_post_samples_per_tier)
    args.probe_mode = str(base_config.get("probe_mode") or args.probe_mode)
    args.probe_cooldown_turns = _as_int(base_config.get("probe_cooldown_turns"), args.probe_cooldown_turns)
    args.probe_similarity_threshold = float(base_config.get("probe_similarity_threshold", args.probe_similarity_threshold))
    args.probe_recent_window = _as_int(base_config.get("probe_recent_window"), args.probe_recent_window)
    args.probe_regen_max_attempts = _as_int(base_config.get("probe_regen_max_attempts"), args.probe_regen_max_attempts)
    args.probe_post_compaction_window_offsets = scan._parse_probe_window_offsets(
        base_config.get("probe_post_compaction_window_offsets")
    )
    args.probe_max_density = float(base_config.get("probe_max_density", args.probe_max_density))
    args.probe_judge_mode = str(base_config.get("probe_judge_mode") or args.probe_judge_mode)
    args.probe_llm_skip_fastpath = bool(base_config.get("probe_llm_skip_fastpath", args.probe_llm_skip_fastpath))
    args.probe_llm_verify_always = bool(base_config.get("probe_llm_verify_always", args.probe_llm_verify_always))
    args.spotcheck_enabled = bool(base_config.get("spotcheck_enabled", args.spotcheck_enabled))
    args.spotcheck_sample_rate = float(base_config.get("spotcheck_sample_rate", args.spotcheck_sample_rate))
    args.spotcheck_min_samples = _as_int(base_config.get("spotcheck_min_samples"), args.spotcheck_min_samples)
    args.spotcheck_max_samples = _as_int(base_config.get("spotcheck_max_samples"), args.spotcheck_max_samples)
    args.spotcheck_seed = _as_int(base_config.get("spotcheck_seed"), args.spotcheck_seed)
    args.turn_gateway_404_retry_max = _as_int(
        base_config.get("turn_gateway_404_retry_max"), args.turn_gateway_404_retry_max
    )
    args.turn_gateway_404_retry_backoff_sec = float(
        base_config.get("turn_gateway_404_retry_backoff_sec", args.turn_gateway_404_retry_backoff_sec)
    )
    args.checkpoint_resume_enabled = bool(base_config.get("checkpoint_resume_enabled", args.checkpoint_resume_enabled))
    args.validity_min_verified_sessions = _as_int(
        base_config.get("validity_min_verified_sessions"), args.validity_min_verified_sessions
    )
    args.validity_max_session_run_error_rate = float(
        base_config.get("validity_max_session_run_error_rate", args.validity_max_session_run_error_rate)
    )
    args.validity_min_compaction_completion_rate = float(
        base_config.get("validity_min_compaction_completion_rate", args.validity_min_compaction_completion_rate)
    )
    args.validity_ci_max_half_width = float(base_config.get("validity_ci_max_half_width", args.validity_ci_max_half_width))
    return args


def _build_probe_llm_cfg(scan: Any, env_name: str) -> Any:
    cfg_path = scan.resolve_config_path()
    cfg = scan.load_config(cfg_path, env_name)
    scan.apply_proxy_from_config(cfg)
    llm_eval_cfg = scan.build_llm_eval_config(cfg)
    user_sim_cfg = scan.build_user_simulator_config(cfg, llm_eval_cfg)
    return scan.LLMEndpointConfig(
        base_url=str(user_sim_cfg.base_url or "").strip(),
        model=str(user_sim_cfg.model or "").strip(),
        api_key=str(user_sim_cfg.api_key or "").strip(),
        timeout_sec=int(user_sim_cfg.timeout_sec),
    )


def recompute_excluded9_official(
    *,
    original_run_id: str,
    retry_run_id: str,
    replacement_map: dict[int, int] | None = None,
    excluded_session_indices: set[int] | None = None,
) -> Path:
    scan = _load_scan_module()
    rep_map = dict(DEFAULT_REPLACEMENT_MAP if replacement_map is None else replacement_map)
    excluded = set(DEFAULT_EXCLUDED_SESSION_INDICES if excluded_session_indices is None else excluded_session_indices)

    original_root = AUTO_TEST_DIR / "results" / "memory_failure" / original_run_id
    retry_root = AUTO_TEST_DIR / "results" / "memory_failure" / retry_run_id
    aggregate_root = original_root / "aggregate"
    out_root = aggregate_root / "excluded9_official"
    out_root.mkdir(parents=True, exist_ok=True)

    original_summary = _load_json(aggregate_root / "summary.json")
    retry_summary = _load_json(retry_root / "aggregate" / "summary.json")

    original_map = {int(s.get("session_index")): s for s in original_summary.get("sessions", []) if isinstance(s, dict)}
    retry_map = {int(s.get("session_index")): s for s in retry_summary.get("sessions", []) if isinstance(s, dict)}

    max_session_idx = max(
        [0]
        + [int(k) for k in original_map.keys()]
        + [int(k) for k in rep_map.keys()]
    )
    merged_sessions: list[dict[str, Any]] = []
    for session_index in range(1, max_session_idx + 1):
        if session_index in excluded:
            continue
        if session_index in rep_map:
            src_index = int(rep_map[session_index])
            src_payload = retry_map.get(src_index)
            if not isinstance(src_payload, dict):
                raise RuntimeError(f"missing retry session payload: retry session {src_index}")
            payload = _normalize_paths_for_source(src_payload, retry_root)
        else:
            src_payload = original_map.get(session_index)
            if not isinstance(src_payload, dict):
                raise RuntimeError(f"missing original session payload: original session {session_index}")
            payload = _normalize_paths_for_source(src_payload, original_root)
        payload["session_index"] = session_index
        merged_sessions.append(payload)
    merged_sessions.sort(key=lambda x: int(x.get("session_index", 0)))

    base_config = original_summary.get("config") if isinstance(original_summary.get("config"), dict) else {}
    args = _build_args(scan, base_config, sessions=len(merged_sessions), warmup_sessions=0)
    probe_llm_cfg = _build_probe_llm_cfg(scan, args.env)

    chart_sessions = [s for s in merged_sessions if int(s.get("session_index", 0)) > args.warmup_sessions]
    chart_values = [
        int(s.get("first_failure_turn_effective"))
        for s in chart_sessions
        if isinstance(s.get("first_failure_turn_effective"), int)
    ]
    chart_sessions_without_failure = sum(1 for s in chart_sessions if s.get("first_failure_turn_effective") is None)
    chart_title = f"压缩失败轮次分布（已排除前 {args.warmup_sessions} 个预热会话）"
    chart_meta = scan._write_distribution_svg(out_root / "failure_turn_distribution.svg", chart_values, chart_title)

    aggregate = {
        "sessions_total": len(merged_sessions),
        "sessions_with_failure": sum(1 for s in merged_sessions if isinstance(s.get("first_failure_turn_raw"), int)),
        "sessions_without_failure": sum(1 for s in merged_sessions if s.get("first_failure_turn_raw") is None),
        "evaluated_sessions_for_chart": len(chart_sessions),
        "chart_samples": len(chart_values),
        "chart_sessions_without_failure": chart_sessions_without_failure,
        "raw_turn_stats_all_sessions": scan._build_stats(
            [int(s.get("first_failure_turn_raw")) for s in merged_sessions if isinstance(s.get("first_failure_turn_raw"), int)]
        ),
        "effective_turn_stats": scan._build_stats(chart_values),
    }
    pipeline_health_summary = scan._build_pipeline_health_summary(merged_sessions)
    compression_effect_summary = scan._build_compression_effect_summary_from_sessions(
        merged_sessions,
        min_post_samples_per_tier=args.min_post_samples_per_tier,
    )
    probe_runtime_summary = scan._build_probe_runtime_summary(merged_sessions)
    probe_spotcheck_summary = scan._build_probe_spotcheck_summary(
        session_payloads=merged_sessions,
        result_root=original_root,
        probe_llm_cfg=probe_llm_cfg,
        args=args,
    )
    probe_outcome_rows = scan._collect_probe_outcome_rows(
        session_payloads=merged_sessions,
        result_root=original_root,
    )
    aggregate["pipeline_health_summary"] = pipeline_health_summary
    aggregate["compression_effect_summary"] = compression_effect_summary
    aggregate["probe_runtime_summary"] = probe_runtime_summary
    aggregate["probe_spotcheck_summary"] = probe_spotcheck_summary
    aggregate["probe_outcome_summary"] = {
        "pass_count": len(probe_outcome_rows.get("pass", [])),
        "fail_count": len(probe_outcome_rows.get("fail", [])),
    }
    statistical_validity = scan._build_statistical_validity_summary(
        aggregate=aggregate,
        args=args,
    )
    aggregate["statistical_validity"] = statistical_validity

    config_out = copy.deepcopy(base_config)
    config_out["sessions"] = len(merged_sessions)
    config_out["warmup_sessions"] = 0
    config_out["results_root"] = original_root.as_posix()
    config_out["excluded9_official_root"] = out_root.as_posix()
    config_out["excluded9_official_replacement_map"] = {str(k): int(v) for k, v in sorted(rep_map.items())}
    config_out["excluded9_official_excluded_session_indices"] = sorted(int(x) for x in excluded)

    anchor_obj = original_summary.get("anchor") if isinstance(original_summary.get("anchor"), dict) else {}
    contract_profile_obj = original_summary.get("contract_profile")
    if isinstance(contract_profile_obj, dict):
        contract_profile_obj = {
            "profile_id": contract_profile_obj.get("profile_id"),
            "anchor_name": contract_profile_obj.get("anchor_name"),
            "memory_channel_expectation": contract_profile_obj.get("memory_channel_expectation"),
            "failure_policy": contract_profile_obj.get("failure_policy"),
            "judge_rubric": contract_profile_obj.get("judge_rubric"),
            "plant_text_template": contract_profile_obj.get("plant_text_template"),
            "probe_text_templates": contract_profile_obj.get("probe_text_templates"),
        }
    else:
        contract_profile_obj = {}

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config_out,
        "chart": chart_meta,
        "aggregate": aggregate,
        "anchor": anchor_obj,
        "contract_profile": contract_profile_obj,
        "sessions": merged_sessions,
    }

    (out_root / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scan._build_result_markdown(out_root / "summary.md", summary_payload)
    scan._build_llm_report(out_root / "memory_failure_llm_report.md", summary_payload)

    (out_root / "pipeline_health_summary.json").write_text(
        json.dumps(pipeline_health_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scan._build_pipeline_health_report_md(
        out_root / "pipeline_health_report.md",
        {"pipeline_health_summary": pipeline_health_summary},
    )

    (out_root / "compression_effect_summary.json").write_text(
        json.dumps(compression_effect_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scan._build_compression_effect_report_md(out_root / "compression_effect_report.md", summary_payload)

    (out_root / "probe_spotcheck_summary.json").write_text(
        json.dumps(probe_spotcheck_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scan._build_probe_spotcheck_report_md(out_root / "probe_spotcheck_report.md", probe_spotcheck_summary)

    (out_root / "statistical_validity_summary.json").write_text(
        json.dumps(statistical_validity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scan._build_statistical_validity_report_md(out_root / "statistical_validity_proof.md", statistical_validity)

    (out_root / "probe_checks_pass.json").write_text(
        json.dumps(probe_outcome_rows.get("pass", []), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_root / "probe_checks_fail.json").write_text(
        json.dumps(probe_outcome_rows.get("fail", []), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scan._build_probe_outcome_report_md(
        path=out_root / "probe_checks_pass.md",
        rows=probe_outcome_rows.get("pass", []),
        report_title="探针通过清单",
        report_intro="本文件列出所有判定为“通过/未确认遗忘”的 probe_check，包含判定理由和涉及原文内容。",
    )
    scan._build_probe_outcome_report_md(
        path=out_root / "probe_checks_fail.md",
        rows=probe_outcome_rows.get("fail", []),
        report_title="探针失败清单",
        report_intro="本文件列出所有判定为“失败/确认遗忘”的 probe_check，包含判定理由和涉及原文内容。",
    )
    scan._build_results_readme_md(out_root / "README.md")
    return out_root


def main(argv: list[str]) -> int:
    original_run_id = "20260427_030702_efe62822"
    retry_run_id = "20260427_183146_fd8da596"
    if len(argv) >= 1 and str(argv[0]).strip():
        original_run_id = str(argv[0]).strip()
    if len(argv) >= 2 and str(argv[1]).strip():
        retry_run_id = str(argv[1]).strip()

    out_root = recompute_excluded9_official(
        original_run_id=original_run_id,
        retry_run_id=retry_run_id,
    )
    summary = _load_json(out_root / "summary.json")
    sessions = summary.get("sessions", [])
    unverified = [
        int(s.get("session_index"))
        for s in sessions
        if not bool((s.get("compaction_completion") or {}).get("completion_verified"))
    ]
    print(f"[INFO] excluded9_official_recomputed={out_root.as_posix()}")
    print(f"[INFO] sessions={len(sessions)} unverified={len(unverified)} unverified_list={unverified}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
