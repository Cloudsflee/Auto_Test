from __future__ import annotations

import copy
import importlib.util
import json
import shutil
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
    spec = importlib.util.spec_from_file_location("memory_scan_mod_migrate", SCAN_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {SCAN_SCRIPT_PATH.as_posix()}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


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


def _copy_and_reindex_session(
    *,
    src_run_root: Path,
    src_session_index: int,
    dst_run_root: Path,
    dst_session_index: int,
) -> dict[str, Any]:
    src_session_dir = src_run_root / "sessions" / f"session_{src_session_index:02d}"
    dst_session_dir = dst_run_root / "sessions" / f"session_{dst_session_index:02d}"
    if not src_session_dir.exists():
        raise RuntimeError(f"source session dir not found: {src_session_dir.as_posix()}")
    if dst_session_dir.exists():
        shutil.rmtree(dst_session_dir)
    shutil.copytree(src_session_dir, dst_session_dir)

    session_meta_path = dst_session_dir / "session_meta.json"
    meta = _load_json(session_meta_path, {})
    if not isinstance(meta, dict):
        meta = {}
    meta["session_index"] = dst_session_index
    meta["paths"] = {
        "raw_events": f"sessions/session_{dst_session_index:02d}/raw_events.jsonl",
        "dialogue": f"sessions/session_{dst_session_index:02d}/dialogue.md",
        "turn_results": f"sessions/session_{dst_session_index:02d}/run_data/turn_results.json",
        "probe_checks": f"sessions/session_{dst_session_index:02d}/run_data/probe_checks.json",
        "probe_timeline": f"sessions/session_{dst_session_index:02d}/run_data/probe_timeline.json",
        "compaction_events": f"sessions/session_{dst_session_index:02d}/run_data/compaction_events.json",
        "workspace_manifest": f"sessions/session_{dst_session_index:02d}/workspace/_manifest.json",
    }
    checkpoint = meta.get("checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint["checkpoint_file"] = f"sessions/session_{dst_session_index:02d}/run_data/session_checkpoint.json"
        meta["checkpoint"] = checkpoint
    session_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def migrate_excluded9_to_full_run(
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
    if not original_root.exists():
        raise RuntimeError(f"original run not found: {original_root.as_posix()}")
    if not retry_root.exists():
        raise RuntimeError(f"retry run not found: {retry_root.as_posix()}")

    original_summary = _load_json(original_root / "aggregate" / "summary.json", {})
    if not isinstance(original_summary, dict):
        raise RuntimeError("original summary.json invalid")
    original_sessions = original_summary.get("sessions")
    if not isinstance(original_sessions, list):
        raise RuntimeError("original summary sessions missing")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    migrated_run_id = f"{ts}_excluded9_official_migrated"
    migrated_root = AUTO_TEST_DIR / "results" / "memory_failure" / migrated_run_id
    sessions_root = migrated_root / "sessions"
    aggregate_root = migrated_root / "aggregate"
    sessions_root.mkdir(parents=True, exist_ok=True)
    aggregate_root.mkdir(parents=True, exist_ok=True)

    max_session_idx = max([0] + [int(x.get("session_index", 0)) for x in original_sessions] + [int(k) for k in rep_map.keys()])
    merged_sessions: list[dict[str, Any]] = []
    for final_idx in range(1, max_session_idx + 1):
        if final_idx in excluded:
            continue
        if final_idx in rep_map:
            src_run = retry_root
            src_idx = int(rep_map[final_idx])
        else:
            src_run = original_root
            src_idx = final_idx
        meta = _copy_and_reindex_session(
            src_run_root=src_run,
            src_session_index=src_idx,
            dst_run_root=migrated_root,
            dst_session_index=final_idx,
        )
        merged_sessions.append(meta)
    merged_sessions.sort(key=lambda x: int(x.get("session_index", 0)))

    base_config = original_summary.get("config") if isinstance(original_summary.get("config"), dict) else {}
    args = _build_args(scan, base_config, sessions=len(merged_sessions), warmup_sessions=0)

    chart_sessions = [s for s in merged_sessions if int(s.get("session_index", 0)) > args.warmup_sessions]
    chart_values = [
        int(s.get("first_failure_turn_effective"))
        for s in chart_sessions
        if isinstance(s.get("first_failure_turn_effective"), int)
    ]
    chart_sessions_without_failure = sum(1 for s in chart_sessions if s.get("first_failure_turn_effective") is None)
    chart_title = f"压缩失败轮次分布（已排除前 {args.warmup_sessions} 个预热会话）"
    chart_meta = scan._write_distribution_svg(aggregate_root / "failure_turn_distribution.svg", chart_values, chart_title)

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
    probe_outcome_rows = scan._collect_probe_outcome_rows(
        session_payloads=merged_sessions,
        result_root=migrated_root,
    )

    # Reuse already corrected excluded9 spotcheck result to avoid re-triggering LLM chain in migration.
    prior_summary = _load_json(original_root / "aggregate" / "excluded9_official" / "summary.json", {})
    prior_spotcheck = {}
    if isinstance(prior_summary, dict):
        prior_agg = prior_summary.get("aggregate") if isinstance(prior_summary.get("aggregate"), dict) else {}
        candidate = prior_agg.get("probe_spotcheck_summary")
        if isinstance(candidate, dict):
            prior_spotcheck = copy.deepcopy(candidate)
    if not prior_spotcheck:
        prior_spotcheck = {
            "status": "not_migrated_from_prior",
            "note": "spotcheck reused migration path missing; set to placeholder.",
        }

    aggregate["pipeline_health_summary"] = pipeline_health_summary
    aggregate["compression_effect_summary"] = compression_effect_summary
    aggregate["probe_runtime_summary"] = probe_runtime_summary
    aggregate["probe_spotcheck_summary"] = prior_spotcheck
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
    config_out["results_root"] = migrated_root.as_posix()
    config_out["migration_source_run_id"] = original_run_id
    config_out["migration_retry_run_id"] = retry_run_id
    config_out["migration_replacement_map"] = {str(k): int(v) for k, v in sorted(rep_map.items())}
    config_out["migration_excluded_session_indices"] = sorted(int(x) for x in excluded)

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

    (aggregate_root / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scan._build_result_markdown(aggregate_root / "summary.md", summary_payload)
    scan._build_llm_report(aggregate_root / "memory_failure_llm_report.md", summary_payload)

    (aggregate_root / "pipeline_health_summary.json").write_text(
        json.dumps(pipeline_health_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scan._build_pipeline_health_report_md(
        aggregate_root / "pipeline_health_report.md",
        {"pipeline_health_summary": pipeline_health_summary},
    )

    (aggregate_root / "compression_effect_summary.json").write_text(
        json.dumps(compression_effect_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scan._build_compression_effect_report_md(aggregate_root / "compression_effect_report.md", summary_payload)

    (aggregate_root / "probe_spotcheck_summary.json").write_text(
        json.dumps(prior_spotcheck, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scan._build_probe_spotcheck_report_md(
        aggregate_root / "probe_spotcheck_report.md",
        prior_spotcheck,
    )

    (aggregate_root / "statistical_validity_summary.json").write_text(
        json.dumps(statistical_validity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scan._build_statistical_validity_report_md(aggregate_root / "statistical_validity_proof.md", statistical_validity)

    (aggregate_root / "probe_checks_pass.json").write_text(
        json.dumps(probe_outcome_rows.get("pass", []), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (aggregate_root / "probe_checks_fail.json").write_text(
        json.dumps(probe_outcome_rows.get("fail", []), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scan._build_probe_outcome_report_md(
        path=aggregate_root / "probe_checks_pass.md",
        rows=probe_outcome_rows.get("pass", []),
        report_title="探针通过清单",
        report_intro="本文件列出所有判定为“通过/未确认遗忘”的 probe_check，包含判定理由和涉及原文内容。",
    )
    scan._build_probe_outcome_report_md(
        path=aggregate_root / "probe_checks_fail.md",
        rows=probe_outcome_rows.get("fail", []),
        report_title="探针失败清单",
        report_intro="本文件列出所有判定为“失败/确认遗忘”的 probe_check，包含判定理由和涉及原文内容。",
    )
    scan._build_results_readme_md(migrated_root / "README.md")

    (migrated_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "run_id": migrated_run_id,
                "results_root": migrated_root.as_posix(),
                "sessions_root": sessions_root.as_posix(),
                "aggregate_root": aggregate_root.as_posix(),
                "aggregate_summary_json": (aggregate_root / "summary.json").as_posix(),
                "aggregate_summary_md": (aggregate_root / "summary.md").as_posix(),
                "aggregate_llm_report_md": (aggregate_root / "memory_failure_llm_report.md").as_posix(),
                "aggregate_pipeline_health_json": (aggregate_root / "pipeline_health_summary.json").as_posix(),
                "aggregate_pipeline_health_md": (aggregate_root / "pipeline_health_report.md").as_posix(),
                "aggregate_compression_effect_json": (aggregate_root / "compression_effect_summary.json").as_posix(),
                "aggregate_compression_effect_md": (aggregate_root / "compression_effect_report.md").as_posix(),
                "aggregate_probe_spotcheck_json": (aggregate_root / "probe_spotcheck_summary.json").as_posix(),
                "aggregate_probe_spotcheck_md": (aggregate_root / "probe_spotcheck_report.md").as_posix(),
                "aggregate_probe_pass_json": (aggregate_root / "probe_checks_pass.json").as_posix(),
                "aggregate_probe_pass_md": (aggregate_root / "probe_checks_pass.md").as_posix(),
                "aggregate_probe_fail_json": (aggregate_root / "probe_checks_fail.json").as_posix(),
                "aggregate_probe_fail_md": (aggregate_root / "probe_checks_fail.md").as_posix(),
                "aggregate_statistical_validity_json": (aggregate_root / "statistical_validity_summary.json").as_posix(),
                "aggregate_statistical_validity_md": (aggregate_root / "statistical_validity_proof.md").as_posix(),
                "aggregate_distribution_chart": chart_meta.get("chart_path"),
                "results_readme_md": (migrated_root / "README.md").as_posix(),
                "session_count": len(merged_sessions),
                "session_dirs": [f"session_{i:02d}" for i in range(1, len(merged_sessions) + 1)],
                "migration_source_run_id": original_run_id,
                "migration_retry_run_id": retry_run_id,
                "migration_replacement_map": {str(k): int(v) for k, v in sorted(rep_map.items())},
                "migration_excluded_session_indices": sorted(int(x) for x in excluded),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return migrated_root


def main(argv: list[str]) -> int:
    original_run_id = "20260427_030702_efe62822"
    retry_run_id = "20260427_183146_fd8da596"
    if len(argv) >= 1 and str(argv[0]).strip():
        original_run_id = str(argv[0]).strip()
    if len(argv) >= 2 and str(argv[1]).strip():
        retry_run_id = str(argv[1]).strip()

    migrated_root = migrate_excluded9_to_full_run(
        original_run_id=original_run_id,
        retry_run_id=retry_run_id,
    )
    summary = _load_json(migrated_root / "aggregate" / "summary.json", {})
    sessions = summary.get("sessions") if isinstance(summary, dict) and isinstance(summary.get("sessions"), list) else []
    unverified = [
        int(s.get("session_index"))
        for s in sessions
        if not bool((s.get("compaction_completion") or {}).get("completion_verified"))
    ]
    print(f"[INFO] migrated_run_root={migrated_root.as_posix()}")
    print(f"[INFO] sessions={len(sessions)} unverified={len(unverified)} unverified_list={unverified}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
