from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


AUTO_TEST_DIR = Path(__file__).resolve().parents[2]
SCAN_SCRIPT_PATH = Path(__file__).resolve().with_name("run_memory_compression_failure_scan.py")


def _load_scan_module() -> Any:
    spec = importlib.util.spec_from_file_location("memory_scan_mod", SCAN_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {SCAN_SCRIPT_PATH.as_posix()}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _session_dirs(sessions_root: Path) -> list[Path]:
    dirs = [p for p in sessions_root.iterdir() if p.is_dir() and p.name.startswith("session_")]
    dirs.sort(key=lambda p: p.name)
    return dirs


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


def _wilson_ci(success: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = success / total
    den = 1.0 + (z * z) / total
    center = (p + (z * z) / (2 * total)) / den
    margin = (z / den) * math.sqrt((p * (1 - p) / total) + ((z * z) / (4 * total * total)))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (round(lo, 4), round(hi, 4))


def _assess_significance(
    *,
    sessions_verified: int,
    independent_post_total: int,
    independent_post_by_tier: dict[str, int],
    min_post_samples_per_tier: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    if sessions_verified < 3:
        reasons.append(f"完成压缩验证的 session 仅 {sessions_verified} 个（建议至少 3 个才可做趋势判断）")
    if sessions_verified < 10:
        reasons.append(f"距离稳定统计建议值 10 个 session 仍不足（当前 {sessions_verified}）")

    insufficient_tiers = [
        tier for tier, n in independent_post_by_tier.items() if int(n) < int(min_post_samples_per_tier)
    ]
    if insufficient_tiers:
        reasons.append(
            "独立探针口径下，后压缩样本不足："
            + ", ".join(f"{tier}={independent_post_by_tier.get(tier, 0)}" for tier in insufficient_tiers)
            + f"（门槛 {min_post_samples_per_tier}）"
        )
    if independent_post_total < (min_post_samples_per_tier * 3):
        reasons.append(
            f"独立探针总样本 {independent_post_total} 偏小（参考下限 {min_post_samples_per_tier * 3}）"
        )

    if reasons:
        level = "不足（仅流程验证）"
        conclusion = "当前结果可用于验证链路是否跑通，不足以支持压缩效果的统计结论。"
    else:
        level = "基础可用（趋势观察）"
        conclusion = "样本达到基础门槛，可做趋势观察；若要稳定结论，仍建议扩展 session 数。"

    return {
        "level": level,
        "conclusion": conclusion,
        "reasons": reasons,
        "independent_post_total": independent_post_total,
        "independent_post_by_tier": independent_post_by_tier,
        "min_post_samples_per_tier": min_post_samples_per_tier,
    }


def build_cn_report(run_root: Path) -> dict[str, Any]:
    scan = _load_scan_module()

    summary_path = run_root / "aggregate" / "summary.json"
    summary = _load_json(summary_path, {})
    config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    min_post_samples = _as_int(config.get("min_post_samples_per_tier"), 3)

    sessions_root = run_root / "sessions"
    session_payloads: list[dict[str, Any]] = []
    session_cn_details: list[dict[str, Any]] = []

    independent_post_total = 0
    independent_post_by_tier = {"must_keep": 0, "should_keep": 0, "may_drop": 0}

    for session_dir in _session_dirs(sessions_root):
        meta = _load_json(session_dir / "session_meta.json", {})
        run_data_dir = session_dir / "run_data"
        turn_results = _load_json(run_data_dir / "turn_results.json", [])
        probe_checks_payload = _load_json(run_data_dir / "probe_checks.json", {})
        checks = probe_checks_payload.get("checks") if isinstance(probe_checks_payload.get("checks"), list) else []
        raw_events_path = session_dir / "raw_events.jsonl"

        compaction_events, compaction_completion = scan._detect_compaction_events(raw_events_path, turn_results)
        first_done_raw = _as_int(compaction_completion.get("first_verified_completion_turn"), 0)
        first_done_turn = first_done_raw if first_done_raw > 0 else None
        compaction_completion_verified = bool(compaction_completion.get("completion_verified"))

        failure_policy = meta.get("failure_policy") if isinstance(meta.get("failure_policy"), dict) else {}
        importance_tier = str(meta.get("importance_tier") or config.get("importance_tier") or "should_keep")
        session_gating = scan._build_session_gating_summary(
            probe_checks=checks,
            first_compaction_done_turn=first_done_turn,
            compaction_completion_verified=compaction_completion_verified,
            configured_tier=importance_tier,
            failure_policy=failure_policy,
        )

        anchor = meta.get("anchor") if isinstance(meta.get("anchor"), dict) else {}
        probe_timeline = scan._build_probe_timeline(checks, first_done_turn, anchor)
        health = scan._build_session_health_metrics(turn_results, compaction_events, compaction_completion)

        session_payload = dict(meta) if isinstance(meta, dict) else {}
        session_payload["compaction_completion"] = compaction_completion
        session_payload["session_gating"] = session_gating
        session_payload["health"] = health
        session_payload["stats"] = {
            "compaction_events": compaction_events,
            "probe_timeline": probe_timeline,
            "compaction_completion": compaction_completion,
        }
        session_payloads.append(session_payload)

        probe_checks = [x for x in checks if isinstance(x, dict) and str(x.get("kind") or "") == "probe_check"]
        post_probe_checks = [
            x
            for x in probe_checks
            if isinstance(first_done_turn, int) and _as_int(x.get("turn"), 0) > int(first_done_turn)
        ]
        independent_post_total += len(post_probe_checks)
        probe_tier_counter: Counter[str] = Counter(
            str(x.get("probe_tier") or "").strip().lower() for x in post_probe_checks
        )
        for tier in independent_post_by_tier:
            independent_post_by_tier[tier] += int(probe_tier_counter.get(tier, 0))

        judge_counter: Counter[str] = Counter()
        cause_counter: Counter[str] = Counter()
        for x in probe_checks:
            verify = x.get("verify_judge") if isinstance(x.get("verify_judge"), dict) else {}
            judge = str(verify.get("judge_state") or "unknown").strip().lower()
            cause = str(verify.get("cause_hint") or "unknown").strip().lower()
            judge_counter[judge] += 1
            cause_counter[cause] += 1

        session_cn_details.append(
            {
                "session_index": _as_int(session_payload.get("session_index"), 0),
                "session_id": str(session_payload.get("session_id") or ""),
                "turns_executed": _as_int(session_payload.get("turns_executed"), 0),
                "end_reason": str(session_payload.get("end_reason") or ""),
                "压缩完成判定": {
                    "完成": compaction_completion_verified,
                    "首个完成轮次": first_done_turn,
                    "完成轮次列表": compaction_completion.get("verified_completion_turns"),
                    "完成方法分布": compaction_completion.get("verified_completion_methods_by_turn"),
                },
                "探针统计": {
                    "probe_check总数": len(probe_checks),
                    "后压缩probe_check数(独立口径)": len(post_probe_checks),
                    "后压缩probe_check按tier(独立口径)": {
                        "must_keep": int(probe_tier_counter.get("must_keep", 0)),
                        "should_keep": int(probe_tier_counter.get("should_keep", 0)),
                        "may_drop": int(probe_tier_counter.get("may_drop", 0)),
                    },
                    "judge_state分布": {k: v for k, v in sorted(judge_counter.items())},
                    "cause_hint分布": {k: v for k, v in sorted(cause_counter.items())},
                },
                "链路健康": {
                    "run_error": bool(health.get("has_run_error_turn")),
                    "timeout": bool(health.get("has_timeout_turn")),
                    "stream_closed": bool(health.get("has_stream_closed_turn")),
                    "callback_round": bool(health.get("has_callback_round_turn")),
                    "multi_request": bool(health.get("has_multi_request_turn")),
                },
            }
        )

    pipeline_health = scan._build_pipeline_health_summary(session_payloads)
    compression_effect = scan._build_compression_effect_summary_from_sessions(
        session_payloads,
        min_post_samples_per_tier=min_post_samples,
    )

    sessions_verified = _as_int(compression_effect.get("sessions_included_verified_compaction"), 0)
    significance = _assess_significance(
        sessions_verified=sessions_verified,
        independent_post_total=independent_post_total,
        independent_post_by_tier=independent_post_by_tier,
        min_post_samples_per_tier=min_post_samples,
    )

    ci_by_tier: dict[str, dict[str, Any]] = {}
    post_total_by_tier = compression_effect.get("tier_post_sample_count")
    post_forgotten_by_tier = compression_effect.get("tier_post_forgotten_count")
    if isinstance(post_total_by_tier, dict) and isinstance(post_forgotten_by_tier, dict):
        for tier in ("must_keep", "should_keep", "may_drop"):
            n = _as_int(post_total_by_tier.get(tier), 0)
            k = _as_int(post_forgotten_by_tier.get(tier), 0)
            lo, hi = _wilson_ci(k, n)
            ci_by_tier[tier] = {
                "post_n": n,
                "post_forgotten_k": k,
                "forgotten_rate": round((k / n), 4) if n > 0 else 0.0,
                "wilson_95_ci": [lo, hi],
            }

    overview = {
        "run_id": run_root.name,
        "results_root": run_root.as_posix(),
        "sessions_total": _as_int(config.get("sessions"), len(session_payloads)),
        "sessions_verified_compaction": sessions_verified,
        "first_verified_completion_turns": [
            d.get("压缩完成判定", {}).get("首个完成轮次")
            for d in session_cn_details
            if isinstance(d.get("压缩完成判定"), dict)
            and isinstance(d.get("压缩完成判定", {}).get("首个完成轮次"), int)
        ],
    }

    conclusion_lines = [
        f"本次 run_id={run_root.name}，共 {overview['sessions_total']} 个 session。",
        f"按修正后的判定，本次压缩完成验证 session 数={sessions_verified}。",
        f"统计意义判定：{significance['level']}。{significance['conclusion']}",
    ]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_summary_json": summary_path.as_posix(),
        "analysis_version": "cn_postprocess_v1",
        "overview": overview,
        "pipeline_health_summary": pipeline_health,
        "compression_effect_summary_recomputed": compression_effect,
        "independent_probe_sample_view": {
            "post_compaction_probe_checks_total": independent_post_total,
            "post_compaction_probe_checks_by_tier": independent_post_by_tier,
            "note": "独立口径=按 probe_check 计数，不做 anchor_points 三倍展开。",
        },
        "statistical_significance": significance,
        "post_compaction_forgotten_rate_ci_95": ci_by_tier,
        "session_details": sorted(session_cn_details, key=lambda x: int(x.get("session_index") or 0)),
        "结论摘要": conclusion_lines,
    }


def _build_cn_markdown(report: dict[str, Any]) -> str:
    ov = report.get("overview") if isinstance(report.get("overview"), dict) else {}
    sig = report.get("statistical_significance") if isinstance(report.get("statistical_significance"), dict) else {}
    ph = report.get("pipeline_health_summary") if isinstance(report.get("pipeline_health_summary"), dict) else {}
    eff = (
        report.get("compression_effect_summary_recomputed")
        if isinstance(report.get("compression_effect_summary_recomputed"), dict)
        else {}
    )
    indep = report.get("independent_probe_sample_view") if isinstance(report.get("independent_probe_sample_view"), dict) else {}
    ci = report.get("post_compaction_forgotten_rate_ci_95") if isinstance(report.get("post_compaction_forgotten_rate_ci_95"), dict) else {}
    sessions = report.get("session_details") if isinstance(report.get("session_details"), list) else []

    lines = [
        "# 压缩测试中文解析报告",
        "",
        "## 1) 结论先看",
        "",
        f"- run_id: `{ov.get('run_id', '')}`",
        f"- session 总数: `{ov.get('sessions_total', 0)}`",
        f"- 压缩完成验证 session 数: `{ov.get('sessions_verified_compaction', 0)}`",
        f"- 统计意义等级: `{sig.get('level', '')}`",
        f"- 判定结论: `{sig.get('conclusion', '')}`",
        "",
        "## 2) 为什么这样判",
        "",
    ]
    for r in sig.get("reasons") or []:
        lines.append(f"- {r}")

    lines.extend(
        [
            "",
            "## 3) 两套样本口径（重点）",
            "",
            f"- 系统展开口径 tier_post_sample_count: `{json.dumps(eff.get('tier_post_sample_count', {}), ensure_ascii=False)}`",
            f"- 独立探针口径 post_probe_checks_by_tier: `{json.dumps(indep.get('post_compaction_probe_checks_by_tier', {}), ensure_ascii=False)}`",
            f"- 口径说明: `{indep.get('note', '')}`",
            "",
            "## 4) 压缩完成与链路健康",
            "",
            f"- compaction_completion_verified_rate: `{ph.get('compaction_completion_verified_rate', 0.0)}`",
            f"- session_timeout_rate: `{ph.get('session_timeout_rate', 0.0)}`",
            f"- session_stream_closed_rate: `{ph.get('session_stream_closed_rate', 0.0)}`",
            f"- session_run_error_rate: `{ph.get('session_run_error_rate', 0.0)}`",
            "",
            "## 5) 后压缩遗忘率区间（95% Wilson）",
            "",
        ]
    )
    for tier in ("must_keep", "should_keep", "may_drop"):
        row = ci.get(tier) if isinstance(ci.get(tier), dict) else {}
        lines.append(
            f"- {tier}: n={row.get('post_n', 0)}, k={row.get('post_forgotten_k', 0)}, "
            f"rate={row.get('forgotten_rate', 0.0)}, ci95={row.get('wilson_95_ci', [0, 0])}"
        )

    lines.extend(
        [
            "",
            "## 6) Session 逐条解析",
            "",
        ]
    )
    for s in sessions:
        comp = s.get("压缩完成判定") if isinstance(s.get("压缩完成判定"), dict) else {}
        probe = s.get("探针统计") if isinstance(s.get("探针统计"), dict) else {}
        health = s.get("链路健康") if isinstance(s.get("链路健康"), dict) else {}
        lines.extend(
            [
                f"### Session {s.get('session_index', 0)}",
                "",
                f"- session_id: `{s.get('session_id', '')}`",
                f"- turns_executed: `{s.get('turns_executed', 0)}`",
                f"- end_reason: `{s.get('end_reason', '')}`",
                f"- 压缩完成: `{comp.get('完成', False)}`",
                f"- 首个完成轮次: `{comp.get('首个完成轮次', None)}`",
                f"- 完成方法分布: `{json.dumps(comp.get('完成方法分布', {}), ensure_ascii=False)}`",
                f"- probe_check总数: `{probe.get('probe_check总数', 0)}`",
                f"- 后压缩probe_check数(独立口径): `{probe.get('后压缩probe_check数(独立口径)', 0)}`",
                f"- 后压缩probe_check按tier(独立口径): `{json.dumps(probe.get('后压缩probe_check按tier(独立口径)', {}), ensure_ascii=False)}`",
                f"- judge_state分布: `{json.dumps(probe.get('judge_state分布', {}), ensure_ascii=False)}`",
                f"- cause_hint分布: `{json.dumps(probe.get('cause_hint分布', {}), ensure_ascii=False)}`",
                f"- 链路健康: `{json.dumps(health, ensure_ascii=False)}`",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess memory-failure run artifacts into Chinese parsed reports.")
    parser.add_argument("--run-id", type=str, default="", help="Run id under auto_test/results/memory_failure.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    run_id = str(args.run_id or "").strip()
    if not run_id:
        raise SystemExit("run-id is required")
    run_root = AUTO_TEST_DIR / "results" / "memory_failure" / run_id
    if not run_root.exists():
        raise SystemExit(f"run not found: {run_root.as_posix()}")

    aggregate_root = run_root / "aggregate"
    aggregate_root.mkdir(parents=True, exist_ok=True)

    report = build_cn_report(run_root)
    json_path = aggregate_root / "cn_parsed_summary.json"
    md_path = aggregate_root / "cn_parsed_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_build_cn_markdown(report), encoding="utf-8")

    print(f"[INFO] cn_summary_json={json_path.as_posix()}")
    print(f"[INFO] cn_report_md={md_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
