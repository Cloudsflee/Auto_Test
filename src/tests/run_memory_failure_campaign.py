from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow direct execution from repo root.
AUTO_TEST_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = AUTO_TEST_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests.run_5turn_session_test import (  # noqa: E402
    apply_proxy_from_config,
    build_create_settings_json,
    build_exec_max_turns,
    build_llm_eval_config,
    build_persist_type,
    build_run_settings,
    build_user_simulator_config,
    load_config,
    normalize_authz,
    now_stamp,
    preflight_cleanup_test_processes,
    resolve_config_path,
    resolve_dotai_base_url,
)
from tests.run_memory_compression_failure_scan import (  # noqa: E402
    ScanArgs as BaseScanArgs,
    SharedRuntime,
    LLMEndpointConfig,
    _build_result_markdown,
    _build_stats,
    _estimate_focus_turn,
    _run_single_session,
    _write_distribution_svg,
)


@dataclass
class CampaignArgs:
    env: str
    seed_sessions: int
    focused_sessions: int
    focused_rounds: int
    parallel_sessions: int
    hard_max_turns: int
    probe_interval: int
    focus_window: int
    skip_preflight_cleanup: bool = False


def parse_args(argv: list[str]) -> CampaignArgs:
    parser = argparse.ArgumentParser(
        description="Run memory-failure campaign: 3-seed then 3 rounds of focused 3-session batches."
    )
    parser.add_argument("--env", type=str, default="", help="Runtime env (prod/test).")
    parser.add_argument("--seed-sessions", type=int, default=3, help="Seed batch session count.")
    parser.add_argument("--focused-sessions", type=int, default=3, help="Focused batch session count.")
    parser.add_argument("--focused-rounds", type=int, default=3, help="How many focused batches to run.")
    parser.add_argument("--parallel-sessions", type=int, default=3, help="Concurrent sessions per batch.")
    parser.add_argument("--hard-max-turns", type=int, default=0, help="Safety cap per session; 0=no cap.")
    parser.add_argument("--probe-interval", type=int, default=3, help="Probe interval for coarse mode.")
    parser.add_argument("--focus-window", type=int, default=3, help="Probe focus window around estimate.")
    parser.add_argument(
        "--skip-preflight-cleanup",
        action="store_true",
        help="Deprecated no-op. Preflight process cleanup is disabled.",
    )
    p = parser.parse_args(argv)
    return CampaignArgs(
        env=str(p.env or "").strip(),
        seed_sessions=max(1, min(30, int(p.seed_sessions))),
        focused_sessions=max(1, min(30, int(p.focused_sessions))),
        focused_rounds=max(1, min(10, int(p.focused_rounds))),
        parallel_sessions=max(1, min(16, int(p.parallel_sessions))),
        hard_max_turns=max(0, min(5000, int(p.hard_max_turns))),
        probe_interval=max(1, min(60, int(p.probe_interval))),
        focus_window=max(1, min(60, int(p.focus_window))),
        skip_preflight_cleanup=bool(p.skip_preflight_cleanup),
    )


def _calc_estimated_failure_turn_from_batch(summary_payload: dict[str, Any]) -> tuple[int | None, str]:
    sessions = summary_payload.get("sessions")
    if not isinstance(sessions, list):
        sessions = []

    raw_turns: list[int] = []
    executed_turns: list[int] = []
    for row in sessions:
        if not isinstance(row, dict):
            continue
        raw = row.get("first_failure_turn_raw")
        if isinstance(raw, int):
            raw_turns.append(raw)
        turns = row.get("turns_executed")
        if isinstance(turns, int):
            executed_turns.append(turns)

    if raw_turns:
        return _estimate_focus_turn(raw_turns), "seed_failure_median"

    if executed_turns:
        fallback = max(4, int(round(statistics.median(executed_turns))))
        return fallback, "seed_no_failure_fallback_executed_turn_median"

    return None, "seed_insufficient_data"


def _write_batch_summary(batch_root: Path, summary_payload: dict[str, Any]) -> None:
    aggregate_root = batch_root / "aggregate"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    (aggregate_root / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _build_result_markdown(aggregate_root / "summary.md", summary_payload)


def _run_batch(
    *,
    batch_name: str,
    campaign_root: Path,
    cfg: Any,
    headers: dict[str, str],
    run_settings: dict[str, Any],
    persist_type: int,
    exec_max_turns: int,
    create_settings_json: str | None,
    user_sim_cfg: Any,
    dotai_base_url: str,
    probe_llm_cfg: LLMEndpointConfig,
    focus_turn: int | None,
    session_count: int,
    parallel_sessions: int,
    args: CampaignArgs,
) -> dict[str, Any]:
    batch_root = campaign_root / batch_name
    sessions_root = batch_root / "sessions"
    aggregate_root = batch_root / "aggregate"
    sessions_root.mkdir(parents=True, exist_ok=True)
    aggregate_root.mkdir(parents=True, exist_ok=True)

    base_args = BaseScanArgs(
        env=args.env,
        sessions=session_count,
        warmup_sessions=0,
        hard_max_turns=args.hard_max_turns,
        probe_interval=args.probe_interval,
        focus_window=args.focus_window,
        parallel_sessions=parallel_sessions,
    )
    shared = SharedRuntime(
        cfg=cfg,
        headers=headers,
        run_settings=run_settings,
        persist_type=persist_type,
        exec_max_turns=exec_max_turns,
        create_settings_json=create_settings_json,
        user_sim_cfg=user_sim_cfg,
        dotai_base_url=dotai_base_url,
        sessions_root=sessions_root,
        args=base_args,
        probe_llm_cfg=probe_llm_cfg,
    )

    session_payloads: list[dict[str, Any]] = []
    if parallel_sessions <= 1 or session_count <= 1:
        for idx in range(1, session_count + 1):
            session_payloads.append(_run_single_session(shared, idx, focus_turn=focus_turn))
    else:
        with ThreadPoolExecutor(max_workers=parallel_sessions) as ex:
            futures: dict[Future[dict[str, Any]], int] = {}
            for idx in range(1, session_count + 1):
                futures[ex.submit(_run_single_session, shared, idx, focus_turn)] = idx
            for fut in as_completed(futures):
                session_payloads.append(fut.result())

    session_payloads_sorted = sorted(session_payloads, key=lambda x: int(x.get("session_index", 0)))
    values_effective: list[int] = []
    for row in session_payloads_sorted:
        v = row.get("first_failure_turn_effective")
        if isinstance(v, int):
            values_effective.append(v)

    chart_title = f"{batch_name} distribution (effective turn = raw)"
    chart_meta = _write_distribution_svg(
        aggregate_root / "failure_turn_distribution.svg",
        values_effective,
        chart_title,
    )
    raw_turns = [
        int(row["first_failure_turn_raw"])
        for row in session_payloads_sorted
        if isinstance(row.get("first_failure_turn_raw"), int)
    ]
    aggregate = {
        "sessions_total": len(session_payloads_sorted),
        "sessions_with_failure": sum(1 for row in session_payloads_sorted if isinstance(row.get("first_failure_turn_raw"), int)),
        "sessions_without_failure": sum(1 for row in session_payloads_sorted if row.get("first_failure_turn_raw") is None),
        "chart_samples": len(values_effective),
        "raw_turn_stats": _build_stats(raw_turns),
        "effective_turn_stats": _build_stats(values_effective),
    }
    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "batch_name": batch_name,
        "focus_turn": focus_turn,
        "config": {
            "runtime_env": cfg.selected_env,
            "session_count": session_count,
            "parallel_sessions": parallel_sessions,
            "hard_max_turns": args.hard_max_turns,
            "probe_interval": args.probe_interval,
            "focus_window": args.focus_window,
        },
        "chart": chart_meta,
        "aggregate": aggregate,
        "sessions": session_payloads_sorted,
    }
    _write_batch_summary(batch_root, summary_payload)
    return summary_payload


def _write_campaign_summary(campaign_root: Path, payload: dict[str, Any]) -> None:
    aggregate_root = campaign_root / "aggregate"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    (aggregate_root / "campaign_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Memory Failure Campaign Summary",
        "",
        f"- campaign_root: `{campaign_root.as_posix()}`",
        f"- estimated_failure_turn_raw: `{payload.get('estimated_failure_turn_raw')}`",
        f"- estimate_source: `{payload.get('estimate_source')}`",
        "",
        "## Batches",
        "",
    ]
    batches = payload.get("batches")
    if not isinstance(batches, list):
        batches = []
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        lines.append(
            f"- {batch.get('batch_name')}: "
            f"sessions={batch.get('sessions_total')} "
            f"with_failure={batch.get('sessions_with_failure')} "
            f"chart={batch.get('chart_path')}"
        )
    lines.extend(
        [
            "",
            "## Focused Combined",
            "",
            f"- chart_path: `{payload.get('focused_combined_chart_path', '')}`",
            f"- stats: `{json.dumps(payload.get('focused_combined_stats', {}), ensure_ascii=False)}`",
            "",
        ]
    )
    (aggregate_root / "campaign_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args(sys.argv[1:])
    skip_cleanup_env = str(os.getenv("AUTO_TEST_SKIP_PREFLIGHT_CLEANUP", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if args.skip_preflight_cleanup or skip_cleanup_env:
        print("[INFO] preflight_cleanup flag/env detected (deprecated no-op)")
    preflight_cleanup_test_processes("run_memory_failure_campaign")
    cfg_path = resolve_config_path()
    cfg = load_config(cfg_path, args.env)
    applied_proxy = apply_proxy_from_config(cfg)
    dotai_base_url = resolve_dotai_base_url(cfg)

    campaign_id = f"{now_stamp()}_{uuid.uuid4().hex[:8]}"
    campaign_root = AUTO_TEST_DIR / "results" / "memory_failure" / campaign_id
    campaign_root.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": normalize_authz(cfg.token),
        "uid": cfg.uid,
        "email": cfg.email,
        "Content-Type": "application/json",
    }
    run_settings = build_run_settings()
    persist_type = build_persist_type()
    exec_max_turns = build_exec_max_turns()
    create_settings_json = build_create_settings_json(run_settings)
    llm_eval_cfg = build_llm_eval_config(cfg)
    user_sim_cfg = build_user_simulator_config(cfg, llm_eval_cfg)
    if (not user_sim_cfg.enabled) or (not user_sim_cfg.base_url) or (not user_sim_cfg.model) or (not user_sim_cfg.api_key):
        raise RuntimeError("user_simulator not configured: check enabled/base_url/model/api_key.")
    probe_llm_cfg = LLMEndpointConfig(
        base_url=str(user_sim_cfg.base_url or "").strip(),
        model=str(user_sim_cfg.model or "").strip(),
        api_key=str(user_sim_cfg.api_key or "").strip(),
        timeout_sec=int(user_sim_cfg.timeout_sec),
    )
    if (not probe_llm_cfg.base_url) or (not probe_llm_cfg.model) or (not probe_llm_cfg.api_key):
        raise RuntimeError("probe_llm config missing: base_url/model/api_key")

    print(f"[INFO] campaign_id={campaign_id}")
    print(f"[INFO] campaign_root={campaign_root.as_posix()}")
    print(f"[INFO] runtime_env={cfg.selected_env}")
    print(f"[INFO] seed_sessions={args.seed_sessions}")
    print(f"[INFO] focused_sessions={args.focused_sessions}")
    print(f"[INFO] focused_rounds={args.focused_rounds}")
    print(f"[INFO] parallel_sessions={args.parallel_sessions}")
    print(f"[INFO] hard_max_turns={args.hard_max_turns}")
    if applied_proxy:
        print(f"[INFO] proxy_from_config={json.dumps(applied_proxy, ensure_ascii=False)}")
    else:
        print("[INFO] proxy_from_config=(none)")

    seed_summary = _run_batch(
        batch_name="seed_3sessions",
        campaign_root=campaign_root,
        cfg=cfg,
        headers=headers,
        run_settings=run_settings,
        persist_type=persist_type,
        exec_max_turns=exec_max_turns,
        create_settings_json=create_settings_json,
        user_sim_cfg=user_sim_cfg,
        dotai_base_url=dotai_base_url,
        probe_llm_cfg=probe_llm_cfg,
        focus_turn=None,
        session_count=args.seed_sessions,
        parallel_sessions=args.parallel_sessions,
        args=args,
    )
    estimated_raw, estimate_source = _calc_estimated_failure_turn_from_batch(seed_summary)
    print(f"[INFO] estimated_failure_turn_raw={estimated_raw} source={estimate_source}")

    focused_summaries: list[dict[str, Any]] = []
    for i in range(1, args.focused_rounds + 1):
        one = _run_batch(
            batch_name=f"focused_round_{i}",
            campaign_root=campaign_root,
            cfg=cfg,
            headers=headers,
            run_settings=run_settings,
            persist_type=persist_type,
            exec_max_turns=exec_max_turns,
            create_settings_json=create_settings_json,
            user_sim_cfg=user_sim_cfg,
            dotai_base_url=dotai_base_url,
            probe_llm_cfg=probe_llm_cfg,
            focus_turn=estimated_raw,
            session_count=args.focused_sessions,
            parallel_sessions=args.parallel_sessions,
            args=args,
        )
        focused_summaries.append(one)

    focused_values: list[int] = []
    for batch in focused_summaries:
        sessions = batch.get("sessions")
        if not isinstance(sessions, list):
            continue
        for row in sessions:
            if not isinstance(row, dict):
                continue
            v = row.get("first_failure_turn_effective")
            if isinstance(v, int):
                focused_values.append(v)

    focused_chart_meta = _write_distribution_svg(
        campaign_root / "aggregate" / "focused_combined_distribution.svg",
        focused_values,
        "Focused rounds combined distribution (effective turn = raw)",
    )

    campaign_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "campaign_id": campaign_id,
        "campaign_root": campaign_root.as_posix(),
        "runtime_env": cfg.selected_env,
        "estimated_failure_turn_raw": estimated_raw,
        "estimate_source": estimate_source,
        "batches": [],
        "focused_combined_chart_path": focused_chart_meta.get("chart_path"),
        "focused_combined_stats": _build_stats(focused_values),
    }

    for batch in [seed_summary, *focused_summaries]:
        agg = batch.get("aggregate")
        if not isinstance(agg, dict):
            agg = {}
        chart = batch.get("chart")
        if not isinstance(chart, dict):
            chart = {}
        campaign_summary["batches"].append(
            {
                "batch_name": batch.get("batch_name"),
                "sessions_total": agg.get("sessions_total"),
                "sessions_with_failure": agg.get("sessions_with_failure"),
                "sessions_without_failure": agg.get("sessions_without_failure"),
                "focus_turn": batch.get("focus_turn"),
                "chart_path": chart.get("chart_path"),
            }
        )

    _write_campaign_summary(campaign_root, campaign_summary)
    (campaign_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "campaign_id": campaign_id,
                "campaign_root": campaign_root.as_posix(),
                "seed_summary_json": (campaign_root / "seed_3sessions" / "aggregate" / "summary.json").as_posix(),
                "focused_round_1_summary_json": (campaign_root / "focused_round_1" / "aggregate" / "summary.json").as_posix(),
                "focused_round_2_summary_json": (campaign_root / "focused_round_2" / "aggregate" / "summary.json").as_posix(),
                "focused_round_3_summary_json": (campaign_root / "focused_round_3" / "aggregate" / "summary.json").as_posix(),
                "campaign_summary_json": (campaign_root / "aggregate" / "campaign_summary.json").as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[INFO] done campaign_root={campaign_root.as_posix()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
