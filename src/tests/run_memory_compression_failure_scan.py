from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import uuid
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

AUTO_TEST_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = AUTO_TEST_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests.run_5turn_session_test import (  # noqa: E402
    REQUIRED_NOTEBOOK_CLEAR_TEXT,
    apply_proxy_from_config,
    build_create_settings_json,
    build_exec_max_turns,
    build_llm_eval_config,
    build_persist_type,
    build_run_settings,
    build_user_simulator_config,
    create_session,
    execute_turn,
    load_config,
    normalize_authz,
    now_stamp,
    resolve_config_path,
    resolve_dotai_base_url,
)
from tests.user_simulator_engine import (  # noqa: E402
    GeneratedRole,
    UserSimulatorState,
    generate_role_for_session,
    generate_user_turn_with_simulator,
)
from tests.workspace_pipeline import export_workspace_view  # noqa: E402


@dataclass
class ScanArgs:
    env: str
    sessions: int
    warmup_sessions: int
    hard_max_turns: int
    probe_interval: int
    focus_window: int
    parallel_sessions: int


@dataclass
class SharedRuntime:
    cfg: Any
    headers: dict[str, str]
    run_settings: dict[str, Any]
    persist_type: int
    exec_max_turns: int
    create_settings_json: str | None
    user_sim_cfg: Any
    dotai_base_url: str
    sessions_root: Path
    args: ScanArgs


def parse_args(argv: list[str]) -> ScanArgs:
    parser = argparse.ArgumentParser(description="Scan memory-compression first-failure turn across sessions.")
    parser.add_argument("--env", type=str, default="", help="Runtime environment (prod/test).")
    parser.add_argument("--sessions", type=int, default=10, help="Session count.")
    parser.add_argument("--warmup-sessions", type=int, default=2, help="Warmup sessions excluded from chart.")
    parser.add_argument("--hard-max-turns", type=int, default=0, help="Safety cap per session; 0 = no cap.")
    parser.add_argument("--probe-interval", type=int, default=3, help="Coarse probe interval.")
    parser.add_argument("--focus-window", type=int, default=3, help="Focus window around estimated failure.")
    parser.add_argument("--parallel-sessions", type=int, default=1, help="Concurrent sessions (warmup always sequential).")
    parsed = parser.parse_args(argv)
    return ScanArgs(
        env=str(parsed.env or "").strip(),
        sessions=max(1, min(300, int(parsed.sessions))),
        warmup_sessions=max(0, min(299, int(parsed.warmup_sessions))),
        hard_max_turns=max(0, min(10000, int(parsed.hard_max_turns))),
        probe_interval=max(1, min(60, int(parsed.probe_interval))),
        focus_window=max(1, min(60, int(parsed.focus_window))),
        parallel_sessions=max(1, min(32, int(parsed.parallel_sessions))),
    )


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _assistant_has_probe(assistant_text: str, probe_id: str) -> bool:
    return bool(assistant_text and probe_id and _compact_text(probe_id) in _compact_text(assistant_text))


def _build_probe_id(session_idx: int) -> str:
    return f"MCF-{now_stamp()}-S{session_idx:02d}-{uuid.uuid4().hex[:8].upper()}"


def _estimate_focus_turn(failure_turns: list[int]) -> int | None:
    if not failure_turns:
        return None
    return max(4, int(round(statistics.median(failure_turns))))


def _build_scan_mode_label(focus_turn: int | None, focus_window: int) -> str:
    return "coarse" if focus_turn is None else f"focused@{focus_turn}+-{focus_window}"


def _should_probe_turn(turn_idx: int, coarse_interval: int, focus_turn: int | None, focus_window: int) -> bool:
    if turn_idx < 4:
        return False
    base_gap = turn_idx - 4
    if focus_turn is None:
        return base_gap % coarse_interval == 0
    left = max(4, focus_turn - focus_window)
    right = focus_turn + focus_window
    if turn_idx < left:
        return base_gap % coarse_interval == 0
    if left <= turn_idx <= right:
        return True
    after_interval = max(1, coarse_interval // 2)
    return (turn_idx - right) % after_interval == 0


def _build_probe_plant_prompt(probe_id: str) -> str:
    return (
        "我们先记一个业务信息，后面我会再确认。\n"
        f"本次活动内部编号：{probe_id}\n"
        "请你先复述这个编号，然后继续等待我下一步指令。"
    )


def _build_probe_check_prompt(turn_idx: int) -> str:
    return f"第{turn_idx}轮快速确认：刚才我给你的活动内部编号是什么？请只回复编号本身，不要解释。"


def _build_filler_prompt(session_idx: int, turn_idx: int, session_id: str) -> str:
    md_path = f"/workspace/advoo/{session_id}/memory_scan_s{session_idx:02d}.md"
    img_path = f"/workspace/advoo/{session_id}/memory_scan_s{session_idx:02d}_t{turn_idx:02d}.jpg"
    templates = [
        "写一段450-650字的口语化运营建议，具体、可执行。",
        "把上一条扩展成五段结构：目标、受众、策略、执行、风险。",
        f"整理为 markdown 并写入 `{md_path}`，再给3条摘要。",
        f"生成一张封面图并保存到 `{img_path}`，风格自然清爽。",
        "给8个更像真人会说的话的标题，并标注适用场景。",
        "改成给老板看的版本，350-500字，语气克制。",
    ]
    return templates[(turn_idx - 3) % len(templates)]


def _summarize_text(text: str, max_len: int = 120) -> str:
    raw = str(text or "").strip().replace("\n", " ")
    return raw if len(raw) <= max_len else raw[: max_len - 3] + "..."


def _write_distribution_svg(path: Path, values: list[int], title: str) -> dict[str, Any]:
    counts = Counter(values)
    if not counts:
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="420">'
            '<rect x="0" y="0" width="1000" height="420" fill="#f7f7f9"/>'
            f'<text x="40" y="60" font-size="24" fill="#222">{title}</text>'
            '<text x="40" y="110" font-size="18" fill="#555">No failure samples after warmup sessions.</text>'
            "</svg>"
        )
        path.write_text(svg + "\n", encoding="utf-8")
    else:
        xs = sorted(counts.keys())
        ys = [counts[x] for x in xs]
        width = 1000
        height = 520
        margin_l = 80
        margin_r = 40
        margin_t = 80
        margin_b = 90
        chart_w = width - margin_l - margin_r
        chart_h = height - margin_t - margin_b
        ymax = max(ys)
        bar_gap = 18
        bar_w = max(18, int((chart_w - bar_gap * (len(xs) + 1)) / max(1, len(xs))))
        parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect x="0" y="0" width="100%" height="100%" fill="#f8fafc"/>',
            f'<text x="40" y="42" font-size="22" fill="#111">{title}</text>',
            '<line x1="80" y1="430" x2="960" y2="430" stroke="#94a3b8" stroke-width="1.2"/>',
            '<line x1="80" y1="80" x2="80" y2="430" stroke="#94a3b8" stroke-width="1.2"/>',
        ]
        for idx, (xv, yv) in enumerate(zip(xs, ys)):
            x = margin_l + bar_gap + idx * (bar_w + bar_gap)
            h = int((yv / ymax) * (chart_h - 10)) if ymax > 0 else 0
            y = margin_t + chart_h - h
            parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="#3b82f6" rx="4" ry="4"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 8}" font-size="14" text-anchor="middle" fill="#1f2937">{yv}</text>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{margin_t + chart_h + 24}" font-size="13" text-anchor="middle" fill="#334155">{xv}</text>')
        parts.extend([
            '<text x="500" y="500" font-size="14" text-anchor="middle" fill="#475569">effective failure turn (raw - 2)</text>',
            '</svg>',
        ])
        path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return {
        "chart_type": "svg",
        "chart_path": path.as_posix(),
        "fallback_svg_path": path.as_posix(),
        "error": "",
    }


def _build_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "distribution": {}}
    counts = Counter(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 3),
        "median": statistics.median(values),
        "distribution": {str(k): v for k, v in sorted(counts.items())},
    }

def _build_result_markdown(path: Path, payload: dict[str, Any]) -> None:
    cfg = payload.get("config", {})
    aggregate = payload.get("aggregate", {})
    chart = payload.get("chart", {})
    sessions = payload.get("sessions", [])
    lines = [
        "# Memory Compression Failure Scan",
        "",
        f"- runtime_env: `{cfg.get('runtime_env', '')}`",
        f"- sessions: `{cfg.get('sessions', 0)}`",
        f"- warmup_sessions_excluded: `{cfg.get('warmup_sessions', 0)}`",
        f"- hard_max_turns: `{cfg.get('hard_max_turns', 0)}`",
        f"- probe_interval: `{cfg.get('probe_interval', 0)}`",
        f"- focus_window: `{cfg.get('focus_window', 0)}`",
        f"- parallel_sessions: `{cfg.get('parallel_sessions', 1)}`",
        f"- results_root: `{cfg.get('results_root', '')}`",
        "",
        "## Aggregate",
        "",
        f"- sessions_with_failure: `{aggregate.get('sessions_with_failure', 0)}`",
        f"- sessions_without_failure: `{aggregate.get('sessions_without_failure', 0)}`",
        f"- evaluated_sessions_for_chart: `{aggregate.get('evaluated_sessions_for_chart', 0)}`",
        f"- chart_samples: `{aggregate.get('chart_samples', 0)}`",
        f"- chart_type: `{chart.get('chart_type', '')}`",
        f"- chart_path: `{chart.get('chart_path', '')}`",
        "",
        "## Distribution (effective turn = raw turn - 2)",
        "",
        f"- stats: `{json.dumps(aggregate.get('effective_turn_stats', {}), ensure_ascii=False)}`",
        "",
        "## Sessions",
        "",
        "| session | mode | focus_turn | first_failure_raw | first_failure_effective | turns_executed | end_reason |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sessions:
        lines.append(
            f"| {row.get('session_index')} | {row.get('scan_mode', '')} | "
            f"{row.get('focus_turn_target', '')} | {row.get('first_failure_turn_raw', '')} | "
            f"{row.get('first_failure_turn_effective', '')} | {row.get('turns_executed', 0)} | "
            f"{row.get('end_reason', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_single_session(shared: SharedRuntime, session_idx: int, focus_turn: int | None) -> dict[str, Any]:
    session_dir = shared.sessions_root / f"session_{session_idx:02d}"
    run_data_dir = session_dir / "run_data"
    session_dir.mkdir(parents=True, exist_ok=True)
    run_data_dir.mkdir(parents=True, exist_ok=True)

    scan_mode = _build_scan_mode_label(focus_turn, shared.args.focus_window)
    probe_id = _build_probe_id(session_idx)
    probe_turns_executed: list[int] = []

    print(f"[INFO] session_start index={session_idx} mode={scan_mode} focus_turn={focus_turn}")

    try:
        generated_role: GeneratedRole = generate_role_for_session(shared.user_sim_cfg)
        print(f"[INFO] user_role session={session_idx} role_name={generated_role.role_name}")
    except Exception as exc:
        print(f"[WARN] user_role_fallback session={session_idx} reason={exc.__class__.__name__}")
        generated_role = GeneratedRole(
            role_name=f"memory-scan-user-{session_idx:02d}",
            role_profile=(
                "- role_name: memory-scan-user\n"
                "- identity: operation specialist\n"
                "- goal: iterate copy and image assets"
            ),
            role_json={
                "role_name": f"memory-scan-user-{session_idx:02d}",
                "identity": "operation specialist",
                "current_goal": "iterate copy and image assets",
            },
        )
    sim_state = UserSimulatorState(previous_response_id="")

    turn_results: list[dict[str, Any]] = []
    probe_checks: list[dict[str, Any]] = []
    first_failure_turn_raw: int | None = None
    create_trace_obj: dict[str, Any] = {}
    end_reason = "unknown"
    session_id = ""
    raw_events_path = session_dir / "raw_events.jsonl"
    latest_workspace_snapshot: dict[str, Any] | None = None

    with raw_events_path.open("w", encoding="utf-8") as raw_fp:
        session_id, create_trace = create_session(
            shared.cfg.base_url,
            shared.headers,
            persist_type=shared.persist_type,
            settings_json=shared.create_settings_json,
        )
        create_trace_obj = {
            "request_id": create_trace.request_id,
            "traceparent": create_trace.traceparent,
            "tracestate": create_trace.tracestate,
            "x_trace_id": create_trace.x_trace_id,
            "backend_trace_id": create_trace.backend_trace_id,
        }
        print(f"[INFO] session_created index={session_idx} session_id={session_id}")

        turn_idx = 1
        while True:
            if shared.args.hard_max_turns > 0 and turn_idx > shared.args.hard_max_turns:
                end_reason = "hard_max_turns_reached"
                break

            if turn_idx == 1:
                user_text = REQUIRED_NOTEBOOK_CLEAR_TEXT
                turn_kind = "reset_template"
            elif turn_idx == 2:
                user_text = _build_probe_plant_prompt(probe_id)
                turn_kind = "plant_probe"
            elif _should_probe_turn(
                turn_idx=turn_idx,
                coarse_interval=shared.args.probe_interval,
                focus_turn=focus_turn,
                focus_window=shared.args.focus_window,
            ):
                user_text = _build_probe_check_prompt(turn_idx)
                turn_kind = "probe_check"
                probe_turns_executed.append(turn_idx)
            else:
                turn_kind = "simulator_filler"
                try:
                    sim_text, should_stop, sim_state = generate_user_turn_with_simulator(
                        sim_cfg=shared.user_sim_cfg,
                        sim_state=sim_state,
                        results=turn_results,
                        role=generated_role,
                        turn_idx=turn_idx,
                        workspace_snapshot=latest_workspace_snapshot,
                    )
                    user_text = str(sim_text or "").strip()
                    if should_stop or not user_text:
                        user_text = _build_filler_prompt(session_idx, turn_idx, session_id)
                        turn_kind = "simulator_fallback_filler"
                except Exception as exc:
                    user_text = _build_filler_prompt(session_idx, turn_idx, session_id)
                    turn_kind = "simulator_error_fallback"
                    print(
                        "[WARN] simulator_turn_fallback "
                        f"session={session_idx} turn={turn_idx} reason={exc.__class__.__name__}"
                    )

            result = execute_turn(
                base_url=shared.cfg.base_url,
                headers=shared.headers,
                session_id=session_id,
                persist_type=shared.persist_type,
                exec_max_turns=shared.exec_max_turns,
                run_settings=shared.run_settings,
                user_text=user_text,
                turn_idx=turn_idx,
                raw_events_fp=raw_fp,
            )
            result["turn_kind"] = turn_kind
            turn_results.append(result)
            snapshot = result.get("workspace_snapshot")
            latest_workspace_snapshot = snapshot if isinstance(snapshot, dict) else None

            if turn_kind in {"plant_probe", "probe_check"}:
                passed = _assistant_has_probe(str(result.get("assistant_text") or ""), probe_id)
                probe_checks.append(
                    {
                        "turn": turn_idx,
                        "kind": turn_kind,
                        "probe_passed": passed,
                        "assistant_excerpt": _summarize_text(str(result.get("assistant_text") or "")),
                    }
                )
                if turn_kind == "probe_check" and not passed:
                    first_failure_turn_raw = turn_idx
                    end_reason = "probe_failure"
                    print(f"[INFO] first_failure_detected session={session_idx} turn={first_failure_turn_raw}")
                    break

            if result.get("run_error"):
                end_reason = "run_error"
                print(
                    "[WARN] turn_error "
                    f"session={session_idx} turn={turn_idx} run_error={result.get('run_error')}"
                )
                break

            turn_idx += 1

        if end_reason == "unknown":
            end_reason = "session_loop_ended"

    workspace_export = export_workspace_view(
        base_url=shared.cfg.base_url,
        headers=shared.headers,
        session_id=session_id,
        raw_events_path=raw_events_path,
        result_dir=session_dir,
        dotai_base_url=shared.dotai_base_url,
        user_id=shared.cfg.uid,
    )

    (run_data_dir / "turn_results.json").write_text(
        json.dumps(turn_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_data_dir / "probe_checks.json").write_text(
        json.dumps(
            {
                "probe_id": probe_id,
                "scan_mode": scan_mode,
                "focus_turn": focus_turn,
                "probe_turns_executed": probe_turns_executed,
                "checks": probe_checks,
                "first_failure_turn_raw": first_failure_turn_raw,
                "end_reason": end_reason,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    first_failure_turn_effective = first_failure_turn_raw - 2 if isinstance(first_failure_turn_raw, int) else None

    session_payload = {
        "session_index": session_idx,
        "session_id": session_id,
        "probe_id": probe_id,
        "scan_mode": scan_mode,
        "focus_turn_target": focus_turn,
        "generated_role_name": generated_role.role_name,
        "probe_turns_executed": probe_turns_executed,
        "first_failure_turn_raw": first_failure_turn_raw,
        "first_failure_turn_effective": first_failure_turn_effective,
        "turns_executed": len(turn_results),
        "end_reason": end_reason,
        "user_simulator": {
            "mode": shared.user_sim_cfg.mode,
            "model": shared.user_sim_cfg.model,
            "capability_mode": shared.user_sim_cfg.capability_mode,
        },
        "create_trace": create_trace_obj,
        "paths": {
            "raw_events": raw_events_path.relative_to(shared.sessions_root.parent).as_posix(),
            "turn_results": (run_data_dir / "turn_results.json").relative_to(shared.sessions_root.parent).as_posix(),
            "probe_checks": (run_data_dir / "probe_checks.json").relative_to(shared.sessions_root.parent).as_posix(),
            "workspace_manifest": (session_dir / "workspace" / "_manifest.json").relative_to(shared.sessions_root.parent).as_posix(),
        },
        "workspace_counts": (workspace_export.get("counts") if isinstance(workspace_export, dict) else {}),
    }
    (session_dir / "session_meta.json").write_text(
        json.dumps(session_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[INFO] session_done "
        f"index={session_idx} turns={len(turn_results)} "
        f"first_failure_raw={first_failure_turn_raw} end_reason={end_reason}"
    )
    return session_payload

def main() -> int:
    args = parse_args(sys.argv[1:])
    cfg_path = resolve_config_path()
    cfg = load_config(cfg_path, args.env)
    applied_proxy = apply_proxy_from_config(cfg)
    dotai_base_url = resolve_dotai_base_url(cfg)

    run_id = f"{now_stamp()}_{uuid.uuid4().hex[:8]}"
    result_root = AUTO_TEST_DIR / "results" / "memory_failure" / run_id
    sessions_root = result_root / "sessions"
    aggregate_root = result_root / "aggregate"
    sessions_root.mkdir(parents=True, exist_ok=True)
    aggregate_root.mkdir(parents=True, exist_ok=True)

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

    print(f"[INFO] config_path={cfg.source_path.as_posix()}")
    print(f"[INFO] runtime_env={cfg.selected_env}")
    print(f"[INFO] run_id={run_id}")
    print(f"[INFO] results_root={result_root.as_posix()}")
    print(f"[INFO] sessions={args.sessions}")
    print(f"[INFO] warmup_sessions={args.warmup_sessions}")
    print(f"[INFO] hard_max_turns={args.hard_max_turns}")
    print(f"[INFO] probe_interval={args.probe_interval}")
    print(f"[INFO] focus_window={args.focus_window}")
    print(f"[INFO] parallel_sessions={args.parallel_sessions}")
    print(f"[INFO] persist_type={persist_type}")
    print(f"[INFO] exec_max_turns={exec_max_turns}")
    print(f"[INFO] dotai_base_url={dotai_base_url or '(disabled)'}")
    print(f"[INFO] user_simulator_mode={user_sim_cfg.mode}")
    print(f"[INFO] user_simulator_model={user_sim_cfg.model}")
    if applied_proxy:
        print(f"[INFO] proxy_from_config={json.dumps(applied_proxy, ensure_ascii=False)}")
    else:
        print("[INFO] proxy_from_config=(none)")

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
        args=args,
    )

    session_payloads: list[dict[str, Any]] = []
    historical_failure_turns: list[int] = []

    warmup_count = min(args.warmup_sessions, args.sessions)
    for idx in range(1, warmup_count + 1):
        payload = _run_single_session(shared, idx, focus_turn=None)
        session_payloads.append(payload)
        ft = payload.get("first_failure_turn_raw")
        if isinstance(ft, int):
            historical_failure_turns.append(ft)

    next_idx = warmup_count + 1
    if next_idx <= args.sessions:
        if args.parallel_sessions <= 1:
            for idx in range(next_idx, args.sessions + 1):
                focus_turn = _estimate_focus_turn(historical_failure_turns)
                payload = _run_single_session(shared, idx, focus_turn=focus_turn)
                session_payloads.append(payload)
                ft = payload.get("first_failure_turn_raw")
                if isinstance(ft, int):
                    historical_failure_turns.append(ft)
        else:
            with ThreadPoolExecutor(max_workers=args.parallel_sessions) as executor:
                in_flight: dict[Future[dict[str, Any]], tuple[int, int | None]] = {}
                submit_idx = next_idx

                while submit_idx <= args.sessions or in_flight:
                    while submit_idx <= args.sessions and len(in_flight) < args.parallel_sessions:
                        focus_turn = _estimate_focus_turn(historical_failure_turns)
                        fut = executor.submit(_run_single_session, shared, submit_idx, focus_turn)
                        in_flight[fut] = (submit_idx, focus_turn)
                        print(
                            "[INFO] session_submitted "
                            f"index={submit_idx} focus_turn={focus_turn} in_flight={len(in_flight)}"
                        )
                        submit_idx += 1

                    done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                    for fut in done:
                        idx_done, focus_done = in_flight.pop(fut)
                        payload = fut.result()
                        session_payloads.append(payload)
                        ft = payload.get("first_failure_turn_raw")
                        if isinstance(ft, int):
                            historical_failure_turns.append(ft)
                        print(
                            "[INFO] session_collected "
                            f"index={idx_done} submitted_focus={focus_done} "
                            f"first_failure_raw={payload.get('first_failure_turn_raw')}"
                        )

    session_payloads_sorted = sorted(session_payloads, key=lambda x: int(x.get("session_index", 0)))

    chart_sessions = [s for s in session_payloads_sorted if int(s["session_index"]) > args.warmup_sessions]
    chart_values = [
        int(s["first_failure_turn_effective"])
        for s in chart_sessions
        if isinstance(s.get("first_failure_turn_effective"), int)
    ]
    chart_sessions_without_failure = sum(1 for s in chart_sessions if s.get("first_failure_turn_effective") is None)
    chart_title = f"Memory Compression Failure Turn Distribution (exclude first {args.warmup_sessions} warmup sessions)"
    chart_meta = _write_distribution_svg(aggregate_root / "failure_turn_distribution.svg", chart_values, chart_title)

    aggregate = {
        "sessions_total": len(session_payloads_sorted),
        "sessions_with_failure": sum(1 for s in session_payloads_sorted if isinstance(s.get("first_failure_turn_raw"), int)),
        "sessions_without_failure": sum(1 for s in session_payloads_sorted if s.get("first_failure_turn_raw") is None),
        "evaluated_sessions_for_chart": len(chart_sessions),
        "chart_samples": len(chart_values),
        "chart_sessions_without_failure": chart_sessions_without_failure,
        "raw_turn_stats_all_sessions": _build_stats([
            int(s["first_failure_turn_raw"]) for s in session_payloads_sorted if isinstance(s.get("first_failure_turn_raw"), int)
        ]),
        "effective_turn_stats": _build_stats(chart_values),
    }

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "runtime_env": cfg.selected_env,
            "config_source": cfg.source_path.as_posix(),
            "base_url": cfg.base_url,
            "dotai_base_url": dotai_base_url,
            "sessions": args.sessions,
            "warmup_sessions": args.warmup_sessions,
            "hard_max_turns": args.hard_max_turns,
            "probe_interval": args.probe_interval,
            "focus_window": args.focus_window,
            "parallel_sessions": args.parallel_sessions,
            "persist_type": persist_type,
            "exec_max_turns": exec_max_turns,
            "run_settings": run_settings,
            "create_settings_json": create_settings_json,
            "results_root": result_root.as_posix(),
        },
        "chart": chart_meta,
        "aggregate": aggregate,
        "sessions": session_payloads_sorted,
    }

    (aggregate_root / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _build_result_markdown(aggregate_root / "summary.md", summary_payload)

    (result_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "run_id": run_id,
                "results_root": result_root.as_posix(),
                "sessions_root": sessions_root.as_posix(),
                "aggregate_root": aggregate_root.as_posix(),
                "aggregate_summary_json": (aggregate_root / "summary.json").as_posix(),
                "aggregate_summary_md": (aggregate_root / "summary.md").as_posix(),
                "aggregate_distribution_chart": chart_meta.get("chart_path"),
                "session_count": len(session_payloads_sorted),
                "session_dirs": [f"session_{i:02d}" for i in range(1, len(session_payloads_sorted) + 1)],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[INFO] aggregate_summary={(aggregate_root / 'summary.json').as_posix()}")
    print(f"[INFO] aggregate_chart={chart_meta.get('chart_path')}")
    print(f"[INFO] done results_root={result_root.as_posix()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
