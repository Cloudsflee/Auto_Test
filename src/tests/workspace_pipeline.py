from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import requests


WORKSPACE_PATH_PATTERN = re.compile(r"/workspace/[^\s\"'`<>(){}\\]+")
MARKDOWN_EXTS = {"md", "markdown", "txt"}
IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "svg", "heic", "heif"}


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _strip_bearer_prefix(value: str) -> str:
    raw = str(value or "").strip()
    if raw.lower().startswith("bearer "):
        return raw.split(" ", 1)[1].strip()
    return raw


def _build_dotai_fs_headers(headers: dict[str, str]) -> dict[str, str]:
    req_headers = {"Content-Type": "application/json"}
    authz = ""
    for key, value in headers.items():
        lower = key.lower()
        if lower in {"uid", "email", "authorization"} and isinstance(value, str) and value.strip():
            req_headers[key] = value
            if lower == "authorization":
                authz = value
    raw_token = _strip_bearer_prefix(authz)
    if raw_token:
        req_headers["token"] = raw_token
    return req_headers


def _dotai_response_code_is_success(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    code = payload.get("code")
    if code is None:
        return True
    if isinstance(code, bool):
        return bool(code)
    if isinstance(code, (int, float)):
        return int(code) in {0, 200}
    if isinstance(code, str):
        return code.strip() in {"0", "200"}
    return False


def _dotai_pick_result(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    if result is not None:
        return result
    return payload.get("data")


def _workspace_path_ext(path: str) -> str:
    p = str(path or "").strip()
    if "." not in p:
        return ""
    return p.rsplit(".", 1)[-1].lower()


def extract_workspace_paths(text: str) -> list[str]:
    raw = str(text or "")
    if not raw:
        return []
    out: set[str] = set()
    for match in WORKSPACE_PATH_PATTERN.findall(raw):
        cleaned = match.strip().rstrip(".,;:!?)]}\"'")
        if cleaned:
            out.add(cleaned)
    return sorted(out)


def _looks_like_workspace_file(path: str) -> bool:
    p = str(path or "").strip()
    return p.startswith("/workspace/") and not p.endswith("/")


def _is_markdown_like_file(path: str) -> bool:
    return _workspace_path_ext(path) in MARKDOWN_EXTS


def _is_image_like_file(path: str) -> bool:
    return _workspace_path_ext(path) in IMAGE_EXTS


def _to_compact_single_line(text: str, max_chars: int = 220) -> str:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 3] + "..."


def _infer_image_hint_from_path(path: str) -> str:
    try:
        file_name = PurePosixPath(str(path or "")).name
    except Exception:
        file_name = str(path or "")
    stem = file_name.rsplit(".", 1)[0]
    tokens = [t for t in re.split(r"[_\-.]+", stem) if t]
    filtered: list[str] = []
    for tok in tokens:
        if re.fullmatch(r"[0-9a-fA-F]{6,}", tok):
            continue
        if re.fullmatch(r"\d{4,}", tok):
            continue
        filtered.append(tok)
    if not filtered:
        return "generated image"
    return " ".join(filtered[:8])


def _normalize_event_text_by_path(raw: dict[str, str] | None) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        p = str(key or "").strip()
        if not _looks_like_workspace_file(p):
            continue
        if not isinstance(value, str):
            continue
        out[p] = value
    return out


def _normalize_event_image_paths(raw: list[str] | None) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        p = str(item or "").strip()
        if not _looks_like_workspace_file(p):
            continue
        if not _is_image_like_file(p):
            continue
        out.append(p)
    return sorted(set(out))


def _parse_files_api_items(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data = payload
    if "code" in payload:
        if payload.get("code") not in (0, 200):
            return []
        data = payload.get("data")
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return [it for it in items if isinstance(it, dict)]
    if isinstance(data, list):
        return [it for it in data if isinstance(it, dict)]
    return []


def _list_workspace_paths_from_files_api(
    base_url: str,
    headers: dict[str, str],
    session_id: str,
) -> tuple[list[str], str]:
    if not base_url or not session_id:
        return [], "missing_base_or_session"

    req_headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
    candidates = [
        (
            f"{base_url}/v1/files/session/{session_id}/workspace/advoo/{session_id}/",
            f"/workspace/advoo/{session_id}",
        ),
        (
            f"{base_url}/v1/files/session/{session_id}/workspace/advoo/{session_id}",
            f"/workspace/advoo/{session_id}",
        ),
        (
            f"{base_url}/v1/files/session/{session_id}/workspace/",
            "/workspace",
        ),
        (
            f"{base_url}/v1/files/session/{session_id}/workspace",
            "/workspace",
        ),
    ]

    last_err = "not_tried"
    for url, base_virtual_path in candidates:
        try:
            resp = requests.get(url, headers=req_headers, timeout=15)
        except Exception as exc:
            last_err = f"request_error:{exc.__class__.__name__}"
            continue
        if resp.status_code < 200 or resp.status_code >= 300:
            last_err = f"http_{resp.status_code}"
            continue

        payload = _safe_json_loads(resp.text if isinstance(resp.text, str) else "")
        items = _parse_files_api_items(payload)
        if not items:
            last_err = "empty_items"
            continue

        paths: list[str] = []
        for item in items:
            name = str(item.get("name") or item.get("fileName") or "").strip()
            if not name:
                continue
            base = base_virtual_path.rstrip("/")
            full_path = f"{base}/{name}"
            item_type = str(item.get("type") or item.get("entryType") or "").strip().lower()
            if item_type in {"dir", "folder"} and not full_path.endswith("/"):
                full_path += "/"
            paths.append(full_path)
        if paths:
            return sorted(set(paths)), ""
        last_err = "empty_paths"
    return [], last_err


def build_workspace_snapshot(
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    event_paths: list[str],
    event_text_by_path: dict[str, str] | None = None,
    event_image_paths: list[str] | None = None,
) -> dict[str, Any]:
    api_paths, api_error = _list_workspace_paths_from_files_api(base_url, headers, session_id)
    merged_paths = sorted(set(api_paths) | set(event_paths))

    file_paths = [p for p in merged_paths if _looks_like_workspace_file(p)]
    md_files = [p for p in file_paths if _is_markdown_like_file(p)]
    image_files = [p for p in file_paths if _is_image_like_file(p)]
    event_file_paths = [p for p in event_paths if _looks_like_workspace_file(p)]
    event_md_files = [p for p in event_file_paths if _is_markdown_like_file(p)]
    event_img_files = [p for p in event_file_paths if _is_image_like_file(p)]
    normalized_text_by_path = _normalize_event_text_by_path(event_text_by_path)
    normalized_event_images = _normalize_event_image_paths(event_image_paths)

    recent_image_files = sorted(set(event_img_files) | set(normalized_event_images))
    recent_md_files = sorted(set(event_md_files) | set(normalized_text_by_path.keys()))
    recent_touched_paths = sorted(
        set(event_file_paths) | set(normalized_text_by_path.keys()) | set(recent_image_files)
    )

    text_artifacts: list[dict[str, Any]] = []
    for path in sorted(normalized_text_by_path.keys())[:12]:
        text = normalized_text_by_path[path]
        text_artifacts.append(
            {
                "path": path,
                "chars": len(text),
                "lines": max(1, text.count("\n") + 1),
                "excerpt": _to_compact_single_line(text, max_chars=220),
            }
        )

    image_artifacts: list[dict[str, Any]] = []
    for path in recent_image_files[:12]:
        image_artifacts.append(
            {
                "path": path,
                "hint": _infer_image_hint_from_path(path),
            }
        )

    source = "none"
    if api_paths and event_paths:
        source = "files_api+events"
    elif api_paths:
        source = "files_api"
    elif event_paths:
        source = "events_fallback"

    return {
        "source": source,
        "api_error": api_error,
        "all_paths": merged_paths[:200],
        "md_files": md_files[:80],
        "image_files": image_files[:80],
        "recent_touched_paths": recent_touched_paths[:80],
        "recent_md_files": recent_md_files[:40],
        "recent_image_files": recent_image_files[:40],
        "recent_artifacts": {
            "text_files": text_artifacts,
            "image_files": image_artifacts,
        },
        "counts": {
            "all_paths": len(merged_paths),
            "md_files": len(md_files),
            "image_files": len(image_files),
            "recent_touched_paths": len(recent_touched_paths),
            "recent_md_files": len(recent_md_files),
            "recent_image_files": len(recent_image_files),
            "recent_text_artifacts": len(text_artifacts),
        },
    }


def render_workspace_snapshot_for_user_sim(snapshot: dict[str, Any] | None) -> str:
    if not isinstance(snapshot, dict):
        return "(workspace snapshot unavailable)"
    counts = snapshot.get("counts")
    if not isinstance(counts, dict):
        counts = {}
    md_files = snapshot.get("md_files")
    if not isinstance(md_files, list):
        md_files = []
    image_files = snapshot.get("image_files")
    if not isinstance(image_files, list):
        image_files = []
    all_paths = snapshot.get("all_paths")
    if not isinstance(all_paths, list):
        all_paths = []
    recent_touched_paths = snapshot.get("recent_touched_paths")
    if not isinstance(recent_touched_paths, list):
        recent_touched_paths = []
    recent_artifacts = snapshot.get("recent_artifacts")
    if not isinstance(recent_artifacts, dict):
        recent_artifacts = {}
    text_artifacts = recent_artifacts.get("text_files")
    if not isinstance(text_artifacts, list):
        text_artifacts = []
    image_artifacts = recent_artifacts.get("image_files")
    if not isinstance(image_artifacts, list):
        image_artifacts = []

    lines = [
        f"source={snapshot.get('source') or 'unknown'}",
        f"total_paths={counts.get('all_paths', len(all_paths))}",
        f"md_files={counts.get('md_files', len(md_files))}",
        f"image_files={counts.get('image_files', len(image_files))}",
        f"turn_touched_paths={counts.get('recent_touched_paths', len(recent_touched_paths))}",
    ]
    if recent_touched_paths:
        lines.append("turn_touched_preview:")
        lines.extend(f"- {p}" for p in recent_touched_paths[:12])
    if text_artifacts:
        lines.append("turn_text_artifacts:")
        for item in text_artifacts[:8]:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "").strip()
            chars = item.get("chars")
            excerpt = _to_compact_single_line(str(item.get("excerpt") or ""), max_chars=180)
            lines.append(f"- {path} | chars={chars} | excerpt={excerpt}")
    if image_artifacts:
        lines.append("turn_image_artifacts:")
        for item in image_artifacts[:8]:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "").strip()
            hint = _to_compact_single_line(str(item.get("hint") or ""), max_chars=120)
            lines.append(f"- {path} | hint={hint}")
    if md_files:
        lines.append("md_preview:")
        lines.extend(f"- {p}" for p in md_files[:12])
    if image_files:
        lines.append("image_preview:")
        lines.extend(f"- {p}" for p in image_files[:12])
    if not md_files and not image_files and all_paths:
        lines.append("path_preview:")
        lines.extend(f"- {p}" for p in all_paths[:12])
    if not all_paths:
        lines.append("(workspace paths empty or unavailable)")
    return "\n".join(lines)


def _workspace_virtual_to_rel_path(virtual_path: str) -> Path:
    raw = str(virtual_path or "").strip()
    if raw.startswith("/"):
        raw = raw[1:]
    if raw.startswith("workspace/"):
        raw = raw[len("workspace/") :]
    parts = [p for p in PurePosixPath(raw).parts if p and p not in {".", ".."}]
    if not parts:
        parts = ["_unknown_path.txt"]
    return Path(*parts)


def _parse_tool_args_to_dict(args_raw: Any) -> dict[str, Any]:
    if isinstance(args_raw, dict):
        return args_raw
    if isinstance(args_raw, str):
        parsed = _safe_json_loads(args_raw)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _collect_workspace_artifacts_from_raw_events(raw_events_path: Path) -> dict[str, Any]:
    all_paths: set[str] = set()
    text_by_path: dict[str, str] = {}
    image_paths: set[str] = set()
    session_id = ""

    if not raw_events_path.exists():
        return {
            "session_id": "",
            "all_paths": [],
            "md_paths": [],
            "image_paths": [],
            "text_by_path": {},
        }

    with raw_events_path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue

            row = _safe_json_loads(line)
            if not isinstance(row, dict):
                continue
            if not session_id:
                session_id = str(row.get("session_id") or "").strip()

            event_raw = row.get("event_raw")
            if not isinstance(event_raw, dict):
                continue

            event_type = str(event_raw.get("type") or "")
            tool_name = str(event_raw.get("toolCallName") or "").strip()
            if event_type not in {"TOOL_CALL_DELTA", "TEXT_DELTA"}:
                all_paths.update(extract_workspace_paths(json.dumps(event_raw, ensure_ascii=False)))

            if event_type in {"TOOL_CALL", "TOOL_CALL_END"} and tool_name in {"write", "edit"}:
                args = _parse_tool_args_to_dict(event_raw.get("args"))
                file_path = str(args.get("filePath") or args.get("path") or "").strip()
                if _looks_like_workspace_file(file_path):
                    content_value = args.get("newText")
                    if content_value is None:
                        content_value = args.get("content")
                    if isinstance(content_value, str):
                        text_by_path[file_path] = content_value
                        all_paths.add(file_path)

            if event_type == "TOOL_RESULT" and tool_name == "image_gen_edit":
                result = event_raw.get("result")
                if isinstance(result, dict):
                    data = result.get("data")
                    if isinstance(data, dict):
                        outputs = data.get("results")
                        if isinstance(outputs, list):
                            for item in outputs:
                                if not isinstance(item, dict):
                                    continue
                                out_path = str(item.get("path") or "").strip()
                                if _looks_like_workspace_file(out_path):
                                    all_paths.add(out_path)
                                    if _is_image_like_file(out_path):
                                        image_paths.add(out_path)

    file_paths = [p for p in sorted(all_paths) if _looks_like_workspace_file(p)]
    md_paths = [p for p in file_paths if _is_markdown_like_file(p)]
    image_paths = sorted(set(image_paths) | {p for p in file_paths if _is_image_like_file(p)})

    return {
        "session_id": session_id,
        "all_paths": file_paths,
        "md_paths": md_paths,
        "image_paths": image_paths,
        "text_by_path": text_by_path,
    }


def _download_workspace_file_bytes_from_files_api(
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    virtual_path: str,
) -> tuple[bytes | None, str]:
    if not base_url or not session_id or not _looks_like_workspace_file(virtual_path):
        return None, "invalid_input"

    req_headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
    base = base_url.rstrip("/")
    normalized = "/" + str(virtual_path).lstrip("/")
    candidates = [
        f"{base}/v1/files/session/{session_id}{normalized}",
        f"{base}/v1/files/session/{session_id}/{normalized.lstrip('/')}",
    ]

    last_err = "not_tried"
    for url in candidates:
        try:
            resp = requests.get(url, headers=req_headers, timeout=20)
        except Exception as exc:
            last_err = f"request_error:{exc.__class__.__name__}"
            continue

        if resp.status_code < 200 or resp.status_code >= 300:
            last_err = f"http_{resp.status_code}"
            continue

        content_type = str(resp.headers.get("Content-Type") or "").lower()
        if "application/json" in content_type:
            payload = _safe_json_loads(resp.text if isinstance(resp.text, str) else "")
            if isinstance(payload, dict):
                data = payload.get("data")
                if isinstance(data, str):
                    return data.encode("utf-8"), ""
                if isinstance(data, dict):
                    text_value = data.get("content")
                    if not isinstance(text_value, str):
                        text_value = data.get("text")
                    if isinstance(text_value, str):
                        return text_value.encode("utf-8"), ""
            last_err = "json_payload_unsupported"
            continue

        return resp.content, ""

    return None, last_err


def _download_workspace_file_bytes_from_dotai_fs(
    dotai_base_url: str,
    headers: dict[str, str],
    user_id: str,
    virtual_path: str,
) -> tuple[bytes | None, str]:
    base = str(dotai_base_url or "").strip().rstrip("/")
    uid = str(user_id or "").strip()
    if not base or not uid or not _looks_like_workspace_file(virtual_path):
        return None, "dotai_invalid_input"

    req_headers = _build_dotai_fs_headers(headers)
    stat_url = f"{base}/dotai/fs/stat"
    stat_body = {
        "userId": uid,
        "traceId": "",
        "bizType": "SANDBOX",
        "spaceId": uid,
        "path": virtual_path,
    }

    try:
        stat_resp = requests.post(stat_url, headers=req_headers, json=stat_body, timeout=20)
    except Exception as exc:
        return None, f"dotai_stat_request_error:{exc.__class__.__name__}"

    if stat_resp.status_code < 200 or stat_resp.status_code >= 300:
        return None, f"dotai_stat_http_{stat_resp.status_code}"

    stat_payload = _safe_json_loads(stat_resp.text if isinstance(stat_resp.text, str) else "")
    if not _dotai_response_code_is_success(stat_payload):
        code = stat_payload.get("code") if isinstance(stat_payload, dict) else "unknown"
        return None, f"dotai_stat_code_{code}"
    stat_result = _dotai_pick_result(stat_payload)
    obj_key = ""
    if isinstance(stat_result, dict):
        obj_key = str(stat_result.get("objKey") or stat_result.get("objectKey") or "").strip()
    if not obj_key:
        return None, "dotai_stat_missing_objKey"

    download_url = f"{base}/dotai/fs/download"
    download_body = {
        "userId": uid,
        "traceId": "",
        "bizType": "SANDBOX",
        "spaceId": uid,
        "reqList": [
            {
                "objKey": obj_key,
                "expireSeconds": 3600,
            }
        ],
    }

    try:
        download_resp = requests.post(download_url, headers=req_headers, json=download_body, timeout=20)
    except Exception as exc:
        return None, f"dotai_download_request_error:{exc.__class__.__name__}"

    if download_resp.status_code < 200 or download_resp.status_code >= 300:
        return None, f"dotai_download_http_{download_resp.status_code}"

    download_payload = _safe_json_loads(download_resp.text if isinstance(download_resp.text, str) else "")
    if not _dotai_response_code_is_success(download_payload):
        code = download_payload.get("code") if isinstance(download_payload, dict) else "unknown"
        return None, f"dotai_download_code_{code}"

    dl_result = _dotai_pick_result(download_payload)
    download_items: list[dict[str, Any]] = []
    if isinstance(dl_result, list):
        download_items = [it for it in dl_result if isinstance(it, dict)]
    elif isinstance(dl_result, dict):
        req_list = dl_result.get("reqList")
        if isinstance(req_list, list):
            download_items = [it for it in req_list if isinstance(it, dict)]
    if not download_items:
        return None, "dotai_download_empty_result"

    signed_url = str(
        download_items[0].get("url")
        or download_items[0].get("downloadUrl")
        or download_items[0].get("signedUrl")
        or ""
    ).strip()
    if not signed_url:
        return None, "dotai_download_missing_url"

    try:
        bin_resp = requests.get(signed_url, timeout=30)
    except Exception as exc:
        return None, f"dotai_binary_request_error:{exc.__class__.__name__}"
    if bin_resp.status_code < 200 or bin_resp.status_code >= 300:
        return None, f"dotai_binary_http_{bin_resp.status_code}"
    return bin_resp.content, ""


def _download_workspace_file_bytes(
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    virtual_path: str,
    dotai_base_url: str = "",
    user_id: str = "",
) -> tuple[bytes | None, str, str]:
    # 1) Try the original files API path first.
    from_files_api, files_api_err = _download_workspace_file_bytes_from_files_api(
        base_url=base_url,
        headers=headers,
        session_id=session_id,
        virtual_path=virtual_path,
    )
    if from_files_api is not None:
        return from_files_api, "", "files_api"

    # 2) Fallback to DotAI FS (stat + download) for workspace binaries/text.
    from_dotai, dotai_err = _download_workspace_file_bytes_from_dotai_fs(
        dotai_base_url=dotai_base_url,
        headers=headers,
        user_id=user_id or str(headers.get("uid") or headers.get("UID") or "").strip(),
        virtual_path=virtual_path,
    )
    if from_dotai is not None:
        return from_dotai, "", "dotai_fs"

    if files_api_err and dotai_err:
        return None, f"{files_api_err};fallback:{dotai_err}", ""
    return None, dotai_err or files_api_err or "download_failed", ""


def export_workspace_view(
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    raw_events_path: Path,
    result_dir: Path,
    dotai_base_url: str = "",
    user_id: str = "",
) -> dict[str, Any]:
    workspace_dir = result_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _collect_workspace_artifacts_from_raw_events(raw_events_path)
    text_by_path = artifacts.get("text_by_path")
    if not isinstance(text_by_path, dict):
        text_by_path = {}
    md_paths = artifacts.get("md_paths")
    if not isinstance(md_paths, list):
        md_paths = []
    image_paths = artifacts.get("image_paths")
    if not isinstance(image_paths, list):
        image_paths = []
    all_paths = artifacts.get("all_paths")
    if not isinstance(all_paths, list):
        all_paths = []

    exported_text: list[dict[str, Any]] = []
    exported_images: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    text_targets = sorted(set(md_paths) | set(text_by_path.keys()))
    for virtual_path in text_targets:
        rel_path = _workspace_virtual_to_rel_path(virtual_path)
        local_path = workspace_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        content = text_by_path.get(virtual_path)
        source = "tool_args"
        if not isinstance(content, str):
            downloaded, err, source_name = _download_workspace_file_bytes(
                base_url=base_url,
                headers=headers,
                session_id=session_id,
                virtual_path=virtual_path,
                dotai_base_url=dotai_base_url,
                user_id=user_id,
            )
            if downloaded is not None:
                content = downloaded.decode("utf-8", errors="replace")
                source = source_name or "download_api"
            else:
                content = (
                    "# Workspace Export Placeholder\n\n"
                    f"- virtual_path: `{virtual_path}`\n"
                    f"- reason: `{err}`\n"
                )
                source = f"placeholder:{err}"
                unresolved.append({"virtual_path": virtual_path, "type": "text", "reason": err})

        local_path.write_text(content, encoding="utf-8")
        exported_text.append(
            {
                "virtual_path": virtual_path,
                "local_path": local_path.relative_to(result_dir).as_posix(),
                "source": source,
            }
        )

    for virtual_path in sorted(set(image_paths)):
        rel_path = _workspace_virtual_to_rel_path(virtual_path)
        local_path = workspace_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        downloaded, err, source_name = _download_workspace_file_bytes(
            base_url=base_url,
            headers=headers,
            session_id=session_id,
            virtual_path=virtual_path,
            dotai_base_url=dotai_base_url,
            user_id=user_id,
        )
        if downloaded:
            local_path.write_bytes(downloaded)
            exported_images.append(
                {
                    "virtual_path": virtual_path,
                    "local_path": local_path.relative_to(result_dir).as_posix(),
                    "source": source_name or "download_api",
                    "bytes": len(downloaded),
                }
            )
            continue

        ref_path = Path(str(local_path) + ".ref.json")
        ref_payload = {
            "virtual_path": virtual_path,
            "reason": err,
            "note": "binary image not downloadable from current files API; path is kept for workspace visibility",
        }
        ref_path.write_text(json.dumps(ref_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        unresolved.append({"virtual_path": virtual_path, "type": "image", "reason": err})

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "raw_events": raw_events_path.as_posix(),
        "download_backends": {
            "files_api_base_url": base_url,
            "dotai_base_url": dotai_base_url,
        },
        "counts": {
            "all_paths": len(all_paths),
            "md_paths": len(md_paths),
            "image_paths": len(image_paths),
            "exported_text_files": len(exported_text),
            "exported_image_files": len(exported_images),
            "unresolved_files": len(unresolved),
        },
        "exported_text_files": exported_text,
        "exported_image_files": exported_images,
        "unresolved_files": unresolved,
        "all_paths": all_paths[:300],
    }
    (workspace_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    (workspace_dir / "README.md").write_text(
        "# Workspace Export\n\n"
        "This directory mirrors files referenced in the session workspace.\n\n"
        "- Markdown/text files: exported from tool-call args first (`write`/`edit`), then download fallback.\n"
        "- Image files: downloaded via files API first, then DotAI FS (`/dotai/fs/stat` + `/dotai/fs/download`).\n"
        "- If both backends fail, `*.ref.json` is written for visibility.\n"
        "- Details: `_manifest.json`.\n",
        encoding="utf-8",
    )

    return manifest
