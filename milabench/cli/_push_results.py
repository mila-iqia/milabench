"""Utility to publish milabench results to a remote dashboard via push key."""

import io
import os
import zipfile

import requests

MILABENCH_DASHBOARD_URL = os.getenv(
    "MILABENCH_DASHBOARD_URL",
    "https://www.milabench.com",
)

MAX_ZIP_SIZE = 500 * 1024 * 1024       # 500 MB
MAX_ENTRY_SIZE = 100 * 1024 * 1024     # 100 MB per entry
MAX_ENTRIES = 50_000


class PublishError(Exception):
    pass


def zip_run_folders(run_folders):
    """Create an in-memory zip archive from a set of run folders (.data files only).

    Validates against server limits before returning.
    """
    buf = io.BytesIO()
    entry_count = 0

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for folder in run_folders:
            folder = str(folder)
            for root, _dirs, files in os.walk(folder):
                for fname in files:
                    if not fname.endswith(".data"):
                        continue

                    filepath = os.path.join(root, fname)
                    file_size = os.path.getsize(filepath)

                    if file_size > MAX_ENTRY_SIZE:
                        raise PublishError(
                            f"File too large: {filepath} "
                            f"({file_size / 1024 / 1024:.1f} MB > {MAX_ENTRY_SIZE / 1024 / 1024:.0f} MB limit)"
                        )

                    entry_count += 1
                    if entry_count > MAX_ENTRIES:
                        raise PublishError(
                            f"Too many files: exceeded {MAX_ENTRIES} entry limit"
                        )

                    arcname = os.path.relpath(filepath, os.path.dirname(folder))
                    zf.write(filepath, arcname)

    total_size = buf.tell()
    if total_size > MAX_ZIP_SIZE:
        raise PublishError(
            f"Archive too large: {total_size / 1024 / 1024:.1f} MB > {MAX_ZIP_SIZE / 1024 / 1024:.0f} MB limit"
        )

    print(f"[publish] Archive: {entry_count} files, {total_size / 1024 / 1024:.1f} MB")
    buf.seek(0)
    return buf


def _publish_stream(endpoint, data, files):
    """Upload via the SSE streaming endpoint and print events in real time."""
    import json

    success = False
    with requests.post(endpoint, data=data, files=files, stream=True, timeout=3600) as resp:
        if resp.headers.get("content-type", "").startswith("text/event-stream"):
            event_type = None
            for line in resp.iter_lines(decode_unicode=True):
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    payload = json.loads(line[6:])
                    _print_sse_event(event_type, payload)
                    if event_type == "done":
                        success = payload.get("status") == "OK"
                    elif event_type == "error":
                        success = False
        else:
            content_type = resp.headers.get("content-type", "")
            print(f"[publish] Unexpected response (HTTP {resp.status_code}, {content_type})")
            try:
                result = resp.json()
                if result.get("status") == "OK":
                    print(f"[publish] Success: {result.get('message')}")
                    success = True
                else:
                    print(f"[publish] Failed: {result.get('message', resp.text)}")
            except (json.JSONDecodeError, ValueError):
                print(f"[publish] Server returned non-JSON response:")
                print(resp.text or "(empty body)")

    return success


def _print_sse_event(event, data):
    match event:
        case "info":
            print(f"[publish] {data['message']} (contributor: {data.get('contributor', '?')})")
        case "run":
            print(f"[publish] Run: {data['name']} ({data['benchmarks']} benchmarks)")
        case "bench":
            print(f"[publish]   Processing {data['name']}...")
        case "bench_done":
            print(f"[publish]   ✓ {data['name']} ({data['events']} events)")
        case "bench_error":
            print(f"[publish]   ✗ {data['name']}: {data['error']}")
        case "done":
            print(f"[publish] {data['message']}")
        case "error":
            print(f"[publish] ERROR: {data['message']}")
            if data.get("traceback"):
                print(data["traceback"])


def publish_results(run_folders, push_key, dashboard_url=None, metadata=None):
    """Zip run folders and push them to the dashboard.

    Args:
        run_folders: iterable of paths to run result directories
        push_key: secret push key for authentication
        dashboard_url: base URL of the dashboard (defaults to MILABENCH_DASHBOARD_URL)
        metadata: optional dict of extra metadata to attach
    """
    url = (dashboard_url or MILABENCH_DASHBOARD_URL).rstrip("/")

    folders = [f for f in run_folders if os.path.isdir(str(f))]
    if not folders:
        print("[publish] No run folders found, skipping publish.")
        return False

    print(f"[publish] Zipping {len(folders)} run folder(s)...")
    buf = zip_run_folders(folders)

    data = {"key": push_key}
    if metadata:
        import json
        data["metadata"] = json.dumps(metadata)

    files = {"file": ("results.zip", buf, "application/zip")}

    endpoint = f"{url}/api/push/zip/stream"
    print(f"[publish] Uploading to {endpoint} ...")
    print(f"[publish] Key: {repr(push_key[:8])}...{repr(push_key[-4:])} (len={len(push_key)})")
    try:
        return _publish_stream(endpoint, data, files)
    except Exception as err:
        import traceback
        print(f"[publish] Push failed: {type(err).__name__}: {err}")
        traceback.print_exc()
        return False
