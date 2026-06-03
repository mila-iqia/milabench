"""Utility to publish milabench results to a remote dashboard via push key."""

import io
import os
import zipfile

import requests

MILABENCH_DASHBOARD_URL = os.getenv(
    "MILABENCH_DASHBOARD_URL",
    "https://www.milabench.com",
)


def zip_run_folders(run_folders):
    """Create an in-memory zip archive from a set of run folders."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for folder in run_folders:
            folder = str(folder)
            for root, _dirs, files in os.walk(folder):
                for fname in files:
                    filepath = os.path.join(root, fname)
                    arcname = os.path.relpath(filepath, os.path.dirname(folder))
                    zf.write(filepath, arcname)
    buf.seek(0)
    return buf


def _publish_stream(endpoint, data, files):
    """Upload via the SSE streaming endpoint and print events in real time."""
    import json

    success = False
    with requests.post(endpoint, data=data, files=files, stream=True, timeout=300) as resp:
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
            result = resp.json()
            if result.get("status") == "OK":
                print(f"[publish] Success: {result.get('message')}")
                success = True
            else:
                print(f"[publish] Failed: {result.get('message', resp.text)}")

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

    # Try the streaming endpoint first, fall back to the regular one
    endpoint = f"{url}/api/push/zip/stream"
    print(f"[publish] Uploading to {endpoint} ...")
    try:
        return _publish_stream(endpoint, data, files)
    except Exception:
        print("[publish] Streaming endpoint unavailable, falling back to standard push...")
        buf.seek(0)
        files = {"file": ("results.zip", buf, "application/zip")}
        endpoint = f"{url}/api/push/zip"
        try:
            resp = requests.post(endpoint, data=data, files=files, timeout=120)
            result = resp.json()
            if result.get("status") == "OK":
                print(f"[publish] Success: {result.get('message', 'uploaded')}")
                return True
            else:
                print(f"[publish] Failed: {result.get('message', resp.text)}")
                return False
        except Exception as err:
            print(f"[publish] Error: {err}")
            return False
