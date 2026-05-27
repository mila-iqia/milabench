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
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in run_folders:
            folder = str(folder)
            for root, _dirs, files in os.walk(folder):
                for fname in files:
                    filepath = os.path.join(root, fname)
                    arcname = os.path.relpath(filepath, os.path.dirname(folder))
                    zf.write(filepath, arcname)
    buf.seek(0)
    return buf


def publish_results(run_folders, push_key, dashboard_url=None, metadata=None):
    """Zip run folders and push them to the dashboard.

    Args:
        run_folders: iterable of paths to run result directories
        push_key: secret push key for authentication
        dashboard_url: base URL of the dashboard (defaults to MILABENCH_DASHBOARD_URL)
        metadata: optional dict of extra metadata to attach
    """
    url = (dashboard_url or MILABENCH_DASHBOARD_URL).rstrip("/")
    endpoint = f"{url}/api/push/zip"

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

    print(f"[publish] Uploading to {endpoint} ...")
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
