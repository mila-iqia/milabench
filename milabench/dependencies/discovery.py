"""Discover available (torch, backend) combinations from the PyTorch wheel index."""

from __future__ import annotations

import re
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from packaging.version import Version

# Matches wheel filenames like:
#   torch-2.12.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl
#   torch-2.10.0+rocm7.1-cp312-cp312-manylinux_2_28_x86_64.whl
#   torch-2.12.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl
WHEEL_PATTERN = re.compile(
    r"torch-(\d+\.\d+\.\d+)\+(cu\d+|rocm[\d.]+|cpu)"
    r"-cp(\d+)-cp\d+\w*-(\w+)\.whl"
)

# Map from wheel tag prefix to our backend name
BACKEND_MAP = {
    "cu": "cuda",
    "rocm": "rocm",
    "cpu": "cpu",
}


@dataclass
class WheelInfo:
    """Parsed information from a torch wheel filename."""

    torch_version: str
    backend_type: str  # "cuda", "rocm", "cpu"
    backend_version: str  # "130", "7.1", "" (for cpu)
    python_version: str  # "312", "313", etc.
    platform: str  # "manylinux_2_28_x86_64", etc.


@dataclass
class DiscoveryConfig:
    """Configuration for index discovery from platforms.toml [pin.discovery]."""

    torch_index: str = "https://download.pytorch.org/whl/torch/"
    torch_min: str = "2.10"
    backends: list[str] = field(default_factory=lambda: ["cuda", "rocm", "cpu"])
    python: str | None = None
    platforms: list[str] | None = None  # e.g. ["manylinux_2_28_x86_64", "manylinux_2_28_aarch64"]
    latest_patch_only: bool = True


def _parse_wheel_filename(filename: str) -> WheelInfo | None:
    """Parse a torch wheel filename into structured info."""
    match = WHEEL_PATTERN.search(filename)
    if not match:
        return None

    torch_version = match.group(1)
    backend_tag = match.group(2)
    python_version = match.group(3)
    platform = match.group(4)

    # Determine backend type and version
    if backend_tag == "cpu":
        backend_type = "cpu"
        backend_version = ""
    elif backend_tag.startswith("cu"):
        backend_type = "cuda"
        backend_version = backend_tag[2:]  # "cu130" → "130"
    elif backend_tag.startswith("rocm"):
        backend_type = "rocm"
        backend_version = backend_tag[4:]  # "rocm7.1" → "7.1"
    else:
        return None

    return WheelInfo(
        torch_version=torch_version,
        backend_type=backend_type,
        backend_version=backend_version,
        python_version=python_version,
        platform=platform,
    )


def _get_current_python_tag() -> str:
    """Get the current Python version as a cpXYZ tag number (e.g. '312')."""
    return f"{sys.version_info.major}{sys.version_info.minor}"


def _get_current_platform() -> str:
    """Best-effort detection of the current platform tag for wheel matching."""
    import platform as plat

    machine = plat.machine()  # x86_64, aarch64, etc.
    system = plat.system().lower()

    if system == "linux":
        return f"manylinux_2_28_{machine}"
    elif system == "darwin":
        return f"macosx_{machine}"
    elif system == "windows":
        if machine in ("AMD64", "x86_64"):
            return "win_amd64"
        return f"win_{machine}"
    return f"linux_{machine}"


def fetch_torch_index(index_url: str) -> str:
    """Fetch the torch wheel index page content."""
    req = urllib.request.Request(index_url, headers={"Accept": "text/html"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_available_wheels(index_content: str) -> list[WheelInfo]:
    """Parse all torch wheel filenames from the index page HTML/text content."""
    wheels = []
    for line in index_content.splitlines():
        # The index page lists filenames (possibly as links)
        # Handle both plain text and HTML anchor formats
        info = _parse_wheel_filename(line)
        if info:
            wheels.append(info)
    return wheels


def _filter_latest_patch(
    combos: list[tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    """Keep only the latest patch version per (backend, backend_ver, torch_minor, arch).

    For example, if both 2.12.0 and 2.12.1 exist for cuda/130/x86_64,
    only keep 2.12.1.
    """
    # Group by (backend, backend_ver, torch_major.minor, arch)
    groups: dict[tuple[str, str, str, str], list[tuple[str, str, str, str]]] = {}
    for backend, backend_ver, torch_ver, arch in combos:
        v = Version(torch_ver)
        minor_key = f"{v.major}.{v.minor}"
        key = (backend, backend_ver, minor_key, arch)
        groups.setdefault(key, []).append((backend, backend_ver, torch_ver, arch))

    result = []
    for key, group in groups.items():
        group.sort(key=lambda x: Version(x[2]), reverse=True)
        result.append(group[0])

    return sorted(result, key=lambda x: (x[0], x[1], Version(x[2]), x[3]))


def _platform_to_arch(platform_tag: str) -> str:
    """Extract the architecture shorthand from a platform tag.

    Examples:
        "manylinux_2_28_x86_64" → "x86_64"
        "manylinux_2_28_aarch64" → "aarch64"
        "win_amd64" → "amd64"
    """
    parts = platform_tag.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("x86_64", "aarch64", "amd64", "arm64", "s390x"):
        return parts[1]
    # For platforms like "manylinux_2_28_x86_64", split on last known arch
    for arch in ("x86_64", "aarch64", "amd64", "arm64", "s390x"):
        if platform_tag.endswith(arch):
            return arch
    return platform_tag


def discover_combinations(
    config: DiscoveryConfig,
) -> list[tuple[str, str, str, str]]:
    """Fetch the torch index and return available (backend, backend_ver, torch_ver, arch) tuples.

    Args:
        config: Discovery configuration from platforms.toml.

    Returns:
        Sorted list of (backend_type, backend_version, torch_version, arch) tuples.
        For cpu backend, backend_version is "".
        arch is the CPU architecture (e.g. "x86_64", "aarch64").
    """
    index_content = fetch_torch_index(config.torch_index)
    wheels = parse_available_wheels(index_content)

    # Apply filters
    python_tag = config.python or _get_current_python_tag()
    platform_tags = config.platforms or [_get_current_platform()]
    torch_min = Version(config.torch_min)

    seen = set()
    combos = []

    for wheel in wheels:
        # Filter by backend
        if wheel.backend_type not in config.backends:
            continue

        # Filter by python version
        if wheel.python_version != python_tag:
            continue

        # Filter by platform (match any in the list)
        if wheel.platform not in platform_tags:
            continue

        # Filter by torch minimum version
        try:
            torch_ver = Version(wheel.torch_version)
        except Exception:
            continue
        if torch_ver < torch_min:
            continue

        arch = _platform_to_arch(wheel.platform)
        key = (wheel.backend_type, wheel.backend_version, wheel.torch_version, arch)
        if key not in seen:
            seen.add(key)
            combos.append(key)

    # Apply latest_patch_only filter
    if config.latest_patch_only:
        combos = _filter_latest_patch(combos)

    # Sort: by backend, then backend version, then torch version, then arch
    combos.sort(key=lambda x: (x[0], x[1], Version(x[2]), x[3]))
    return combos


def print_discovered_combinations(combos: list[tuple[str, str, str, str]]) -> None:
    """Pretty-print discovered combinations (for --dry-run output)."""
    if not combos:
        print("No combinations discovered.")
        return

    print(f"Discovered {len(combos)} combinations:\n")

    current_backend = None
    for backend, backend_ver, torch_ver, arch in combos:
        label = f"{backend}{backend_ver}" if backend_ver else backend
        if label != current_backend:
            current_backend = label
            print(f"  {label}:")
        print(f"    torch {torch_ver} ({arch})")
