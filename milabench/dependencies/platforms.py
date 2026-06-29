"""Load and validate platforms.toml, resolve variables."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


_BACKEND_PREFIXES = {
    "cu": "cuda",
    "cuda": "cuda",
    "rocm": "rocm",
    "cpu": "cpu",
    "hpu": "hpu",
    "xpu": "xpu",
}


def _normalize_overrides(overrides: dict[str, str]) -> dict[str, str]:
    """Normalize CLI overrides.

    Handles:
      - backend=cu130 → cuda=130
      - backend=rocm7.2 → rocm=7.2
      - backend=cpu → (no version needed)
      - torch=2.10 → torch=2.10.0 (ensure 3-part version)
    """
    result = dict(overrides)

    # Parse "backend" shorthand into backend_type + version
    if "backend" in result:
        raw = result.pop("backend")
        match = re.match(r"([a-zA-Z]+)(.*)", raw)
        if match:
            prefix, version = match.group(1).lower(), match.group(2)
            backend_name = _BACKEND_PREFIXES.get(prefix, prefix)
            if version:
                result.setdefault(backend_name, version)
        else:
            result.setdefault("cpu", "")

    # Normalize torch version to 3 parts (2.10 → 2.10.0)
    if "torch" in result:
        parts = result["torch"].split(".")
        if len(parts) == 2:
            result["torch"] = f"{result['torch']}.0"

    return result


@dataclass
class PlatformConfig:
    """Parsed platforms.toml with resolved variable interpolation."""

    vars: dict[str, str] = field(default_factory=dict)
    pin_matrix: PinMatrix | None = None
    discovery: DiscoveryConfig | None = None
    backends: dict[str, BackendConfig] = field(default_factory=dict)
    compat: dict[str, CompatEntry] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def resolve_vars(self, overrides: dict[str, str] | None = None) -> dict[str, str]:
        """Return vars dict with CLI overrides applied.

        Normalizes overrides first:
          - backend=cu130 → cuda=130
          - backend=rocm7.2 → rocm=7.2
          - torch=2.10 → torch=2.10.0

        Also injects derived variables:
          - cuda_major: first 2 chars of cuda version (e.g. "130" → "13", "126" → "12")
        """
        resolved = dict(self.vars)
        if overrides:
            resolved.update(_normalize_overrides(overrides))

        # Derived variables
        if "cuda" in resolved and "cuda_major" not in resolved:
            resolved["cuda_major"] = resolved["cuda"][:2]

        return resolved

    def resolve_string(self, template: str, overrides: dict[str, str] | None = None) -> str:
        """Interpolate {var} placeholders in a string."""
        variables = self.resolve_vars(overrides)
        return template.format(**variables)

    def get_backend(self, name: str) -> BackendConfig:
        if name not in self.backends:
            # Return a default backend config (pypi only, no constraints)
            # This allows backends discovered from the index that aren't
            # explicitly configured in platforms.toml to still work.
            return BackendConfig(name=name)
        return self.backends[name]


@dataclass
class BackendConfig:
    """Configuration for a single backend (cuda, rocm, etc.)."""

    name: str
    indexes: IndexConfig = field(default_factory=lambda: IndexConfig())
    constraints: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, OverrideConfig] = field(default_factory=dict)


@dataclass
class IndexConfig:
    """Index URL configuration for a backend."""

    index_url: str = "https://pypi.org/simple"
    extra_index_url: list[str] = field(default_factory=list)
    find_links: list[str] = field(default_factory=list)


@dataclass
class OverrideConfig:
    """Override configuration for a package that needs special install handling."""

    package: str
    install: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class CompatRule:
    """A single condition -> constraint mapping from [compat.*]."""

    conditions: str  # e.g. "torch>=2.11,cuda>=13"
    constraint: str  # e.g. "<0.18"


@dataclass
class CompatEntry:
    """Compat rules for one package."""

    package: str
    rules: list[CompatRule] = field(default_factory=list)


@dataclass
class DiscoveryConfig:
    """Configuration for index-based discovery from platforms.toml [pin.discovery]."""

    torch_index: str = "https://download.pytorch.org/whl/torch/"
    torch_min: str = "2.10"
    backends: list[str] = field(default_factory=lambda: ["cuda", "rocm", "cpu"])
    python: str | None = None
    platforms: list[str] | None = None  # e.g. ["manylinux_2_28_x86_64", "manylinux_2_28_aarch64"]
    latest_patch_only: bool = True


@dataclass
class PinMatrix:
    """Defines which (backend, torch) combinations to pin (legacy static matrix)."""

    torch: list[str] = field(default_factory=list)
    backends: dict[str, list[str]] = field(default_factory=dict)
    exclude: list[dict[str, str]] = field(default_factory=list)

    def combinations(self) -> list[tuple[str, str, str]]:
        """Yield valid (backend_name, backend_version, torch_version) tuples."""
        combos = []
        for torch_ver in self.torch:
            for backend_name, backend_versions in self.backends.items():
                for backend_ver in backend_versions:
                    if not self._is_excluded(backend_name, backend_ver, torch_ver):
                        combos.append((backend_name, backend_ver, torch_ver))
        return combos

    def _is_excluded(self, backend_name: str, backend_ver: str, torch_ver: str) -> bool:
        for exc in self.exclude:
            match = True
            if backend_name in exc and exc[backend_name] != backend_ver:
                match = False
            if backend_name not in exc:
                match = False
            if "torch" in exc and exc["torch"] != torch_ver:
                match = False
            if match:
                return True
        return False


def _find_platforms_toml(start_path: Path | None = None) -> Path:
    """Locate platforms.toml, searching from milabench repo root."""
    if start_path is None:
        start_path = Path(__file__).parent.parent.parent

    candidates = [
        start_path / "platforms.toml",
        Path(os.environ.get("MILABENCH_CONFIG_DIR", "")) / "platforms.toml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"platforms.toml not found. Searched: {[str(c) for c in candidates]}"
    )


def load_platform_config(
    path: Path | str | None = None,
    overrides: dict[str, str] | None = None,
) -> PlatformConfig:
    """Load and parse platforms.toml.

    Args:
        path: Explicit path to platforms.toml. If None, auto-detected.
        overrides: CLI variable overrides (e.g. {"cuda": "130", "torch": "2.12.0"}).

    Returns:
        PlatformConfig with all sections parsed.
    """
    if path is None:
        path = _find_platforms_toml()
    else:
        path = Path(path)

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    config = PlatformConfig(raw=raw)

    # Parse [vars]
    config.vars = {k: str(v) for k, v in raw.get("vars", {}).items()}

    # Parse [pin.discovery] (preferred) or [pin.matrix] (legacy)
    pin_raw = raw.get("pin", {})
    discovery_raw = pin_raw.get("discovery", {})
    matrix_raw = pin_raw.get("matrix", {})

    if discovery_raw:
        # platforms can be a list or a single string
        platforms_val = discovery_raw.get("platforms")
        if isinstance(platforms_val, str):
            platforms_val = [platforms_val]

        config.discovery = DiscoveryConfig(
            torch_index=discovery_raw.get("torch_index", "https://download.pytorch.org/whl/torch/"),
            torch_min=str(discovery_raw.get("torch_min", "2.10")),
            backends=discovery_raw.get("backends", ["cuda", "rocm", "cpu"]),
            python=discovery_raw.get("python"),
            platforms=platforms_val,
            latest_patch_only=discovery_raw.get("latest_patch_only", True),
        )
    elif matrix_raw:
        config.pin_matrix = PinMatrix(
            torch=matrix_raw.get("torch", []),
            backends=matrix_raw.get("backends", {}),
            exclude=matrix_raw.get("exclude", []),
        )

    # Parse backend sections (cuda, rocm, hpu, xpu, cpu)
    backend_names = {"cuda", "rocm", "hpu", "xpu", "cpu"}
    for name in backend_names:
        if name not in raw:
            continue

        section = raw[name]
        backend = BackendConfig(name=name)

        # Indexes
        idx = section.get("indexes", {})
        if idx:
            backend.indexes = IndexConfig(
                index_url=idx.get("index-url", "https://pypi.org/simple"),
                extra_index_url=idx.get("extra-index-url", []),
                find_links=idx.get("find-links", []),
            )

        # Constraints
        backend.constraints = section.get("constraints", {})

        # Overrides
        overrides_raw = section.get("overrides", {})
        for pkg_name, override_data in overrides_raw.items():
            backend.overrides[pkg_name] = OverrideConfig(
                package=pkg_name,
                install=override_data.get("install"),
                env=override_data.get("env", {}),
            )

        config.backends[name] = backend

    # Parse [compat.*] sections
    compat_raw = raw.get("compat", {})
    for pkg_name, rules_dict in compat_raw.items():
        if not isinstance(rules_dict, dict):
            continue
        entry = CompatEntry(package=pkg_name)
        for conditions_key, constraint_value in rules_dict.items():
            entry.rules.append(CompatRule(
                conditions=conditions_key,
                constraint=str(constraint_value),
            ))
        config.compat[pkg_name] = entry

    return config
