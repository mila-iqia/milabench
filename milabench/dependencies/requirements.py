"""Load and validate per-benchmark requirements.toml files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from .platforms import PlatformConfig


SUPPORTED_BACKENDS = ("cuda", "rocm", "hpu", "xpu")


@dataclass
class BenchmarkRequirements:
    """Parsed requirements.toml for a single benchmark."""

    path: Path
    common: list[str] = field(default_factory=list)
    backends: dict[str, BackendSection] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def is_enabled(self, backend: str) -> bool:
        """Check if this benchmark supports the given backend."""
        section = self.backends.get(backend)
        if section is None:
            # No section = not explicitly declared, treat as enabled
            # (inherits common deps only)
            return True
        return section.enabled

    def get_dependencies(self, backend: str) -> list[str]:
        """Get combined [common] + [backend] dependencies."""
        if not self.is_enabled(backend):
            return []

        deps = list(self.common)
        section = self.backends.get(backend)
        if section and section.dependencies:
            deps.extend(section.dependencies)
        return deps


@dataclass
class BackendSection:
    """A backend section from requirements.toml."""

    name: str
    enabled: bool = True
    dependencies: list[str] = field(default_factory=list)
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


def has_toml_requirements(benchmark_path: Path | str) -> bool:
    """Check if a benchmark directory has a requirements.toml file."""
    path = Path(benchmark_path)
    if path.is_file():
        return path.name == "requirements.toml"
    return (path / "requirements.toml").exists()


def load_benchmark_requirements(path: Path | str) -> BenchmarkRequirements:
    """Load a benchmark's requirements.toml.

    Args:
        path: Path to requirements.toml file or the benchmark directory.

    Returns:
        BenchmarkRequirements with all sections parsed.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "requirements.toml"

    if not path.exists():
        raise FileNotFoundError(f"Requirements file not found: {path}")

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    reqs = BenchmarkRequirements(path=path, raw=raw)

    # Parse [common]
    common_section = raw.get("common", {})
    reqs.common = common_section.get("dependencies", [])

    # Parse backend sections
    for backend_name in SUPPORTED_BACKENDS:
        if backend_name not in raw:
            continue

        section_data = raw[backend_name]
        section = BackendSection(name=backend_name)

        if isinstance(section_data, dict):
            section.enabled = section_data.get("enabled", True)
            section.dependencies = section_data.get("dependencies", [])
            section.overrides = section_data.get("overrides", {})
        else:
            section.enabled = False

        reqs.backends[backend_name] = section

    return reqs


def resolve_benchmark(
    benchmark_path: Path | str,
    backend: str,
    platform_config: PlatformConfig,
    overrides: dict[str, str] | None = None,
) -> list[str]:
    """Resolve a benchmark's requirements to a flat list of package specs.

    This is the main entry point for generating install requirements from TOML.

    Args:
        benchmark_path: Path to benchmark directory or requirements.toml.
        backend: The backend to resolve for (cuda, rocm, etc.).
        platform_config: Loaded platform configuration.
        overrides: CLI variable overrides.

    Returns:
        List of resolved package specifiers (e.g. ["torch>=2.10", "vllm==v0.18.1+cu130"]).

    Raises:
        ValueError: If the benchmark doesn't support the given backend.
    """
    reqs = load_benchmark_requirements(benchmark_path)

    if not reqs.is_enabled(backend):
        raise ValueError(
            f"Benchmark at {benchmark_path} does not support backend '{backend}' "
            f"(enabled=false in requirements.toml)"
        )

    raw_deps = reqs.get_dependencies(backend)
    variables = platform_config.resolve_vars(overrides)

    resolved = []
    for dep in raw_deps:
        try:
            resolved.append(dep.format(**variables))
        except KeyError as e:
            raise ValueError(
                f"Variable {e} used in '{dep}' (from {reqs.path}) "
                f"is not defined in platforms.toml [vars] or CLI overrides. "
                f"Available: {list(variables.keys())}"
            ) from e

    return resolved
