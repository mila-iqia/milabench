"""Process platform and benchmark overrides (replaces bb_after_install.sh)."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .platforms import PlatformConfig, OverrideConfig
from .requirements import load_benchmark_requirements


@dataclass
class ResolvedOverride:
    """A fully resolved override ready for execution."""

    package: str
    install_command: str
    env: dict[str, str] = field(default_factory=dict)
    source: str = ""  # "platform" or "benchmark"

    def execute(self, pip_cmd: str = "pip", cwd: Path | None = None) -> subprocess.CompletedProcess:
        """Execute the override install command."""
        env = dict(os.environ)
        env.update(self.env)

        return subprocess.run(
            self.install_command,
            shell=True,
            env=env,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
        )


def get_overrides(
    benchmark_path: Path | str,
    backend: str,
    platform_config: PlatformConfig,
    overrides: dict[str, str] | None = None,
) -> list[ResolvedOverride]:
    """Collect all applicable overrides for a benchmark on a given backend.

    Processing order: platform overrides first, then benchmark overrides.
    Benchmark overrides for the same package supersede platform overrides.

    Args:
        benchmark_path: Path to the benchmark directory.
        backend: Backend name (cuda, rocm, etc.).
        platform_config: Loaded platform configuration.
        overrides: CLI variable overrides.

    Returns:
        Ordered list of resolved overrides to execute after main install.
    """
    variables = platform_config.resolve_vars(overrides)
    result: dict[str, ResolvedOverride] = {}

    # Platform-level overrides
    backend_config = platform_config.backends.get(backend)
    if backend_config:
        for pkg_name, override_cfg in backend_config.overrides.items():
            resolved = _resolve_override(override_cfg, variables, source="platform")
            if resolved:
                result[pkg_name] = resolved

    # Benchmark-level overrides (supersede platform for same package)
    benchmark_path = Path(benchmark_path)
    if (benchmark_path / "requirements.toml").exists():
        reqs = load_benchmark_requirements(benchmark_path)
        section = reqs.backends.get(backend)
        if section and section.overrides:
            for pkg_name, override_data in section.overrides.items():
                override_cfg = OverrideConfig(
                    package=pkg_name,
                    install=override_data.get("install"),
                    env=override_data.get("env", {}),
                )
                resolved = _resolve_override(override_cfg, variables, source="benchmark")
                if resolved:
                    result[pkg_name] = resolved

    return list(result.values())


def _resolve_override(
    cfg: OverrideConfig,
    variables: dict[str, str],
    source: str,
) -> ResolvedOverride | None:
    """Resolve variable placeholders in an override config."""
    if not cfg.install:
        return None

    try:
        install_cmd = cfg.install.format(**variables)
        env = {k: v.format(**variables) for k, v in cfg.env.items()}
    except KeyError as e:
        raise ValueError(
            f"Variable {e} used in override for '{cfg.package}' "
            f"is not defined in platforms.toml [vars] or CLI overrides."
        ) from e

    return ResolvedOverride(
        package=cfg.package,
        install_command=install_cmd,
        env=env,
        source=source,
    )


async def apply_overrides(
    benchmark_path: Path | str,
    backend: str,
    platform_config: PlatformConfig,
    overrides: dict[str, str] | None = None,
    cwd: Path | None = None,
    dry_run: bool = False,
) -> list[tuple[ResolvedOverride, subprocess.CompletedProcess | None]]:
    """Apply all overrides for a benchmark.

    Args:
        benchmark_path: Path to the benchmark directory.
        backend: Backend name.
        platform_config: Loaded platform configuration.
        overrides: CLI variable overrides.
        cwd: Working directory for override commands.
        dry_run: If True, don't execute, just return the resolved overrides.

    Returns:
        List of (override, result) tuples. Result is None in dry_run mode.
    """
    resolved = get_overrides(benchmark_path, backend, platform_config, overrides)
    results = []

    for override in resolved:
        if dry_run:
            results.append((override, None))
        else:
            result = override.execute(cwd=cwd)
            if result.returncode != 0:
                print(
                    f"WARNING: Override for '{override.package}' failed "
                    f"(source={override.source}):\n{result.stderr}"
                )
            results.append((override, result))

    return results
