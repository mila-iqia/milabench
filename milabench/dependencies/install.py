"""Install-time requirement generation and pip argument building."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .platforms import PlatformConfig
from .pin import get_constraint_file
from .requirements import load_benchmark_requirements, resolve_benchmark


@dataclass
class InstallArgs:
    """Complete set of arguments needed to install a benchmark's deps."""

    requirements_file: Path
    constraint_file: Path | None
    index_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    _temp_files: list[Path] = field(default_factory=list, repr=False)

    def as_pip_args(self) -> list[str]:
        """Build the full pip install argument list."""
        args = ["-r", str(self.requirements_file)]
        if self.constraint_file and self.constraint_file.exists():
            args.extend(["-c", str(self.constraint_file)])
        args.extend(self.index_args)
        return args

    def cleanup(self):
        """Remove temporary files."""
        for f in self._temp_files:
            f.unlink(missing_ok=True)


def get_index_args(
    platform_config: PlatformConfig,
    backend: str,
    overrides: dict[str, str] | None = None,
) -> list[str]:
    """Build index URL arguments for pip/uv install.

    Args:
        platform_config: Loaded platform configuration.
        backend: Backend name (cuda, rocm, etc.).
        overrides: Variable overrides from CLI.

    Returns:
        List of --index-url, --extra-index-url, --find-links arguments.
    """
    backend_config = platform_config.get_backend(backend)
    variables = platform_config.resolve_vars(overrides)
    args = []

    idx = backend_config.indexes
    if idx.index_url:
        args.extend(["--index-url", idx.index_url.format(**variables)])

    for url in idx.extra_index_url:
        args.extend(["--extra-index-url", url.format(**variables)])

    for url in idx.find_links:
        args.extend(["--find-links", url.format(**variables)])

    return args


def install_args(
    benchmark_path: Path | str,
    platform_config: PlatformConfig,
    backend: str,
    pin_dir: Path,
    overrides: dict[str, str] | None = None,
    unpinned: bool = False,
) -> InstallArgs:
    """Generate the complete install arguments for a benchmark.

    This is the main entry point called from pack.py install().

    Args:
        benchmark_path: Path to the benchmark directory.
        platform_config: Loaded platform configuration.
        backend: Backend name (cuda, rocm, etc.).
        pin_dir: Path to .pin/ directory.
        overrides: CLI variable overrides (e.g. {"cuda": "130", "torch": "2.12.0"}).
        unpinned: If True, skip constraint file (NGC/dev mode).

    Returns:
        InstallArgs with paths and arguments ready for pip install.
    """
    benchmark_path = Path(benchmark_path)
    all_overrides = _merge_backend_override(backend, platform_config, overrides)

    # Resolve dependencies to flat list
    resolved_deps = resolve_benchmark(
        benchmark_path, backend, platform_config, all_overrides
    )

    # Write temp requirements file
    temp_req = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix=f"milabench-{benchmark_path.name}-",
        delete=False,
    )
    for dep in resolved_deps:
        temp_req.write(f"{dep}\n")
    temp_req.close()
    temp_req_path = Path(temp_req.name)

    # Find matching constraint file
    constraint_file = None
    if not unpinned:
        import platform as plat

        variables = platform_config.resolve_vars(all_overrides)
        backend_version = variables.get(backend, "")
        torch_version = variables.get("torch", "")
        arch = plat.machine()  # x86_64, aarch64, etc.

        if backend_version and torch_version:
            # Try arch-less constraint file first (current default)
            constraint_file = get_constraint_file(
                pin_dir, backend, backend_version, torch_version
            )
            if not constraint_file.exists():
                # Fall back to arch-specific file (legacy)
                constraint_file = get_constraint_file(
                    pin_dir, backend, backend_version, torch_version, arch
                )
                if not constraint_file.exists():
                    constraint_file = None

    # Build index args
    index_args = get_index_args(platform_config, backend, all_overrides)

    return InstallArgs(
        requirements_file=temp_req_path,
        constraint_file=constraint_file,
        index_args=index_args,
        _temp_files=[temp_req_path],
    )


def _merge_backend_override(
    backend: str,
    platform_config: PlatformConfig,
    overrides: dict[str, str] | None,
) -> dict[str, str]:
    """Merge the backend name into overrides so it's available as a variable."""
    merged = dict(overrides) if overrides else {}
    # The backend version comes from either the override or the default vars
    # e.g., if backend="cuda" and vars has cuda="130", that's the default
    return merged
