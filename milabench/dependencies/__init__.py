"""TOML-based dependency management for milabench.

This module provides the new dependency resolution system that replaces
the requirements.in + per-benchmark pip-compile output with a TOML-based
declaration and shared constraint files pinned per (backend, torch) combo.

Public API:
    load_platform_config()  - parse platforms.toml, resolve variables
    resolve_benchmark()     - TOML → flat requirements list for one benchmark
    get_constraint_file()   - (backend, torch) → path to .pin/ file
    get_index_args()        - resolved --index-url, --extra-index-url, --find-links
    pin_combination()       - pin one (backend, torch) combo via uv pip compile
    pin_all()               - iterate the full matrix
    get_overrides()         - collect applicable overrides for a benchmark
    has_toml_requirements() - check if a benchmark has requirements.toml
"""

from .platforms import load_platform_config, PlatformConfig
from .requirements import (
    load_benchmark_requirements,
    resolve_benchmark,
    has_toml_requirements,
    BenchmarkRequirements,
)
from .pin import pin_combination, pin_all, get_constraint_file, constraint_filename
from .install import install_args, get_index_args
from .overrides import get_overrides, apply_overrides
from .discovery import discover_combinations, DiscoveryConfig, print_discovered_combinations

__all__ = [
    "load_platform_config",
    "PlatformConfig",
    "load_benchmark_requirements",
    "resolve_benchmark",
    "has_toml_requirements",
    "BenchmarkRequirements",
    "pin_combination",
    "pin_all",
    "get_constraint_file",
    "constraint_filename",
    "install_args",
    "get_index_args",
    "get_overrides",
    "apply_overrides",
    "discover_combinations",
    "DiscoveryConfig",
    "print_discovered_combinations",
]
