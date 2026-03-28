"""CI helpers for milabench.

Usage as a milabench subcommand:
    milabench ci --config config/standard.yaml
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from ..config import build_config


def get_benchmark_groups(config_path, exclude_tags=None):
    """Group non-private, enabled benchmarks by their definition path.

    Returns a dict mapping definition path to a sorted list of benchmark names.
    """
    config = build_config(config_path)

    if exclude_tags is None:
        exclude_tags = set()

    groups = defaultdict(list)

    for name, defn in config.items():
        definition = defn.get("definition")
        if not definition:
            continue

        if not defn.get("enabled", True):
            continue

        tags = set(defn.get("tags", []))
        if tags & exclude_tags:
            continue

        groups[definition].append(name)

    return {k: sorted(v) for k, v in groups.items()}


def format_groups_for_ci(groups):
    """Return a sorted list of {name, select} objects for the CI matrix.

    ``name`` is the definition folder basename (e.g. "torchvision").
    ``select`` is the comma-separated list of benchmark names for --select.
    """
    result = []
    for definition, names in groups.items():
        folder_name = Path(definition).name
        result.append({
            "name": folder_name,
            "select": ",".join(sorted(names)),
        })
    return sorted(result, key=lambda g: g["name"])


# -- milabench CLI integration ------------------------------------------------

def cli_ci():
    """Output benchmark groups as JSON for CI matrix generation.

    Usage:
        milabench ci --config config/standard.yaml
        milabench ci --config config/standard.yaml --exclude-tags multinode,gated
    """
    from coleo import Option, tooled

    @tooled
    def _inner():
        # Path to the benchmark config file
        config: Option = os.environ.get("MILABENCH_CONFIG", None)

        # Comma-separated tags to exclude from grouping
        exclude_tags: Option & str = ""

        if config is None:
            print("Error: --config is required", file=sys.stderr)
            return 1

        exclude = set(exclude_tags.split(",")) if exclude_tags else set()
        groups = get_benchmark_groups(config, exclude_tags=exclude)
        result = format_groups_for_ci(groups)
        print(json.dumps(result))
        return 0

    return _inner()


# -- Standalone entrypoint ----------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CI helpers for milabench")
    parser.add_argument("--config", required=True, help="Path to benchmark config YAML")
    parser.add_argument(
        "--exclude-tags",
        default="",
        help="Comma-separated tags to exclude (e.g. multinode,gated)",
    )

    args = parser.parse_args()
    exclude = set(args.exclude_tags.split(",")) if args.exclude_tags else set()
    groups = get_benchmark_groups(args.config, exclude_tags=exclude)
    result = format_groups_for_ci(groups)
    print(json.dumps(result))
