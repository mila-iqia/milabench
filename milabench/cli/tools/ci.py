"""CI helpers for milabench.

Usage as a milabench subcommand:
    milabench ci --config config/standard.yaml
"""

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from argklass.command import Command

from ...config import build_config


def get_benchmark_groups(config_path, exclude_tags=None):
    """Group non-private, enabled benchmarks by their definition path.

    Returns a dict mapping definition path to a sorted list of (name, defn) pairs.
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

        groups[definition].append((name, defn))

    return {k: sorted(v, key=lambda x: x[0]) for k, v in groups.items()}


def format_groups_for_ci(groups):
    """Return a sorted list of {name, select, multinode} objects for the CI matrix.

    ``name`` is the definition folder basename (e.g. "torchvision").
    ``select`` is the comma-separated list of benchmark names for --select.
    ``multinode`` is true if any benchmark in the group requires more than one machine.
    """
    result = []
    for definition, entries in groups.items():
        folder_name = Path(definition).name
        names = [name for name, _ in entries]
        multinode = any(defn.get("num_machines", 1) > 1 for _, defn in entries)
        result.append({
            "name": folder_name,
            "select": ",".join(sorted(names)),
            "multinode": multinode,
        })
    return sorted(result, key=lambda g: g["name"])


def cli_ci(args):
    """Output benchmark groups as JSON for CI matrix generation."""

    config = args.config
    if config is None:
        config = os.environ.get("MILABENCH_CONFIG", None)

    if config is None:
        print("Error: --config is required", file=sys.stderr)
        return 1

    exclude = set(args.exclude_tags.split(",")) if args.exclude_tags else set()
    groups = get_benchmark_groups(config, exclude_tags=exclude)
    result = format_groups_for_ci(groups)
    print(json.dumps(result))
    return 0


class Ci(Command):
    """Output benchmark groups as JSON for CI matrix generation."""

    name = "ci"

    # fmt: off
    @dataclass
    class Arguments:
        """Output benchmark groups as JSON for CI matrix generation."""
        config       : Optional[str] = None  # Path to the benchmark config YAML
        exclude_tags : str           = ""    # Comma-separated tags to exclude (e.g. multinode,gated)
    # fmt: on

    @staticmethod
    def execute(args):
        cli_ci(args)


COMMANDS = Ci
