"""Resolve and display the benchmark configuration."""

from dataclasses import dataclass
from typing import Optional

from argklass.command import Command

from ...config import _config_layers
from ...merge import merge


def cli_resolve(args):
    """Generate a resolved configuration"""

    overrides = {}
    configs = [args.config, overrides]

    config = {}
    for layer in _config_layers(configs):
        config = merge(config, layer)

    wip_config = {}
    parents = []

    for benchname, benchconfig in config.items():
        is_enabled = benchconfig.get("enabled", False)
        is_weighted = benchconfig.get("weight", 0)

        parent = benchconfig.get("inherits", None)

        if parent:
            parents.append(parent)

        condition = is_enabled
        if args.lean:
            condition = is_enabled and is_weighted

        if condition:
            wip_config[benchname] = benchconfig

    parents = set(parents)
    for parent in parents:
        wip_config[parent] = config[parent]

    resolved = ["dirs", "config_file", "config_base"]
    for benchname, benchconfig in wip_config.items():
        for field in resolved:
            benchconfig.pop(field, None)

    import yaml

    print(yaml.dump(wip_config))


class Resolve(Command):
    """Resolve and display the benchmark configuration."""

    name = "resolve"

    # fmt: off
    @dataclass
    class Arguments:
        """Resolve and display the benchmark configuration."""
        config : str            = None   # Path to the benchmark config YAML
        lean   : bool           = False  # Only show enabled and weighted benchmarks
    # fmt: on

    @staticmethod
    def execute(args):
        cli_resolve(args)


COMMANDS = Resolve
