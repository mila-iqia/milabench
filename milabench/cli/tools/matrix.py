"""Generate a resolved benchmark configuration from a matrix run."""

import sys
from dataclasses import dataclass
from typing import Optional

import yaml
from argklass.command import Command
from argklass.arguments import group

from ...common import CommonArguments, deduce_arch, get_base_defaults, is_selected
from ...config import build_config
from ...merge import merge
from ...sizer import resolve_argv
from ...system import build_system_config


def clean_config(config, args):
    disabled_benches = []

    selector = is_selected(args)

    for benchname, benchconfig in config.items():
        if "system" in benchconfig:
            del benchconfig["system"]

        if not selector(benchconfig):
            disabled_benches.append(benchname)

    for bench in disabled_benches:
        config.pop(bench)


class MatrixCmd(Command):
    """Generate a resolved benchmark configuration from a matrix run."""

    name = "matrix"

    # fmt: off
    @dataclass
    class Arguments:
        """Generate a resolved benchmark configuration from a matrix run."""
        shared : CommonArguments = group(CommonArguments)
    # fmt: on

    @staticmethod
    def execute(args):
        overrides = dict()

        arch = deduce_arch()

        base = getattr(args, "base", None)
        system_file = getattr(args, "system", None)

        base_defaults = get_base_defaults(base=base, arch=arch, run_name="matrix")

        system_config = build_system_config(
            system_file, defaults={"system": base_defaults["_defaults"]["system"]}, gpu=True
        )

        overrides = merge({"*": system_config}, overrides)

        config_path = getattr(args, "config", None)
        config = build_config(base_defaults, config_path, overrides)

        clean_config(config, args)

        def resolve_args(conf, argv):
            from ...pack import Package

            pack = Package(conf)

            pack_args = []
            for k, v in argv.items():
                pack_args.append(k)
                pack_args.append(v)

            final_args = resolve_argv(pack, pack_args)

            i = 0
            for k, v in argv.items():
                if final_args[i] == k:
                    argv[k] = final_args[i + 1]
                    i += 2
                    continue
                print(f"Missing resolved argument {k}")

            return argv

        for _, conf in config.items():
            conf["argv"] = resolve_args(conf, conf["argv"])

        yaml.dump(config, sys.stdout)


COMMANDS = MatrixCmd
