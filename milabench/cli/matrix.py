from dataclasses import dataclass
import sys

import yaml
from coleo import Option, tooled

from ..system import build_system_config
from ..common import deduce_arch, build_config, get_base_defaults, merge, is_selected
from ..sizer import resolve_argv, scale_argv


# fmt: off
@dataclass
class Arguments:
    base      : str = None
    system    : str = None
    config    : str = None
    select    : str = None
    exclude   : str = None
# fmt: on


@tooled
def arguments():
    base: Option & str

    system: Option & str = None

    config: Option & str = None

    select: Option & str = None

    exclude: Option & str = None

    return Arguments(base, system, config, select, exclude)


def clean_config(config, args):
    disabled_benches = []

    if args.select:
        args.select = set(args.select.split(","))

    if args.exclude:
        args.exclude = set(args.exclude.split(","))

    for benchname, benchconfig in config.items():
        if "system" in benchconfig:
            del benchconfig["system"]

        if not is_selected(benchconfig, args):
            disabled_benches.append(benchname)

    for bench in disabled_benches:
        config.pop(bench)


@tooled
def cli_matrix_run(args=None):
    if args is None:
        args = arguments()

    overrides = dict()

    arch = deduce_arch()

    base_defaults = get_base_defaults(base=args.base, arch=arch, run_name="matrix")

    system_config = build_system_config(
        args.system, defaults={"system": base_defaults["_defaults"]["system"]}, gpu=True
    )

    overrides = merge({"*": system_config}, overrides)

    config = build_config(base_defaults, args.config, overrides)

    clean_config(config, args)

    def resolve_args(conf, argv):
        from ..pack import Package
        pack = Package(conf)

        args = []
        for k, v in argv.items():
            args.append(k)
            args.append(v)
    
        sized_args = scale_argv(pack, args)
        final_args = resolve_argv(pack, sized_args)

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

    # for k in config:
    #     print(k)

    yaml.dump(config, sys.stdout)
