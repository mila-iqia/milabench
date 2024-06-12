from dataclasses import dataclass

from coleo import Option, tooled
import yaml
import sys

from ..common import deduce_arch, build_config, build_system_config, get_base_defaults, merge, is_selected


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
        if 'system' in benchconfig:
            del benchconfig['system']

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

    base_defaults = get_base_defaults(
        base=args.base,
        arch=arch,
        run_name='matrix'
    )

    system_config = build_system_config(
        args.system,
        defaults={"system": base_defaults["_defaults"]["system"]},
        gpu=True
    )

    overrides = merge({"*": system_config}, overrides)

    config = build_config(base_defaults, args.config, overrides)

    clean_config(config, args)


    for k in config:
        print(k)

    # yaml.dump(config, sys.stdout)