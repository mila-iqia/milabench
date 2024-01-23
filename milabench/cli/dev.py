import os
import subprocess
import sys
from dataclasses import dataclass

from coleo import Option, tooled

from ..common import Option, get_multipack, selection_keys


# fmt: off
@dataclass
class Arguments:
    select  : str = "*"
# fmt: on


@tooled
def arguments():
    # The name of the benchmark to develop
    select: Option & str = "*"

    return Arguments(select)


@tooled
def cli_dev(args=None):
    if args is None:
        args = arguments()

    mp = get_multipack(run_name="dev")

    for pack in mp.packs.values():
        if args.select in selection_keys(pack.config):
            break
    else:
        sys.exit(f"Cannot find a benchmark with selector {args.select}")

    subprocess.run(
        [os.environ["SHELL"]],
        env=pack.full_env(),
        cwd=pack.dirs.code,
    )
