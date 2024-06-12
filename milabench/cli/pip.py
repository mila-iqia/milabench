from dataclasses import dataclass, field

from coleo import Option, tooled

from ..common import get_multipack, run_sync


# fmt: off
@dataclass
class Arguments:
    args: list = field(default_factory=list)
# fmt: on


@tooled
def arguments():
    # pip arguments
    # [remainder]
    args: Option = []

    return Arguments(args)


@tooled
def cli_pip(args=None):
    """Run pip on every pack"""
    if args is None:
        args = arguments()
    mp = get_multipack(run_name="pip")

    for pack in mp.packs.values():
        run_sync(pack.pip_install(*args.args))
