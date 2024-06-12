import os
from dataclasses import dataclass

from coleo import Option, tooled

from ..common import _short_make_report
from ..schedule import post_comment_on_pr


# fmt: off
@dataclass
class Arguments:
    remote  : str 
    branch  : str 
    base    : str = os.getenv("MILABENCH_BASE", None)
    config  : str = os.getenv("MILABENCH_CONFIG", None)
    token   : str = os.getenv("MILABENCH_GITHUB_PAT")
# fmt: on


@tooled
def arguments():
    remote: str & Option = None

    branch: str & Option = None

    base: Option & str = os.getenv("MILABENCH_BASE", None)

    config: Option & str = os.getenv("MILABENCH_CONFIG", None)

    token: str & Option = os.getenv("MILABENCH_GITHUB_PAT")

    return Arguments(remote, branch, base, config, token)


@tooled
def cli_write_report_to_pr(args=None):
    if args is None:
        args = arguments()

    assert args.base is not None

    runfolder = os.path.join(args.base, "runs")

    def filter(folder):
        for f in ("install", "prepare"):
            if f in folder:
                return False
        return True

    runs = []
    for folder in os.listdir(runfolder):
        if filter(folder):
            runs.append(os.path.join(runfolder, folder))

    report = _short_make_report(runs, args.config)

    post_comment_on_pr(args.remote, args.branch, "```\n" + report + "\n```", args.token)
