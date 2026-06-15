"""Write benchmark report as a GitHub PR comment."""

import os
from dataclasses import dataclass
from typing import Optional

from argklass.command import Command

from ...common import _short_make_report
from ..slurm.schedule import post_comment_on_pr


class PR(Command):
    """Write benchmark report as a GitHub PR comment."""

    name = "pr"

    # fmt: off
    @dataclass
    class Arguments:
        """Write benchmark report as a GitHub PR comment."""
        remote : str           = None                                      # GitHub remote (owner/repo)
        branch : str           = None                                      # Branch name
        base   : Optional[str] = os.getenv("MILABENCH_BASE", None)        # Base path
        config : Optional[str] = os.getenv("MILABENCH_CONFIG", None)       # Config file
        token  : Optional[str] = os.getenv("MILABENCH_GITHUB_PAT")        # GitHub PAT
    # fmt: on

    @staticmethod
    def execute(args):
        assert args.base is not None

        runfolder = os.path.join(args.base, "runs")

        def filter_fn(folder):
            for f in ("install", "prepare"):
                if f in folder:
                    return False
            return True

        runs = []
        for folder in os.listdir(runfolder):
            if filter_fn(folder):
                runs.append(os.path.join(runfolder, folder))

        report = _short_make_report(runs, args.config)
        post_comment_on_pr(args.remote, args.branch, "```\n" + report + "\n```", args.token)


COMMANDS = PR
