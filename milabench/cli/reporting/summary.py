"""Produce a JSON summary of a previous run."""

import json
from dataclasses import dataclass
from typing import Optional

from argklass.command import Command

from ...common import _read_reports
from ...summary import make_summary


class Summary(Command):
    """Produce a JSON summary of a previous run."""

    name = "summary"

    # fmt: off
    @dataclass
    class Arguments:
        """Produce a JSON summary of a previous run."""
        runs : list[str]     = None  # Directory(ies) containing the run data
        out  : Optional[str] = None  # Output file
    # fmt: on

    @staticmethod
    def execute(args):
        all_data = _read_reports(*args.runs)
        summary = make_summary(all_data)

        if args.out is not None:
            with open(args.out, "w") as file:
                json.dump(summary, file, indent=4)
        else:
            print(json.dumps(summary, indent=4))


COMMANDS = Summary
