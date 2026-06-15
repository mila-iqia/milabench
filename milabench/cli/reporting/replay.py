"""Replay validation scenario from a run folder."""

import os
from dataclasses import dataclass
from typing import Optional

from argklass.command import Command

from ...testing import replay_validation_scenario
from ...utils import validation_layers
from ...common import validation_names


class Replay(Command):
    """Replay validation scenario from a run folder."""

    name = "replay"

    # fmt: off
    @dataclass
    class Arguments:
        """Replay validation scenario from a run folder."""
        folder      : str           = None                                       # Directory containing the run data
        validations : Optional[str] = None                                       # Validation layers to enable
        dash        : str           = os.getenv("MILABENCH_DASH", "long")        # Dashboard type
        noterm      : bool          = os.getenv("MILABENCH_NOTERM", "0") == "1"  # Disable terminal
        fulltrace   : bool          = False                                      # Show full stacktrace
    # fmt: on

    @staticmethod
    def execute(args):
        layers = validation_names(args.validations)
        print(args.folder)
        replay_validation_scenario(
            args.folder,
            *validation_layers(*layers, short=not args.fulltrace),
        )


COMMANDS = Replay
