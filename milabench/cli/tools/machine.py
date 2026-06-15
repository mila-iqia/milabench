"""Display machine information (GPU, CPU, etc.)."""

from dataclasses import dataclass

from argklass.command import Command


class Machine(Command):
    """Display machine metadata."""

    name = "machine"

    @dataclass
    class Arguments:
        """Display machine metadata."""

    @staticmethod
    def execute(args):
        from ...metadata import machine_metadata

        try:
            from bson.json_util import dumps as to_json
        except ImportError:
            import json
            to_json = lambda obj, **kw: json.dumps(obj, **kw, default=str)

        print(to_json(machine_metadata(), indent=2))


COMMANDS = Machine
