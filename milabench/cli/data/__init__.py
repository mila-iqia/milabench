"""Data management commands."""

from argklass.command import ParentCommand


class Data(ParentCommand):
    """Manage benchmark data: prepare+run, archiving, shared setup, gated models."""

    name: str = "data"

    @staticmethod
    def module():
        import milabench.cli.data
        return milabench.cli.data


COMMANDS = Data
