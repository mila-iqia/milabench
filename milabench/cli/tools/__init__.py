"""Development and maintenance tools."""

from argklass.command import ParentCommand


class Tools(ParentCommand):
    """Development tools, environment utilities, and maintenance commands."""

    name: str = "tools"

    @staticmethod
    def module():
        import milabench.cli.tools
        return milabench.cli.tools


COMMANDS = Tools
