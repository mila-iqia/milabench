"""Reporting and analysis commands."""

from argklass.command import ParentCommand


class Analysis(ParentCommand):
    """Compare runs, publish results, and other analysis tools."""

    name: str = "analysis"

    @staticmethod
    def module():
        import milabench.cli.reporting
        return milabench.cli.reporting


COMMANDS = Analysis
