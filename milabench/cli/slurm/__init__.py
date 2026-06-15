"""Slurm-related commands."""

from argklass.command import ParentCommand


class Slurm(ParentCommand):
    """Manage Slurm integration: system config generation and job scheduling."""

    name: str = "slurm"

    @staticmethod
    def module():
        import milabench.cli.slurm
        return milabench.cli.slurm


COMMANDS = Slurm
