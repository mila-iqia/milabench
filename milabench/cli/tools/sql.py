"""Setup/manage the SQL database for milabench results."""

from dataclasses import dataclass

from argklass.command import Command


class Sql(Command):
    """Generate the SQL setup scripts for the milabench database."""

    name = "sqlsetup"

    @dataclass
    class Arguments:
        """Generate the SQL setup scripts for the milabench database."""

    @staticmethod
    def execute(args):
        from ...metrics.sqlalchemy import generate_database_sql_setup

        generate_database_sql_setup()


COMMANDS = Sql
