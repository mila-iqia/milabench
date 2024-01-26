from coleo import tooled


@tooled
def cli_sqlsetup():
    from ..metrics.sqlalchemy import generate_database_sql_setup

    generate_database_sql_setup()
