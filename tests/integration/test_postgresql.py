import os

from milabench.common import _read_reports, make_summary
from milabench.metrics.report import fetch_data, make_pivot_summary
from milabench.metrics.sqlalchemy import SQLAlchemy, create_database
from milabench.testing import replay_run, show_diff
from milabench.utils import multilogger

# Setup POSTGRESQL instance
# -------------------------
#
# sudo -i -u postgres
# psql
# CREATE DATABASE milabench;
# \c milabench
# setup.sql

USER = os.getenv("POSTGRES_USER", "username")
PSWD = os.getenv("POSTGRES_PSWD", "password")
DB = os.getenv("POSTGRES_DB", "milabench")
HOST = os.getenv("POSTGRES_HOST", "localhost")
PORT = os.getenv("POSTGRES_PORT", 5432)

TEST_INSTANCE = f"postgresql://{USER}:{PSWD}@{HOST}:{PORT}/{DB}"
create_database(TEST_INSTANCE)


def test_sqlalchemy_postgresql(runs_folder):
    run_dir = runs_folder / "sedumoje.2023-03-24_13:57:35.089747"

    with SQLAlchemy(TEST_INSTANCE) as logger:
        with multilogger(logger) as log:
            for msg in replay_run(run_dir):
                log(msg)

        df_post = fetch_data(logger.client, "sedumoje")

    replicated = make_pivot_summary("sedumoje", df_post)

    # Compare
    # -------
    runs = [run_dir]
    reports = _read_reports(*runs)
    summary = make_summary(reports.values())

    show_diff(summary, replicated)
