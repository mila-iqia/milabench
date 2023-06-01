from copy import deepcopy

from milabench.utils import multilogger
from milabench.testing import replay_run, show_diff
from milabench.metrics.sqlalchemy import SQLAlchemy
from milabench.metrics.report import fetch_data, make_pivot_summary
from milabench.cli import _read_reports, make_summary


def test_sqlalchemy_sqlite(runs_folder):
    run_dir = runs_folder / "sedumoje.2023-03-24_13:57:35.089747"

    with SQLAlchemy() as logger:
        with multilogger(logger) as log:
            for msg in replay_run(run_dir):
                log(msg)

        df_post = fetch_data(logger.client, "sedumoje")

    replicated = make_pivot_summary("sedumoje", df_post)

    # Compare
    # -------
    from milabench.report import make_dataframe

    runs = [run_dir]
    reports = _read_reports(*runs)
    summary = make_summary(reports.values())

    show_diff(summary, replicated)
