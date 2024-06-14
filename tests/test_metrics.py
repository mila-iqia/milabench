from milabench.common import _read_reports, make_summary
from milabench.metrics.report import fetch_data, make_pivot_summary
from milabench.metrics.sqlalchemy import SQLAlchemy
from milabench.testing import replay_run, show_diff
from milabench.utils import multilogger


def test_sqlalchemy_sqlite(runs_folder):
    run_dir = runs_folder / "sedumoje.2023-03-24_13:57:35.089747"
    run_name = "sedumoje"

    with SQLAlchemy() as logger:
        with multilogger(logger) as log:
            for msg in replay_run(run_dir):
                log(msg)

        assert len(logger.pending_metrics) == 0
        df_post = fetch_data(logger.client, run_name)

    print(df_post)
    replicated = make_pivot_summary(run_name, df_post)

    # Compare
    # -------
    runs = [run_dir]
    reports = _read_reports(*runs)
    summary = make_summary(reports.values())

    show_diff(summary, replicated)
