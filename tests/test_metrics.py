import os
import zipfile

from milabench.common import _read_reports, make_summary
from milabench.metrics.report import fetch_data, make_pivot_summary
from milabench.metrics.sqlalchemy import SQLAlchemy
from milabench.testing import replay_run, replay_zipfile, show_diff
from milabench.utils import multilogger

import pytest


@pytest.fixture
def clean_db():
    if os.path.exists("sqlite.db"):
        os.remove("sqlite.db")

    yield

    # if os.path.exists("sqlite.db"):
    #     os.remove("sqlite.db")


def test_sqlalchemy_sqlite(runs_folder, clean_db, tmp_path):
    run_dir = runs_folder / "8GPUs.zip"
    run_name = "8xA100-SXM-80Go"

    with SQLAlchemy() as logger:
        replay_zipfile(run_dir, logger)

        assert len(logger.pending_metrics) == 0
        df_post = fetch_data(logger.client, run_name)

    run_name = df_post.run.iloc[0]
    replicated = make_pivot_summary(run_name, df_post)

    print(replicated)

    # Unpack the zip so we can read the reports with the old code
    with zipfile.ZipFile(run_dir, "r") as archive:
        archive.extractall(tmp_path)

    # Compare
    # -------
    runs = [tmp_path]
    reports = _read_reports(*runs)
    summary = make_summary(reports)

    print(summary)

    show_diff(summary, replicated)
