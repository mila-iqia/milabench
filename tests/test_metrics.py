from milabench.utils import multilogger
from milabench.testing import replay_run
from milabench.metrics.sqlalchemy import SQLAlchemy


def test_sqlalchemy(runs_folder):
    with multilogger(SQLAlchemy()) as log:
        for msg in replay_run(runs_folder / "sedumoje.2023-03-24_13:57:35.089747"):
            log(msg)
