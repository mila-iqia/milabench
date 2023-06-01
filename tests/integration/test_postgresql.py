from milabench.utils import multilogger
from milabench.testing import replay_run
from milabench.metrics.sqlalchemy import SQLAlchemy, Pack, Exec, Metric

import pandas as pd
from bson.json_util import dumps as to_json
from bson.json_util import loads as from_json
import sqlalchemy
from sqlalchemy.orm import Session


# Setup POSTGRESQL instance
# -------------------------
#
# sudo -i -u postgres
# psql
# CREATE DATABASE milabench;
# \c milabench
# setup.sql

TEST_INSTANCE = "postgresql://username:password@localhost:5432/milabench"
TEST_INSTANCE = "sqlite:///sqlite.db"


def test_sqlalchemy(runs_folder, monkeypatch):
    with multilogger(SQLAlchemy(TEST_INSTANCE)) as log:
        for msg in replay_run(runs_folder / "sedumoje.2023-03-24_13:57:35.089747"):
            log(msg)


def test_sqlalchemy_report():
    engine = sqlalchemy.create_engine(
        TEST_INSTANCE,
        echo=False,
        future=True,
        json_serializer=to_json,
        json_deserializer=from_json,
    )

    stmt = (
        sqlalchemy.select(
            Exec.name,
            Pack.name,
            Metric.name,
            sqlalchemy.func.avg(Metric.value),
            sqlalchemy.func.count(Metric.value),
        )
        .join(Exec, Metric.exec_id == Exec._id)
        .join(Pack, Metric.pack_id == Pack._id)
        .group_by(Exec.name, Pack.name, Metric.name)
    )

    results = []
    with Session(engine) as sess:
        for row in sess.execute(stmt):
            results.append(row)

    print(pd.DataFrame(results))
