import os
import sys

import pytest
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from milabench.metrics.sqlalchemy import SQLAlchemy, create_database, Exec, Metric, Pack, Weight
from milabench.metrics.report import fetch_data_by_id, make_pandas_report
from milabench.web.plot import (
    grouped_plot,
    regular_average,
    average_drop_min_max,
    perf_per_bench_query,
    weighted_perf_per_bench_query,
    sql_direct_report
)

def is_ci():
    return os.environ.get("GITHUB_ACTIONS ", "false").lower() == "true"


# FIXME: This test expects some data to be present in the database
if is_ci():
    pytest.skip("Skipping DB integration tests in CI environment", allow_module_level=True)


# PostgreSQL connection settings
USER = os.getenv("POSTGRES_USER", "username")
PSWD = os.getenv("POSTGRES_PSWD", "password")
DB = os.getenv("POSTGRES_DB", "milabench")
HOST = os.getenv("POSTGRES_HOST", "localhost")
PORT = os.getenv("POSTGRES_PORT", 5432)

TEST_INSTANCE = f"postgresql://{USER}:{PSWD}@{HOST}:{PORT}/{DB}"


@pytest.fixture(scope="function")
def db_session():
    """Create a database session for testing"""
    create_database(TEST_INSTANCE)
    engine = create_engine(TEST_INSTANCE)

    # Create tables
    from milabench.metrics.sqlalchemy import Base
    Base.metadata.create_all(engine)

    session = Session(engine)
    yield session

    session.close()
    engine.dispose()


@pytest.fixture(scope="function")
def sample_data(db_session):
    """Get existing data for execution_id 48"""
    session = db_session

    # Check if execution_id 48 exists
    exec_record = session.query(Exec).filter(Exec._id == 48).first()
    if not exec_record:
        pytest.skip("Execution ID 48 does not exist in the database")

    # Get related data
    packs = session.query(Pack).filter(Pack.exec_id == 48).all()
    metrics = session.query(Metric).filter(Metric.exec_id == 48).all()
    weights = session.query(Weight).filter(Weight.profile == "default").all()

    return {
        "exec_id": 48,
        "exec_record": exec_record,
        "packs": packs,
        "metrics": metrics,
        "weights": weights,
        "session": session
    }


def test_sql_direct_report_basic(sample_data):
    """Test sql_direct_report with basic parameters"""
    session = sample_data["session"]
    exec_ids = [sample_data["exec_id"]]

    query = sql_direct_report(exec_ids, profile="default", drop_min_max=True)
    result = session.execute(query)
    rows = result.fetchall()

    # Should return results if data exists
    if len(rows) > 0:
        assert hasattr(rows[0], 'exec_id')
        assert hasattr(rows[0], 'bench')
        assert hasattr(rows[0], 'n')
        assert hasattr(rows[0], 'ngpu')
        assert hasattr(rows[0], 'perf')
        assert hasattr(rows[0], 'sem')
        assert hasattr(rows[0], 'std')
        assert hasattr(rows[0], 'score')
        assert hasattr(rows[0], 'weight')
        assert hasattr(rows[0], 'enabled')
        assert hasattr(rows[0], 'log_score')
        assert hasattr(rows[0], 'order')
        assert hasattr(rows[0], 'weight_total')

    # Check that left join was used
    weight = session.query(Weight).filter(Weight.profile == "default").all()

    assert len(weight) == 49
    assert len(rows) == len(weight)

    for r in rows:
        print(r)


def test_sql_direct_report_vs_pandas(sample_data, db_session):
    """Test sql_direct_report with basic parameters"""
    session = sample_data["session"]
    exec_ids = [sample_data["exec_id"]]

    query = sql_direct_report(exec_ids, profile="default", drop_min_max=True)
    result = session.execute(query)
    rows = result.fetchall()

    # Make pandas report
    pd_report, replicated = make_pandas_report(db_session.connection(), 48)

    sq_report = pd.DataFrame(rows)
    pd_report = pd_report.reset_index()

    bench_names = pd_report["index"]

    no_nan = lambda x: x if not pd.isna(x) else 0

    acc = 0
    for name in bench_names:
        pd_row = pd_report[pd_report["index"] == name]
        sq_row = sq_report[sq_report["bench"] == name]

        sq_count = sq_row["count"].iloc[0]
        pd_count = no_nan(replicated[name]["train_rate"]["debug_count"])
        total_count = no_nan(replicated[name]["train_rate"]["count"])
        
        pandas_perf = no_nan(pd_row["perf"].iloc[0])
        sql_perf = no_nan(sq_row["perf"].iloc[0])

        diff = pandas_perf - sql_perf
        acc += abs(diff)
    
        if abs(diff) > 1e-6 or pd_count != sq_count:
            print(f"{name:>40}: {pandas_perf:10.2f} != {sql_perf:10.2f} ({diff:10.2f}) "
                  f"| {int(pd_count):4d} != {int(sq_count):4d} ({int(pd_count) - int(sq_count):4d}) "
                  f"{int(total_count):4d}")
        

    print(f"Total diff: {acc:10.2f}")
    assert acc < 1e-6