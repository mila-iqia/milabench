import os
import pytest

from milabench.web import push_server
from milabench.metrics.sqlalchemy import SQLAlchemy, create_database
from milabench.metrics.report import fetch_data, make_pivot_summary


@pytest.fixture()
def database():
    USER = os.getenv("POSTGRES_USER", "username")
    PSWD = os.getenv("POSTGRES_PSWD", "password")
    DB = os.getenv("POSTGRES_DB", "milabench")
    HOST = os.getenv("POSTGRES_HOST", "localhost")
    PORT = os.getenv("POSTGRES_PORT", 5432)

    TEST_INSTANCE = f"postgresql://{USER}:{PSWD}@{HOST}:{PORT}/{DB}"
    create_database(TEST_INSTANCE)
    return TEST_INSTANCE


@pytest.fixture()
def app():
    app = push_server({
        "TESTING": True,
    })

    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()


def test_push_server(client, runs_folder, database):
    response = client.post("/push", data={
        "file": (runs_folder / "8GPUs.zip").open("rb"),
    })

    assert response.status_code == 200

    with SQLAlchemy(database) as logger:
        df_post = fetch_data(logger.client, "8xA100-SXM-80Go")

    replicated = make_pivot_summary("8xA100-SXM-80Go", df_post)

    print(replicated)
