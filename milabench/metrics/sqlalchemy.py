from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass

from ..structs import BenchLogEntry

import sqlalchemy
from sqlalchemy.dialects import postgresql

from bson.json_util import dumps as to_json
from bson.json_util import loads as from_json

from sqlalchemy import (
    BINARY,
    JSON,
    Float,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    insert,
)
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, declarative_base

Base = declarative_base()


class Exec(Base):
    __tablename__ = "execs"

    _id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(256))
    namespace = Column(String(256))
    created_time = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON)
    status = Column(String(256))


class Pack(Base):
    __tablename__ = "packs"

    _id = Column(Integer, primary_key=True, autoincrement=True)
    exec_id = Column(Integer, ForeignKey("execs._id"), nullable=False)
    created_time = Column(DateTime, default=datetime.utcnow)
    name = Column(String(256))
    tag = Column(String(256))
    config = Column(JSON)


class Metric(Base):
    __tablename__ = "metrics"

    _id = Column(Integer, primary_key=True, autoincrement=True)
    exec_id = Column(Integer, ForeignKey("execs._id"), nullable=False)
    pack_id = Column(Integer, ForeignKey("packs._id"), nullable=False)
    name = Column(String(256))
    value = Column(Float)
    gpu_id = Column(Integer)  # GPU id


class SQLAlchemy:
    pass


META = 0
START = 1
DATA = 2


@dataclass
class PackState:
    # Order safety check
    # makes sure events are called in order
    step: int = 0
    pack: dict = None
    config: dict = None
    early_stop: bool = False
    error: int = 0


SETUP = """
CREATE SCHEMA milabench_schema;
ALTER TABLE execs SET SCHEMA milabench_schema;
ALTER TABLE packs SET SCHEMA milabench_schema;
ALTER TABLE metrics SET SCHEMA milabench_schema;

CREATE GROUP milabench_user;

GRANT CONNECT ON DATABASE milabench TO milabench_user;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA milabench_schema TO milabench_user;

ALTER TABLE execs OWNER TO milabench_user;
ALTER TABLE execs OWNER TO milabench_user;
ALTER TABLE execs OWNER TO milabench_user;

CREATE GROUP milabench_viewer;

GRANT CONNECT ON DATABASE milabench TO milabench_viewer;
GRANT select ON ALL TABLES IN SCHEMA milabench_schema TO milabench_viewer;

CREATE USER username PASSWORD 'password';
ALTER GROUP milabench_user ADD USER username;
"""


def generate_database_sql_setup(uri=None):
    """Users usally do not have create table permission.
    We generate the code to create the table so someone with permission can execute the script.
    """

    dummy = "sqlite:///sqlite.db"
    if uri is None:
        uri = dummy

    with open("setup.sql", "w") as file:

        def metadata_dump(sql, *multiparams, **params):
            sql = str(sql.compile(dialect=postgresql.dialect()))
            sql = sql.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")

            file.write(f"{sql};")
            file.write("-- \n")

        engine = sqlalchemy.create_mock_engine(
            uri, strategy="mock", executor=metadata_dump
        )
        Base.metadata.create_all(engine)

        file.write(SETUP)


class SQLAlchemy:
    def __init__(self, uri="sqlite:///sqlite.db") -> None:
        self.engine = sqlalchemy.create_engine(
            uri,
            echo=False,
            future=True,
            json_serializer=to_json,
            json_deserializer=from_json,
        )

        if uri.startswith("sqlite"):
            try:
                Base.metadata.create_all(self.engine)
            except DBAPIError as err:
                print("could not create database schema because of {err}")

        self.session = Session(self.engine)
        self.meta = None
        self.run = None
        self.states = defaultdict(PackState)

        self.pending_metrics = []
        self.batch_size = 50

    def pack_state(self, entry) -> PackState:
        return self.states[entry.tag]

    def __call__(self, entry):
        return self.on_event(entry)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self._bulk_insert()

        # Interrupted because on_end() was not called
        for state in self.states.values():
            self.update_pack_status(state.pack, "interrupted")

        status = "done"
        if len(self.states) > 0:
            status = "interrupted"

        if self.run is not None:
            self.update_run_status(status)

        self.states = defaultdict(PackState)
        self.session.commit()
        self.session.__exit__(*args, **kwargs)

    def update_run_status(self, status):
        self.run.status = status
        self.session.commit()

    def update_pack_status(self, pack, status):
        pack.status = status
        self.session.commit()

    def on_event(self, entry: BenchLogEntry):
        method = getattr(self, f"on_{entry.event}", None)

        if method is not None:
            method(entry)

    def on_new_run(self, entry):
        self.run = Exec(
            name=entry.pack.config["run_name"],
            namespace=None,
            created_time=datetime.utcnow(),
            meta=entry.data,
            status="running",
        )
        self.session.add(self.run)
        self.session.commit()
        self.session.refresh(self.run)

    def on_new_pack(self, entry):
        state = self.pack_state(entry)
        state.pack = Pack(
            exec_id=self.run._id,
            created_time=datetime.utcnow(),
            name=entry.pack.config["name"],
            tag=entry.tag,
            config=entry.pack.config,
        )
        self.session.add(state.pack)
        self.session.commit()
        self.session.refresh(state.pack)

    def on_meta(self, entry: BenchLogEntry):
        if self.run is None:
            self.on_new_run(entry)

        if entry.tag not in self.states:
            self.on_new_pack(entry)

        state = self.pack_state(entry)
        assert state.step == META
        state.step += 1

    def on_start(self, entry):
        state = self.pack_state(entry)

        assert state.step == START
        state.step += 1

    def on_phase(self, entry):
        pass

    def on_error(self, entry):
        state = self.pack_state(entry)
        state.error += 1

    def on_line(self, entry):
        pass

    def _push_metric(self, run_id, pack_id, name, value, gpu_id=None):
        self.pending_metrics.append(
            Metric(
                exec_id=run_id,
                pack_id=pack_id,
                name=name,
                value=value,
                gpu_id=gpu_id,
            )
        )

    def _change_gpudata(self, run_id, pack_id, k, v):
        for gpu_id, values in v.items():
            for metric, value in values.items():
                if metric == "memory":
                    use, mx = value
                    value = use / mx

                self._push_metric(run_id, pack_id, f"gpu.{metric}", value, gpu_id)

    def on_data(self, entry):
        state = self.pack_state(entry)
        assert state.step == DATA

        run_id = self.run._id
        pack_id = state.pack._id

        for k, v in entry.data.items():
            if k in ("task", "units", "progress"):
                continue

            # GPU data would have been too hard to query
            # so the gpu_id is moved to its own column
            # and each metric is pushed as a separate document
            if k == "gpudata":
                self._change_gpudata(run_id, pack_id, k, v)
                return

            # We request an ordered write so the document
            # will be ordered by their _id (i.e insert time)
            self._push_metric(run_id, pack_id, k, v)

        if len(self.pending_metrics) >= self.batch_size:
            self._bulk_insert()

    def _bulk_insert(self):
        if len(self.pending_metrics) <= 0:
            return

        # stmt = insert(Metric).values(self.pending_metrics)
        # self.session.execute(stmt)

        self.session.add_all(self.pending_metrics)
        self.session.commit()
        self.pending_metrics = []

    def on_stop(self, entry):
        state = self.pack_state(entry)
        assert state.step == DATA
        state.early_stop = True

    def on_end(self, entry):
        state = self.pack_state(entry)
        assert state.step == DATA

        status = "done"
        if state.early_stop:
            status = "early_stop"

        if state.error > 0:
            status = "error"

        self.update_pack_status(state.pack, status)
        self.states.pop(entry.tag)

        # even if the pack has ended we have other
        # packs still pushing metrics
        if len(self.states) == 0:
            self._bulk_insert()


if __name__ == "__main__":
    generate_database_sql_setup()
