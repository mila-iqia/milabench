import numbers
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime

import sqlalchemy
from bson.json_util import dumps as to_json, loads as from_json
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, declarative_base

from ..structs import BenchLogEntry

Base = declarative_base()


class Exec(Base):
    __tablename__ = "execs"

    _id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(256))
    namespace = Column(String(256))
    created_time = Column(DateTime, default=datetime.utcnow)
    meta = Column(JSON)
    status = Column(String(256))

    __table_args__ = (Index("exec_name", "name"),)


class Pack(Base):
    __tablename__ = "packs"

    _id = Column(Integer, primary_key=True, autoincrement=True)
    exec_id = Column(Integer, ForeignKey("execs._id"), nullable=False)
    created_time = Column(DateTime, default=datetime.utcnow)
    name = Column(String(256))
    tag = Column(String(256))
    config = Column(JSON)
    command = Column(JSON)

    __table_args__ = (
        Index("exec_pack_query", "exec_id"),
        Index("pack_query", "name", "exec_id"),
        Index("pack_tag", "tag"),
    )


class Metric(Base):
    __tablename__ = "metrics"

    _id = Column(Integer, primary_key=True, autoincrement=True)
    exec_id = Column(Integer, ForeignKey("execs._id"), nullable=False)
    pack_id = Column(Integer, ForeignKey("packs._id"), nullable=False)

    # Insert Time
    order = Column(Integer)

    name = Column(String(256))
    namespace = Column(String(256))
    value = Column(Float)
    unit = Column(String(128))

    job_id = Column(Integer)  # Job ID
    gpu_id = Column(String(36))  # GPU id

    __table_args__ = (
        Index("metric_query", "exec_id", "pack_id"),
        Index("metric_name", "name"),
    )


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
    start: int = 0
    command = None


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
            sql = sql.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS")

            file.write(f"{sql};")
            file.write("-- \n")

        engine = sqlalchemy.create_mock_engine(
            uri, strategy="mock", executor=metadata_dump
        )
        Base.metadata.create_all(engine)


def create_database(uri):
    engine = sqlalchemy.create_engine(
        uri,
        echo=False,
        future=True,
        json_serializer=to_json,
        json_deserializer=from_json,
    )

    try:
        Base.metadata.create_all(engine)
    except DBAPIError:
        print("could not create database schema because of {err}")


def _get_pack_ids(pack):
    devices = pack.config.get("devices", ["ALL"])

    job_id = str(pack.config.get("job-number", 0))
    gpu_id = ",".join(str(i) for i in devices)

    return job_id, gpu_id


class SQLAlchemy:
    def __init__(self, uri="sqlite:///sqlite.db", meta_override=None) -> None:
        if uri.startswith("sqlite"):
            create_database(uri)

        self.engine = sqlalchemy.create_engine(
            uri,
            echo=False,
            future=True,
            json_serializer=to_json,
            json_deserializer=from_json,
        )

        self.meta_override = meta_override
        self.session = Session(self.engine)
        self.meta = None
        self.run = None
        self.states = defaultdict(PackState)

        self.pending_metrics = []
        self.batch_size = 1000

    @property
    def client(self):
        return self.engine

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
            if state.pack:
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
            meta=self.meta_override or entry.data,
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
        if entry.tag not in self.states:
            # We have not received the meta tag
            self.on_meta(BenchLogEntry(entry.pack, event="meta", data={}))

        state = self.pack_state(entry)

        state.pack.command = entry.data["command"]
        state.start = entry.data["time"]

        assert state.step == START
        state.step += 1

    def on_phase(self, entry):
        pass

    def on_error(self, entry):
        state = self.pack_state(entry)
        state.error += 1

    def on_line(self, entry):
        pass

    def _push_metric(
        self,
        run_id,
        pack_id,
        name,
        value,
        gpu_id=None,
        job_id=None,
        namespace=None,
        unit=None,
    ):
        if not isinstance(value, numbers.Number):
            print(f"Unexpected value {value} for metric {name}")
            return

        self.pending_metrics.append(
            Metric(
                exec_id=run_id,
                pack_id=pack_id,
                order=time.time(),
                name=name,
                namespace=namespace,
                unit=unit,
                value=value,
                gpu_id=gpu_id,
                job_id=job_id,
            )
        )

    def _change_gpudata(self, run_id, pack_id, k, v, jobid):
        for gpu_id, values in v.items():
            for metric, value in values.items():
                if metric == "memory":
                    use, mx = value
                    value = use / mx

                self._push_metric(
                    run_id, pack_id, f"gpu.{metric}", value, gpu_id, job_id=jobid
                )

    def on_data(self, entry):
        state = self.pack_state(entry)
        assert state.step == DATA

        run_id = self.run._id
        pack_id = state.pack._id
        job_id, gpu_id = _get_pack_ids(state.pack)

        if "progress" in entry.data:
            return

        data = deepcopy(entry.data)

        # GPU
        gpudata = data.pop("gpudata", None)
        if gpudata is not None:
            # GPU data would have been too hard to query
            # so the gpu_id is moved to its own column
            # and each metric is pushed as a separate document
            self._change_gpudata(run_id, pack_id, "gpudata", gpudata, job_id)
        else:
            # Standard
            unit = data.pop("units", None)
            task = data.pop("task", None)

            if len(data) == 1:
                k, v = list(data.items())[0]

                self._push_metric(
                    run_id,
                    pack_id,
                    k,
                    v,
                    gpu_id=gpu_id,
                    job_id=job_id,
                    unit=unit,
                    namespace=task,
                )
            else:
                print(f"Unknown format {entry.data}")

        if len(self.pending_metrics) >= self.batch_size:
            self._bulk_insert()

    def _bulk_insert(self):
        if len(self.pending_metrics) <= 0:
            return

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

        run_id = self.run._id
        pack_id = state.pack._id

        job_id, gpu_id = _get_pack_ids(state.pack)

        end = entry.data["time"]
        self._push_metric(
            run_id, pack_id, "walltime", end - state.start, gpu_id, job_id
        )

        self._push_metric(
            run_id, pack_id, "return_code", entry.data["return_code"], gpu_id, job_id
        )

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
