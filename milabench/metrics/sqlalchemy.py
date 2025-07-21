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
    text,
    Computed,
    Boolean,
    UniqueConstraint,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, declarative_base
from sqlalchemy.orm import declared_attr

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

    # Visibility works as a level, this way we can do show all runs <= 2
    #  0= public
    #  1= private
    visibility = Column(Integer, default=0)


    __table_args__ = (
        Index("exec_name", "name"),
        Index("exec_visibility", "visibility"),
        Index(
            'execs_meta_gpus_0_product_idx',
            text("(meta -> 'accelerators' -> 'gpus' -> '0' ->> 'product')"),
            postgresql_using='btree'
        ),
        # Pivot query optimization indexes
        Index(
            'idx_exec_meta_pytorch_version',
            text("(meta -> 'pytorch' ->> 'version')"),
            postgresql_using='btree'
        ),
        Index(
            'idx_exec_meta_pytorch_torch',
            text("(meta -> 'pytorch' ->> 'torch')"),
            postgresql_using='btree'
        ),
        # Index(
        #     'idx_exec_meta_accelerators_gin',
        #     text("meta -> 'accelerators'"),
        #     postgresql_using='gin'
        # ),
        # Index(
        #     'idx_exec_meta_pytorch_gin',
        #     text("meta -> 'pytorch'"),
        #     postgresql_using='gin'
        # )
    )

    def as_dict(self):
        return {
            "_id": self._id,
            "name": self.name,
            "namespace": self.namespace,
            "created_time": self.created_time,
            "meta": self.meta,
            "status": self.status
        }


class Pack(Base):
    __tablename__ = "packs"

    _id = Column(Integer, primary_key=True, autoincrement=True)
    exec_id = Column(Integer, ForeignKey("execs._id"), nullable=False)
    created_time = Column(DateTime, default=datetime.utcnow)
    name = Column(String(256))
    tag = Column(String(256))
    config = Column(JSON)
    command = Column(JSON)
    status = Column(String(256))

    #
    @declared_attr
    def ngpu(cls):
        try:
            if Base.metadata.bind and Base.metadata.bind.dialect.name != 'sqlite':
                return Column(Integer, Computed("((config->>'num_machines')::int * json_array_length(config->'devices'))"))
            else:
                return Column(Integer)  # Empty placeholder
        except:
            return Column(Integer)  # Empty placeholder


    # @property
    # def gpu_count(self):
    #     return len(self.config.get("devices", [1])) if self.config else 1

    # @property
    # def node_count(self):
    #     return self.config.get("num_machines", 1) if self.config else 1

    # @property
    # def ngpu(self):
    #     return self.gpu_count * self.node_count

    __table_args__ = (
        Index("exec_pack_query", "exec_id"),
        Index("pack_query", "name", "exec_id"),
        Index("pack_tag", "tag"),
        # Pivot query optimization indexes
        Index("idx_pack_name", "name"),  # For faster name lookups in Weight joins
    )

    def as_dict(self):
        return {
            "_id": self._id,
            "exec_id": self.exec_id,
            "name": self.name,
            "tag": self.tag,
            "created_time": self.created_time,
            "config": self.config,
            "command": self.command,
            "status": self.status
        }


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
        # Pivot query optimization indexes
        Index("idx_metric_name_value", "name", "value"),  # For DISTINCT queries with filtering
        Index("idx_metric_exec_pack_name", "exec_id", "pack_id", "name"),  # For base_report_view joins
        Index("idx_metric_pack_name", "pack_id", "name"),  # For faster pack-metric joins
        Index("idx_metric_exec_name", "exec_id", "name"),  # For exec-metric aggregations
    )

    def as_dict(self):
        return {
            "_id": self._id,
            "exec_id": self.exec_id,
            "pack_id": self.pack_id,
            "order": self.order,
            "name": self.name,
            "namespace": self.namespace,
            "value": self.value,
            "unit": self.unit,
            "job_id": self.job_id,
            "gpu_id": self.gpu_id,
        }


class SavedQuery(Base):
    """Save queries to easy access"""
    __tablename__ = "saved_queries"

    _id = Column(Integer, primary_key=True, autoincrement=True)

    name = Column(String(256))
    query = Column(JSON)
    created_time = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("saved_queries_name", "name"),
    )

    def as_dict(self):
        return {
            "_id": self._id,
            "name": self.name,
            "query": self.query,
            "created_time": self.created_time,
        }


class Weight(Base):
    """Save queries to easy access"""
    __tablename__ = "weights"

    _id = Column(Integer, primary_key=True, autoincrement=True)

    profile = Column(String(256), nullable=False)
    pack = Column(String(256), nullable=False)
    weight = Column(Integer, default=0, nullable=False)
    # 1XXX: Synthetic
    # 2XXX: CV
    # 3XXX: NLP
    # 4XXX: RL
    # 5XXX: Graphs

    # 1XX: Transformer
    # 2XX: Convnets
    # 3XX: MLP
    priority = Column(Integer, default=0, nullable=False)
    enabled = Column(Boolean, default=False, nullable=False)
    group1 = Column(String(256), nullable=True)
    group2 = Column(String(256), nullable=True)
    group3 = Column(String(256), nullable=True)
    group4 = Column(String(256), nullable=True)

    __table_args__ = (
        UniqueConstraint("profile", "pack", name="uq_profile_pack"),
        Index("weight_profile_pack", "profile", "pack"),
        # Pivot query optimization indexes
        Index("idx_weight_profile_enabled", "profile", "enabled"),  # For enabled weight filtering
        Index("idx_weight_pack", "pack"),  # For pack name lookups
        Index("idx_weight_profile_priority", "profile", "priority"),  # For ordered results
    )

    def __repr__(self):
        return f"Weight({self.as_dict()})"

    def as_dict(self):
        return {
            "_id": self._id,
            "profile": self.profile,
            "pack": self.pack,
            "weight": self.weight,
            "priority": self.priority,
            "enabled": self.enabled,
            "group1": self.group1,
            "group2": self.group2,
            "group3": self.group3,
            "group4": self.group4,
        }


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
    import os

    dummy = "sqlite:///sqlite.db"
    if uri is None:
        uri = dummy


    filename = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "tables.sql")

    with open(filename, "w") as file:
        file.write("--\n")
        file.write("-- Generated using:\n")
        file.write("--\n")
        file.write("--      python -m milabench.metrics.sqlalchemy\n")
        file.write("--\n")

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


        file.write(base_weight_profile().replace("        ", ""))
        file.write("-- \n")



def base_weight_profile():
    return """
        INSERT INTO
            weights (profile, weight, priority, pack, enabled, group1, group2)
        VALUES
            ('default', 0, 1000, 'fp16', TRUE, 'SYNTHETIC', 'FLOPS'),
            ('default', 0, 1001, 'bf16', TRUE, 'SYNTHETIC', 'FLOPS'),
            ('default', 0, 1002, 'tf32', TRUE, 'SYNTHETIC', 'FLOPS'),
            ('default', 0, 1003, 'fp32', TRUE, 'SYNTHETIC', 'FLOPS'),
            ('default', 0, 2201, 'convnext_large-fp32', TRUE, 'CV', 'CONVNET'),
            ('default', 0, 2202, 'convnext_large-fp16', TRUE, 'CV', 'CONVNET'),
            ('default', 0, 2203, 'convnext_large-tf32', TRUE, 'CV', 'CONVNET'),
            ('default', 1, 2204, 'convnext_large-tf32-fp16', TRUE, 'CV', 'CONVNET'),
            ('default', 1, 2205, 'resnet50', TRUE, 'CV', 'CONVNET'),
            ('default', 0, 2206, 'resnet50-noio', TRUE, 'CV', 'CONVNET'),
            ('default', 0, 2207, 'resnet152-ddp-gpus', TRUE, 'CV', 'CONVNET'),
            ('default', 1, 2208, 'regnet_y_128gf', TRUE, 'CV', 'CONVNET'),
            ('default', 0, 2209, 'lightning', TRUE, 'CV', 'CONVNET'),
            ('default', 1, 2210, 'lightning-gpus', TRUE, 'CV', 'CONVNET'),
            ('default', 0, 2211, 'focalnet', TRUE, 'CV', 'CONVNET'),
            ('default', 0, 2012, 'diffusion-single', TRUE, 'CV', 'DIFFUSION'),
            ('default', 1, 2013, 'diffusion-gpus', TRUE, 'CV', 'DIFFUSION'),
            ('default', 1, 2014, 'diffusion-nodes', FALSE, 'CV', 'DIFFUSION'),
            ('default', 0, 2101, 'dinov2-giant-single', TRUE, 'CV', 'TRANSFORMER'),
            ('default', 1, 2102, 'dinov2-giant-gpus', TRUE, 'CV', 'TRANSFORMER'),
            ('default', 0, 2103, 'dinov2-giant-nodes', FALSE, 'CV', 'TRANSFORMER'),
            ('default', 1, 2104, 'llava-single', TRUE, 'CV', 'TRANSFORMER'),
            ('default', 0, 2105, 'llava-gpus', FALSE, 'CV', 'TRANSFORMER'),
            ('default', 1, 2106, 'vjepa-single', TRUE, 'CV', 'TRANSFORMER'),
            ('default', 1, 2107, 'vjepa-gpus', TRUE, 'CV', 'TRANSFORMER'),
            ('default', 0, 3100, 'bert-fp32', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 0, 3101, 'bert-fp16', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 0, 3102, 'bert-tf32', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 1, 3103, 'bert-tf32-fp16', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 0, 3104, 't5', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 1, 3105, 'reformer', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 0, 3106, 'whisper', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 1, 3107, 'llama', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 1, 3108, 'llm-lora-single', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 1, 3109, 'llm-lora-ddp-gpus', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 1, 3110, 'llm-lora-ddp-nodes', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 1, 3111, 'llm-lora-mp-gpus', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 1, 3112, 'llm-full-mp-gpus', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 1, 3113, 'llm-full-mp-nodes', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 1, 3114, 'rlhf-single', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 0, 3115, 'rlhf-gpus', TRUE, 'NLP', 'TRANSFORMER'),
            ('default', 1, 4201, 'torchatari', TRUE, 'RL', 'CONVNET'),
            ('default', 1, 4302, 'brax', TRUE, 'RL', 'MLP'),
            ('default', 0, 4303, 'dqn', TRUE, 'RL', 'MLP'),
            ('default', 1, 4304, 'ppo', TRUE, 'RL', 'MLP'),
            ('default', 0, 4305, 'cleanrljax', FALSE, 'RL', 'MLP'),
            ('default', 1, 5000, 'pna', TRUE, 'GRAPHS', 'GNN'),
            ('default', 1, 5001, 'dimenet', TRUE, 'GRAPHS', 'GNN'),
            ('default', 1, 5002, 'recursiongfn', TRUE, 'GRAPHS', 'GFlow')
        ;"""

def create_database(uri):
    engine = sqlalchemy.create_engine(
        uri,
        echo=False,
        future=True,
        json_serializer=to_json,
        json_deserializer=from_json,
    )

    try:
        Base.metadata.bind = engine
        Base.metadata.create_all(engine)

        with Session(engine) as session:
            session.execute(text(base_weight_profile()))
            session.commit()

    except DBAPIError as err:
        print(f"could not create database schema because of {err}")


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

    def start_new_run(self):
        self.meta = None
        self.run = None
        self.states = defaultdict(PackState)

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
        order=None,
        gpu_id=None,
        job_id=None,
        namespace=None,
        unit=None,
    ):
        if not isinstance(value, numbers.Number):
            print(f"Unexpected value {value} for metric {name}")
            return

        if order is None:
            order = time.time()

        def get_gpu_id(gid):
            try:
                return int(gid)
            except:
                return -1

        self.pending_metrics.append(
            Metric(
                exec_id=run_id,
                pack_id=pack_id,
                order=order,
                name=name,
                namespace=namespace,
                unit=unit,
                value=value,
                gpu_id=gpu_id, # get_gpu_id(gpu_id),
                job_id=job_id,
            )
        )

    def _change_gpudata(self, run_id, pack_id, k, v, jobid, metric_time=None):
        for gpu_id, values in v.items():
            for metric, value in values.items():
                unit = None
                match metric:
                    case "memory":
                        use, mx = value
                        value = use / mx
                        unit = "%"
                    case "load":
                        unit = "%"
                    case "temperature":
                        unit = "°C"
                    case "power":
                        unit = "W"

                self._push_metric(
                    run_id, pack_id, f"gpu.{metric}", value, gpu_id=gpu_id, job_id=jobid, order=metric_time, unit=unit
                )

    def _push_composed_data(self, run_id, pack_id, gpu_id, k, v, jobid, metric_time):
        for metric, value in v.items():
            unit = None

            match metric:
                case "memory":
                    used, mx = value
                    value = used / mx
                    unit = "%"

            self._push_metric(
                run_id, pack_id, f"{k}.{metric}", value, gpu_id=gpu_id, job_id=jobid, order=metric_time, unit=unit
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

        metric_time = data.pop("time", time.time())

        # GPU
        if (gpudata := data.pop("gpudata", None)) is not None:
            # GPU data would have been too hard to query
            # so the gpu_id is moved to its own column
            # and each metric is pushed as a separate document
            self._change_gpudata(run_id, pack_id, "gpudata", gpudata, job_id, metric_time=metric_time)

        elif (process := data.pop("process", None)) is not None:
            self._push_composed_data(run_id, pack_id, gpu_id, "process", process, job_id, metric_time=metric_time)

        elif (cpudata := data.pop("cpudata", None)) is not None:
            self._push_composed_data(run_id, pack_id, gpu_id, "cpudata", cpudata, job_id, metric_time=metric_time)

        elif (iodata := data.pop("iodata", None)) is not None:
            self._push_composed_data(run_id, pack_id, gpu_id, "iodata", iodata, job_id, metric_time=metric_time)

        elif (netdata := data.pop("netdata", None)) is not None:
            self._push_composed_data(run_id, pack_id, gpu_id, "netdata", netdata, job_id, metric_time=metric_time)

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
                    order=metric_time,
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
            run_id, pack_id, "walltime", end - state.start, gpu_id=gpu_id, job_id=job_id
        )

        status = "done"
        if state.early_stop:
            status = "early_stop"

        if state.error > 0:
            status = "error"

        self._push_metric(
            run_id, pack_id, "return_code", entry.data["return_code"], gpu_id=gpu_id, job_id=job_id,
            namespace=status
        )

        status_code = 1
        if status in ("early_stop", "done"):
            status_code = 0

        self._push_metric(
            run_id, pack_id, "status", status_code, gpu_id=gpu_id, job_id=job_id
        )

        self.update_pack_status(state.pack, status)
        self.states.pop(entry.tag)

        # even if the pack has ended we have other
        # packs still pushing metrics
        if len(self.states) == 0:
            self._bulk_insert()


if __name__ == "__main__":
    generate_database_sql_setup()
