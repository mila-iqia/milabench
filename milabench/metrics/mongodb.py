from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass

from ..structs import BenchLogEntry


import pymongo


class Run:
    pass


class Pack:
    pass


class Metric:
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


class MongoDB:
    def __init__(self, uri, database="milabench") -> None:
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[database]

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
            self.update_pack_status(state.pack["_id"], "interrupted")

        status = "done"
        if len(self.states) > 0:
            status = "interrupted"

        if self.run is not None:
            self.update_run_status(status)

        self.states = defaultdict(PackState)

    def update_run_status(self, status):
        self._run.update_one(
            {"_id": self.run["_id"]},
            {
                "$set": {
                    "status": status,
                }
            },
        )

    def update_pack_status(self, pack_id, status):
        self._pack.update_one(
            {"_id": pack_id},
            {
                "$set": {
                    "status": status,
                }
            },
        )

    @property
    def _run(self):
        return self.db["run"]

    @property
    def _pack(self):
        return self.db["pack"]

    @property
    def _metrics(self):
        return self.db["metrics"]

    def on_event(self, entry: BenchLogEntry):
        method = getattr(self, f"on_{entry.event}", None)

        if method is not None:
            method(entry)

    def on_new_run(self, entry):
        self.run = {
            "name": entry.pack.config["run_name"],
            "namespace": None,
            "created_time": datetime.utcnow(),
            "meta": entry.data,
            "status": "running",
        }
        result = self._run.insert_one(self.run)
        self.run["_id"] = result.inserted_id

    def on_new_pack(self, entry):
        state = self.pack_state(entry)
        state.pack = {
            "exec_id": self.run["_id"],
            "created_time": datetime.utcnow(),
            "name": entry.pack.config["name"],
            "tag": entry.tag,
            "config": entry.pack.config,
        }
        result = self._pack.insert_one(state.pack)
        state.pack["_id"] = result.inserted_id

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
            pymongo.InsertOne(
                {
                    "exec_id": run_id,
                    "pack_id": pack_id,
                    "name": name,
                    "value": value,
                    "gpu_id": gpu_id,
                }
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

        run_id = self.run["_id"]
        pack_id = state.pack["_id"]

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

        self._metrics.bulk_write(self.pending_metrics)
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

        self.update_pack_status(state.pack["_id"], status)
        self.states.pop(entry.tag)

        # even if the pack has ended we have other
        # packs still pushing metrics
        if len(self.states) == 0:
            self._bulk_insert()
