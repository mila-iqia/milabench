import os
import requests
import subprocess
from threading import Thread, Lock, Event
import json
from flask import request

from ..pack import Package
from ..structs import BenchLogEntry
from .slurm import JOBRUNNER_LOCAL_CACHE

#
#   1. Have the server register the metric receiver route
#   2. Make milabench use the `HTTPMetricPusher` logger
#   

class BenchEntryRebuilder:
    """Rebuild the benchentry from a stream of data entry"""
    
    event_order = [
        "meta",
        "config",
        "start",
        "data",
        "stop",
        "end",
    ]

    def __init__(self, jr_job_id=None) -> None:
        self.jr_job_id = jr_job_id
        self.pack = None
        self.meta = None

    def benchentry(self, tag=None, **kwargs) -> BenchLogEntry:
        return BenchLogEntry(self.pack, **kwargs)

    def __call__(self, entry):
        match entry["event"]:
            case "meta":
                self.meta = entry
                yield None

            case "config":
                # Change the path where we are saving things
                entry["data"]["dirs"]["runs"] = os.path.join(JOBRUNNER_LOCAL_CACHE, self.jr_job_id)
            
                self.pack = Package(config=entry["data"])
                if self.meta is not None:
                    yield self.benchentry(**self.meta)
                    self.meta = None
                yield self.benchentry(**entry)

            case _:
                yield self.benchentry(*entry)


def metric_receiver(app, receiver_factory=lambda x: BenchEntryRebuilder(x)):
    registry = {}

    @app.route('/api/metric/<string:jr_job_id>', methods=['POST'])
    def receive_metric(jr_job_id: str):
        nonlocal registry

        lines = request.get_data(as_text=True).split("\n")

        receiver = registry.setdefault(jr_job_id, receiver_factory(jr_job_id))

        # NOTE: we lose access to entry.pack here
        if receiver is not None:

            for line in lines:
                line = json.load(line)

                for entry in receiver(line):
                    yield entry


def reverse_ssh_tunnel(hostname):
    ssh_process = subprocess.Popen([
        "ssh",
        "-N", 
        "-R", "5000:localhost:5000",
        hostname
    ])
    return ssh_process


class HTTPMetricPusher:
    """Push milabench metrics to a webserver
    
    Notes
    -----

    You will need a reverse SSH Tunnel

        ssh -R 9000:localhost:5000 compute-node
    """

    def __init__(self, url, jr_job_id=os.getenv("JR_JOB_ID"), interval=1.0) -> None:
        assert jr_job_id is not None

        self.jr_job_id = jr_job_id
        self.url = f"{self.url}/api/metric/{jr_job_id}"
        self.lock = Lock()
        self.pending_messages = []

        self.interval = interval
        self._stop_event = Event()
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()
    
    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self._stop_event.set()
        self._thread.join()
        self.push()

    def on_event(self, entry: BenchLogEntry):
        with self.lock:
            d = entry.dict()
            d.pop("pack")

            try:
                self.pending_messages.append(json.dumps(d))
            except TypeError:
                self.pending_messages.append(f'{{"#unrepresentable": {str(d)} }}')
    
    def _loop(self):
        while not self._stop_event.wait(self.interval):
            self.push()

    def push(self):
        with self.lock:
            if not self.pending_messages:
                return

            messages = self.pending_messages
            self.pending_messages = []

        try:
            batch = "\n".join(m.json() for m in messages)
            requests.post(self.url, data=batch, timeout=5)
        except Exception as e:
            print(f"Failed to push metrics: {e}")
            with self.lock:
                self.pending_messages = messages + self.pending_messages