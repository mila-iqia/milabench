import os
import requests
import subprocess
from threading import Thread, Lock, Event
import json

from milabench.pack import Package
from milabench.structs import BenchLogEntry

#

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
        self.url = f"{url}/api/metric/{jr_job_id}"
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

    def __call__(self, entry):
        return self.on_event(entry)

    def on_event(self, entry: BenchLogEntry):
        with self.lock:
            d = entry.dict()
            d.pop("pack")

            if "tag" not in d:
                d["tag"] = d.tag

            try:
                self.pending_messages.append(json.dumps(d))
            except TypeError:
                self.pending_messages.append(f'{{"tag": "{d.tag}", "#unrepresentable": {str(d)} }}')

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
            batch = "\n".join(m for m in messages)
            requests.post(self.url, data=batch, timeout=5)
        except Exception as e:
            print(f"Failed to push metrics: {e}")
            with self.lock:
                self.pending_messages = messages + self.pending_messages