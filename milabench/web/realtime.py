import os
import requests
import subprocess
from threading import Thread, Lock, Event
import json

from ..pack import Package
from ..structs import BenchLogEntry

#
#   1. Have the server register the metric receiver route
#   2. Make milabench use the `HTTPMetricPusher` logger
#

# Global variable to store socketio instance
socketio_instance = None

def set_socketio_instance(socketio):
    """Set the global socketio instance for broadcasting"""
    global socketio_instance
    socketio_instance = socketio

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
        from .constant import JOBRUNNER_LOCAL_CACHE

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
                yield self.benchentry(**entry)


def metric_receiver(app):
    from flask import request
    rebuilder_registry = {}
    process_registry = {}

    @app.route('/api/metric/<string:hostname>')
    def open_reverse_ssh(hostname: str):
        cmd = reverse_ssh_tunnel(hostname)

        nonlocal process_registry

        if process_registry.get(hostname) is None:
            proc = subprocess.Popen(cmd)
            process_registry[hostname] = proc
            return {"status": "ok", "message": "created"}

        return {"status": "ok", "message": "already exists"}

    @app.route('/api/metric/<string:hostname>', methods=['DELETE'])
    def close_reverse_ssh(hostname: str):
        nonlocal process_registry

        if (proc := process_registry.get(hostname)) is not None:
            proc.terminate()
            proc.wait()  # Wait for process to actually terminate
            del process_registry[hostname]  # Clean up registry
            return {"status": "ok", "message": "stopped"}
        return {"status": "ok", "message": "does not exist"}

    @app.route('/api/metric/<string:jr_job_id>', methods=['POST'])
    def receive_metric(jr_job_id: str):
        # We do not control when we receive the data
        # we might lose the first few messages
        nonlocal rebuilder_registry
        global socketio_instance

        lines = request.get_data(as_text=True).split("\n")

        # Forward the metric to the clients
        if socketio_instance:
            for line in lines:
                if line :=  line.strip():
                    try:
                        json_data = json.loads(line)

                        socketio_instance.emit('metric_data', {
                            'jr_job_id': jr_job_id,
                            'data': json_data,
                        })
                    except json.JSONDecodeError:
                        # If not valid JSON, still broadcast as raw data
                        socketio_instance.emit('metric_data', {
                            'jr_job_id': jr_job_id,
                            'data': None,
                            'raw_line': line
                        })

        return {}

#
# 
# milabench benchmarks -> POST to Server -> Server forward to the dashboard through websocket
# milabench benchmarks -> push to database
# 


def reverse_ssh_tunnel(hostname):
    # ssh -N -R 5000:localhost:5000 cn-d004.server.mila.quebec
    return [
        "ssh",
        "-N",
        "-R", "5000:localhost:5000",
        hostname
    ]


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