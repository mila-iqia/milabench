from dataclasses import dataclass
from datetime import datetime
from collections import deque
from contextlib import contextmanager
import numpy as np
import time
import traceback
import socket
import sys
import json
import os
import tempfile
import torch

from .monitor import GPUMonitor
from .helpers import resolve


_environ_save = {
    "CUDA_VISIBLE_DEVICES",
    "MILARUN_DATAROOT",
}


def get_gpu_name():
    current_device = torch.cuda.current_device()
    return torch.cuda.get_device_name(current_device)


class SimpleTimer:
    def __init__(self, sync=None):
        self.start = None
        self.end = None
        self.sync = sync
        if self.sync:
            self.sync()

    def __enter__(self):
        if self.start is not None:
            raise Exception("Cannot use a SimpleTimer twice")
        self.start = time.time()
        return self

    def __exit__(self, exc_type, value, tb):
        if self.sync:
            self.sync()
        self.end = time.time()
        self.result = self.end - self.start

    def report(self):
        return {
            "type": "milabench.experiment.SimpleTimer",
            "start": str(datetime.fromtimestamp(self.start)),
            "end": str(datetime.fromtimestamp(self.end)),
            "time": self.result,
        }


@dataclass
class Counter:
    count: int
    metrics: dict

    def add(self, n):
        self.count += n

    def set_count(self, n):
        self.count = n

    def log(self, **metrics):
        self.metrics.update(metrics)


class RateLogger:
    def __init__(self, sample_duration=30, max_count=None, sync=None):
        self.sample_duration = sample_duration
        self.current = 0
        self.count = 0
        self.results = []
        self.max_count = max_count
        self.total_count = 0
        self.sync = sync
        self.start = 0
        self.end = 0
        self.total_time = 0
        self.metrics = {}
        if self.sync:
            self.sync()

    def would_log(self, duration):
        return self.current + duration >= self.sample_duration

    def elapse_sync(self):
        if self.sync:
            start = time.time()
            self.sync()
            end = time.time()
            self.elapse(end - start, 0)

    def elapse(self, duration, count):
        self.current += duration
        self.count += count
        self.total_time += duration
        self.total_count += count
        if self.current >= self.sample_duration:
            nblocks = self.current / self.sample_duration
            unit = self.count / nblocks
            self.results.extend([unit / self.sample_duration] * int(nblocks))
            self.current = self.current % self.sample_duration
            self.count = self.count % unit
            if self.metrics:
                items = []
                if self.metrics.get("count", True):
                    l = len(str(self.max_count or 0))
                    items.append(
                        f"[{int(self.total_count):>{l}}/{self.max_count or '?'}]",
                    )
                if self.metrics.get("eta", True):
                    estimate = self.total_time * (self.max_count / self.total_count)
                    eta = estimate - self.total_time
                    items.append(
                        f"[ETA: {eta//60:.0f}m{eta%60:02.0f}s]"
                    )
                if self.metrics.get("rate", True):
                    items.append(
                        f"[{self.results[-1]:.2f} items/s]"
                    )
                for k, v in self.metrics.items():
                    if k in ("count", "eta", "rate"):
                        continue
                    items.append(f"{k}={v}")
                print(*items)

    @contextmanager
    def __call__(self, *, count=1, key=None):
        if not self.start:
            self.start = time.time()
        count = Counter(count=count, metrics=self.metrics)
        start = time.time()
        yield count
        end = time.time()
        if self.sync and self.would_log(end - start):
            # Sync only when we go over the sample_duration
            self.sync()
            end = time.time()
        self.elapse(end - start, count.count)
        self.end = time.time()

    def finalize(self):
        self.elapse_sync()
        if self.current > 0:
            self.results.append(self.count / self.current)
        self.current = 0
        self.count = 0
        return self.results

    def report(self):
        self.finalize()
        wall_time = self.end - self.start if self.start and self.end else 0
        return {
            "type": "milabench.experiment.RateLogger",
            "start": str(datetime.fromtimestamp(self.start)),
            "end": str(datetime.fromtimestamp(self.end)),
            "wall_time": wall_time,
            "time": self.total_time,
            "overhead": wall_time - self.total_time,
            "sample_duration": self.sample_duration,
            "rates": self.results,
            "metrics": {k: v for k, v in self.metrics.items() if k not in ("count", "eta", "rate")},
        }

    def done(self):
        return self.total_count >= self.max_count


class Chronos:
    def __init__(self):
        self.chronos = {}

    def create(self, name, type, **kwargs):
        if name in self.chronos:
            raise Exception(f"Chrono {name} already exists.")
        if type == "rate":
            chrono = RateLogger(**kwargs)
        elif type == "timer":
            chrono = SimpleTimer(**kwargs)
        else:
            raise Exception(f"Unknown chrono type: {type}")
        self.chronos[name] = chrono
        return chrono


class Experiment:
    def __init__(self, name, job_id, dataroot, outdir=None, monitor_gpu_usage=True):
        self.name = name
        self.job_id = job_id
        self.dataroot = dataroot
        self.outdir = outdir
        self.monitor_gpu_usage = monitor_gpu_usage
        self.tmp = outdir or tempfile.mkdtemp()
        self.chronos = Chronos()
        self.metrics = {}
        self.timings = {}
        self.results = {
            "name": self.name,
            "job_id": job_id,
            "dataroot": dataroot,
            "outdir": outdir,
        }
        self.usage = None

    def execute(self, fn):
        if self.monitor_gpu_usage:
            monitor = GPUMonitor(1)
            monitor.setDaemon(True)
            monitor.start()
        try:
            fn()
        except Exception as e:
            exc_text = traceback.format_exc()
            self.results["success"] = False
            self.results["error"] = exc_text
            print(exc_text, file=sys.stderr)
        else:
            self.results["success"] = True
            self.results["error"] = None
        if self.monitor_gpu_usage:
            monitor.stop()
            self.usage = {
                gid: {
                    k: {
                        "min": np.min(v) if v else -1,
                        "mean": np.mean(v) if v else -1,
                        "max": np.max(v) if v else -1,
                    }
                    for k, v in gdata.items()
                }
                for gid, gdata in monitor.data.items()
            }

    def __getitem__(self, key):
        return self.results[key]

    def __setitem__(self, key, value):
        self.results[key] = value

    def set_fields(self, fields):
        self.results.update(fields)

    def report(self):
        timings = {
            k: chrono.report()
            for k, chrono in self.chronos.chronos.items()
        }
        timings = {**timings, **self.timings}
        metrics = self.metrics
        for report in timings.values():
            metrics.update(report.get("metrics", {}))

        return {
            **self.results,
            "hostname": socket.gethostname(),
            "gpu": get_gpu_name(),
            "device_count": torch.cuda.device_count(),
            "environ": {k: v for k, v in os.environ.items() if k in _environ_save},
            "metrics": metrics,
            "timings": timings,
            "gpu_monitor": self.usage,
        }

    def experiment_string(self, include_failure=False):
        parts = [
            self.name,
            f"J{self.job_id}" if self.job_id else None,
            None if (not include_failure or self.results.get("success", True)) else "FAIL",
            datetime.utcnow().strftime(r'%Y%m%d-%H%M%S-%f'),
        ]
        return ".".join(x for x in parts if x)

    def write(self, out=None):
        if out is None:
            out = self.outdir

        report = self.report()
        dump = json.dumps(report, indent=4)

        if out:
            if out.endswith(".json"):
                with open(out, "w") as f:
                    print(dump, file=f)
            else:
                os.makedirs(out, exist_ok=True)
                return self._write_to_dir(report, out)
        else:
            print(dump)

    def _write_to_dir(self, report, outdir):
        filename_parts = [
            self.name,
            f"J{self.job_id}" if self.job_id else None,
            None if self.results["success"] else "FAIL",
            datetime.utcnow().strftime(r'%Y%m%d-%H%M%S-%f'),
        ]
        filename = ".".join(x for x in filename_parts if x)
        filename += ".json"
        json_report = json.dumps(report, indent=4)
        with open(os.path.join(outdir, filename), 'w') as file:
            print(json_report, file=file)
        return json_report

    def resolve_dataset(self, name):
        result = resolve(name)
        result["environment"] = {
            "root": self.dataroot
        }
        return result

    def get_dataset(self, name, *args, **kwargs):
        fn = resolve(name)
        dataset = fn(self.dataroot, *args, **kwargs)
        dataset.avail()
        return dataset

    def results_directory(self):
        rdir = os.path.join(self.tmp, self.experiment_string())
        os.makedirs(rdir, exist_ok=True)
        return rdir

    def time(self, name):
        return self.chronos.create(name, type="timer")
