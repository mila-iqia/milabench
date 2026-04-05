"""Generic milabench datafolder processor, it also supports extracting data from the run and bench name

The MetricExtractor generate a flat structure so we can directly process the results using pandas
"""

import multiprocessing as mp
import queue
import traceback
from datetime import datetime
import json
import re
from collections import defaultdict
import glob
from pathlib import Path
import threading
import hashlib
from dataclasses import dataclass

_bench_tags = [
    "concurrency=conc([0-9]+)",
    "max_context=mxctx([0-9]+)",
    "max_batch_token=mxbt([0-9]+)",
    "worker=w([0-9]+)",
    "multiple=m([0-9]+)",
    "batch_power=bp([0-9]+)",
    "capacity=c([0-9]+(Go)?)",
    "device=D([0-9]+)",
]

_run_tags = [
    "clock=g([0-9]+)",
    "power=p([0-9]+)",
    "observation=o([0-9]+)",
]


def make_tags(tags_def):
    tags = dict()
    for tag in tags_def:
        name, regex = tag.split("=")
        tags[name] = re.compile(regex)
    return tags

bench_tags = make_tags(_bench_tags)


run_tags = make_tags(_run_tags)


def extract_tags(name, tags):
    for tag, pat in tags.items():
        if m := pat.search(name):
            value = m.group(1)
            yield tag, value


def workitem_readfolder(folder, meta):
    return {"action": "folder", "value": str(folder), "meta": meta}


def workitem_readfile(file, meta):
    return {"action": "file", "value": str(file), "meta": meta}


class Worker:
    def __init__(self, worker_pack):
        work_queue, result_queue, error_queue, active, pending, done, results = worker_pack
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.error_queue = error_queue
        self.active = active
        self.jobs = pending
        self.done = done
        self.results = results

    def inc_done(self):
        with self.done.get_lock():
            self.done.value += 1

    def inc_jobs(self):
        with self.jobs.get_lock():
            self.jobs.value += 1

    def push_work(self, work):
        self.inc_jobs()
        self.work_queue.put(work)

    def inc_results(self):
        with self.results.get_lock():
            self.results.value += 1

    def push_result(self, result):
        self.inc_results()
        self.result_queue.put(result)

    def run(self):
        while self.active.is_set():
            try:
                task = self.work_queue.get(timeout=0.1)

                try:
                    self(task)
                finally:
                    self.inc_done()

            except queue.Empty:
                continue

            except Exception as err:
                traceback.print_exc()

    def __call__(self, task):
        pass


def worker_fn(i, cls, worker_pack, args, kwargs):
    worker = cls(worker_pack, *args, **kwargs)
    worker.run()


class _Value:
    def __init__(self, type: str, value) -> None:
        self.value = value
        self.lock = threading.RLock()

    def get_lock(self):
        return self.lock


class Multiprocessing:
    Queue = mp.Queue
    Event = mp.Event
    Value = mp.Value
    Worker = mp.Process


class Threading:
    Queue = queue.Queue
    Event = threading.Event
    Value = _Value
    Worker = threading.Thread


def nice_cpu_count():
    return max(1, mp.cpu_count() - 2)


class DataProcessor:
    """We need to process per group of bench, processing everything in parallel takes too much memory"""
    def __init__(self, worker_cls, *args, worker_count=nice_cpu_count(), backend=Multiprocessing, **kwargs):
        self.work_queue = backend.Queue(maxsize=0)
        self.result_queue = backend.Queue()
        self.error_queue = backend.Queue()
        self.active = backend.Event()
        self.active.set()

        self.jobs = backend.Value("i", 0)
        self.done = backend.Value("i", 0)
        self.results = backend.Value("i", 0)
        self.retrieved = 0

        worker_pack = self.work_queue, self.result_queue, self.error_queue, self.active, self.jobs, self.done, self.results
        args = (worker_cls, worker_pack, args, kwargs)
        self.workers = [backend.Worker(target=worker_fn, args=(i,) + args) for i in range(worker_count)]

        for w in self.workers:
            w.start()

    def is_finished(self):
        with self.jobs.get_lock():
            jobs = self.jobs.value

        with self.done.get_lock():
            done = self.done.value

        with self.results.get_lock():
            results = self.results.value

        return jobs == done and self.result_queue.empty() and results == self.retrieved

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.active.clear()

        unfinished = 0
        ignored = 0

        while not self.is_finished():
            # remove work first
            try:
                self.work_queue.get(timeout=0)

                with self.done.get_lock():
                    self.done.value += 1

                unfinished += 1
            except queue.Empty:
                pass

            #
            #   remove results
            #
            try:
                r = self.result_queue.get(timeout=0)
                self.retrieved += 1
                ignored += 1
            except queue.Empty:
                pass

        for i, w in enumerate(self.workers):
            w.join()

        if unfinished > 0:
            print(f"{unfinished} work item aborted")

        if ignored > 0:
            print(f"{ignored} results ignored")

    def inc_jobs(self):
        with self.jobs.get_lock():
            self.jobs.value += 1

    def push_work(self, work):
        self.inc_jobs()
        self.work_queue.put(work)

    def __call__(self, folderpat, meta=None):
        if meta is None:
            meta = {}

        #
        # To start up the worker the main train do the first work
        # which will schedule more work
        #
        found = 0
        for folder in glob.iglob(folderpat, recursive=False):
            folder = Path(folder)
            if folder.is_file():
                meta = extract_meta_from_run_folder(folder.parent, meta)

            readfolder(self, folder, meta)
            found += 1

        if found == 0:
            raise RuntimeError(f"folder not found {folderpat}")

        while not self.is_finished():
            try:
                item = self.result_queue.get(timeout=0.1)
                self.retrieved += 1
                yield item
            except queue.Empty:
                continue

        self.active.clear()

        for i, w in enumerate(self.workers):
            w.join()


def flatten_values(payload, namespace=None):
    if namespace is None:
        namespace = tuple()

    if isinstance(payload, list):
        for i, val in enumerate(payload):
            yield from flatten_values(val, namespace + (str(i),))

    elif isinstance(payload, dict):
        for k, v in payload.items():
            nspace = namespace + (str(k),)
            yield from flatten_values(v, nspace)

    else:
        yield ".".join(namespace), payload


def insert_path(job_meta, name, entry):
    i = 0
    while True:
        p = f"{name}{i}"

        if val := job_meta.get(p):
            if val == entry.name:
                return
            i += 1
        else:
            job_meta[p] = entry.name

def extract_meta_from_run_folder(entry, meta):
    """Keep the folder path to the data file as it might be information"""
    run_meta = dict(extract_tags(entry.name, run_tags))

    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', entry.name)
    if match:
        run_meta["date"] = datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")

    job_meta = {**meta, **run_meta}

    insert_path(job_meta, "p", entry)

    return job_meta


def readfolder(worker, entry, meta):
    """Extract meta information from the folder/file name and push the item as work to be futher processed"""
    if entry.is_dir():
        job_meta = extract_meta_from_run_folder(entry, meta)
        worker.push_work(workitem_readfolder(entry, job_meta))

    elif entry.is_file():
        frags = entry.name.split(".", maxsplit=1)
        bench_meta = dict(extract_tags(entry.name, bench_tags))
        job_meta = {"bench": frags[0], **meta, **bench_meta}
        worker.push_work(workitem_readfile(entry, job_meta))


DEFAULT_IGNORED = tuple(["$queued", "progress.0", "progress.1", "progress"])
ACCEPT_FILE = lambda file, meta: True



class EventProcessor(Worker):
    """Base milabench event processor"""
    def __init__(self, worker_pack, *args, accept_file=ACCEPT_FILE, ignored=DEFAULT_IGNORED, **kwargs):
        super().__init__(worker_pack)
        self.ignored = ignored
        self.accept_file = accept_file

    def __call__(self, task):
        match task["action"]:
            case "folder":
                self.readfolder(task["value"], task["meta"])
            case "file":
                self.readfile(task["value"], task["meta"])
            case _:
                raise RuntimeError("Unknown action")

    def readfolder(self, folder, meta):
        """Push content of folder as new work item"""
        for entry in Path(folder).iterdir():
            readfolder(self, entry, meta)

    def readfile(self, file, meta):
        """Read data files produced by milabench and process them line by line"""
        if file.endswith(".data") and self.accept_file(file, meta):
            with open(file, "r") as fp:

                run_id = hashlib.sha256(file.encode()).hexdigest()[:8]

                # Build a shared metadata, that events can modify
                file_meta = {"run_id": run_id, **meta}

                self.file_start(file)

                for line in fp.readlines():
                    metric_line = json.loads(line)

                    if (stop := self.processline(metric_line, file_meta)) is True:
                        break

                self.file_end(file)

    def file_start(self, filename):
        pass

    def file_end(self, filename):
        pass

    def processline(self, line, meta):
        """Dispatch events"""
        match line["event"]:
            case "config":
                return self.config(line, meta)

            case "meta":
                return self.meta(line, meta)

            case "start":
                return self.start(line, meta)

            case "phase":
                return self.phase(line, meta)

            # data
            case "data":
                return self.data(line["data"], meta)

            case "line":
                return self.line(line["data"], line["pipe"], meta)

            case "overseer_error":
                return self.overseer_error(line, meta)

            case "error":
                return self.error(line, meta)

            case "end":
                return self.end(line, meta)

            case "message":
                return self.message(line, meta)

            case "format_error":
                return self.format_error(line, meta)

            case "stop":
                return self.stop(line, meta)

            case event:
                print(f"Unhandled event {event} {line}")

    def config(self, event, meta):
        pass

    def meta(self, event, meta):
        pass

    def start(self, event, meta):
        pass

    def phase(self, event, meta):
        pass

    def data(self, data, meta):
        pass

    def line(self, line, pipe, meta):
        pass

    def overseer_error(self, event, meta):
        pass

    def error(self, event, meta):
        pass

    def end(self, event, meta):
        pass

    def message(self, message, meta):
        pass

    def format_error(self, event, meta):
        pass

    def stop(self, event, meta):
        pass


@dataclass
class EventTracking:
    start_time: float = None
    rc_code: int = None
    stop: bool = False
    error: bool = False
    gpu_count: int  = 1

    def success(self):
        if self.stop:
            return True

        if self.rc_code != 0:
            return False

        return True


class MetricExtractor(EventProcessor):
    def __init__(self, worker_pack, *args, accept_file=ACCEPT_FILE, ignored=DEFAULT_IGNORED, **kwargs):
        super().__init__(worker_pack, accept_file=accept_file, ignored=ignored)

        self.should_simplify_gpudata = True
        self.start_event = {}

    @staticmethod
    def makekey(meta):
        frags = []
        for k, v in sorted(meta.items(), key=lambda item: item[0]):
            frags.append(v)
        return tuple(v)

    def start(self, event, meta):
        if start_time := event["data"].get("time"):
            key = MetricExtractor.makekey(meta)
            tracking = self.start_event.setdefault(key, EventTracking())
            tracking.start_time = start_time

    def end(self, event, meta):
        key = MetricExtractor.makekey(meta)
        data = event["data"]

        if tracking := self.start_event.get(key):
            end_time = data["time"]
            self.push_result({**meta, "metric": "elapsed", "value": end_time- tracking.start_time, "unit": "s", "time": end_time})

            tracking.rc_code = data["return_code"]
            self.push_result({**meta, "metric": "success", "value": int(tracking.success()), "time": end_time})

            self.push_result({**meta, "metric": "ngpu", "value": int(tracking.gpu_count), "time": end_time})

    def stop(self, event, meta):
        key = MetricExtractor.makekey(meta)
        if tracking := self.start_event.get(key):
            tracking.stop = True

    def error(self, event, meta):
        print(event)

    def simplify_gpudata(self, meta, gpudata, time, task, units, unit):
        key = MetricExtractor.makekey(meta)
        if tracking := self.start_event.get(key):
            tracking.ngpu = len(gpudata)

        if "device" in meta:
            # Single device
            assert len(gpudata) <= 1, f"{gpudata}"

            for device_id, metrics in gpudata.items():
                for k, v in metrics.items():
                    if isinstance(v, list):
                        # For memory [0, 1]
                        for i, item in enumerate(v):
                            metric = {
                                "metric": f"gpudata.{k}.{i}",
                                "value": item,
                                "time": time,
                                "unit": units or unit,
                                "task": task,
                                "count": 1,
                                **meta
                            }
                            self.push_result(metric)
                    else:
                        # Standard
                        metric = {
                            "metric": f"gpudata.{k}",
                            "value": v,
                            "time": time,
                            "unit": units or unit,
                            "task": task,
                            "count": 1,
                            **meta
                        }
                        self.push_result(metric)
        else:
            # for multiGPU workloads
            device_sum = defaultdict(float)
            device_cnt = defaultdict(float)

            for device_id, metrics in gpudata.items():
                for k, v in metrics.items():
                    if isinstance(v, list):
                        for i, item in enumerate(v):
                            device_sum[f"{k}.{i}"] += item
                            device_cnt[f"{k}.{i}"] += 1
                    else:
                        device_sum[k] += v
                        device_cnt[k] += 1


            for k, v in device_cnt.items():
                sum = device_sum[k]
                cnt = device_cnt[k]

                metric = {
                    "metric": f"gpudata.{k}",
                    "value": sum/cnt,
                    "time": time,
                    "unit": units or unit,
                    "task": task,
                    "count": cnt,
                    **meta
                }
                self.push_result(metric)

    def data(self, data, meta):
        unit = data.pop("unit", None)
        units = data.pop("units", None)
        time = data.pop("time", None)
        task = data.pop("task", None)
        batch_id = data.pop("batch_id", None)

        if (gpudata := data.pop("gpudata", {})) and self.should_simplify_gpudata:
            self.simplify_gpudata(meta, gpudata, time, task, units, unit)

        for k, v in flatten_values(data, tuple([])):
            if k in self.ignored:
                continue

            # if time is None:
            #    print(data, "missing time", meta)

            metric = {
                "metric": k,
                "value": v,
                "time": time,
                "unit": units or unit,
                "task": task,
                **meta
            }

            if batch_id is not None:
                metric["batch_id"] = batch_id

            self.push_result(metric)


class MetaExtractor(EventProcessor):
    def meta(self, event, meta):
        self.push_result(event.pop("data"))
        return True


class ConfigExtractor(EventProcessor):
    def config(self, event, meta):
        self.push_result(event.pop("data"))
        return True


class LogExtractor(EventProcessor):
    def error(self, event, meta):
        self.push_result({**event, **meta})

    def line(self, line, pipe, meta):
        self.push_result({"text": line, "pipe": pipe, **meta})

    def message(self, event, meta):
        self.push_result({**event["data"], **meta})


class BenchmarkStatusExtractor(EventProcessor):
    def __init__(self, worker_pack, *args, accept_file=ACCEPT_FILE, ignored=DEFAULT_IGNORED, **kwargs):
        super().__init__(worker_pack, *args, accept_file=accept_file, ignored=ignored, **kwargs)
        self.start_event = {}

    def file_end(self, filename):
        key, tracking = self.start_event.popitem()

        if tracking.success():
            self.push_result({"status": "success"})
        else:
            self.push_result({"status": "failed"})

    def start(self, event, meta):
        if start_time := event["data"].get("time"):
            key = MetricExtractor.makekey(meta)
            tracking = self.start_event.setdefault(key, EventTracking())
            tracking.start_time = start_time

    def end(self, event, meta):
        key = MetricExtractor.makekey(meta)
        data = event["data"]

        if tracking := self.start_event.get(key):
            tracking.rc_code = data["return_code"]

    def stop(self, event, meta):
        key = MetricExtractor.makekey(meta)
        if tracking := self.start_event.get(key):
            tracking.stop = True

    def error(self, event, meta):
        key = MetricExtractor.makekey(meta)
        if tracking := self.start_event.get(key):
            tracking.error = True


def fetch_benchmark_status(name):
    with DataProcessor(LogExtractor, backend=Threading) as proc:
            for i, item in enumerate(proc(name)):
                return item["status"]



def extract_milabench_metrics(folder):
    """Milabench metric extractor.

    The metric stream is flatten for convenience.
    meta data information is extracted from the bench name and the run name.
    The file path information is kept as p0/p1/p2/p3/etc...

    We can use this function to inspect and analyse results.

    Arguments
    ---------
    folder: str
        it can a path to a

        1. folder containing multiple runs (./sxm_runs)
        2. folder of a single run          (./sxm_runs/p600.o500.2025-12-26_07-20-23)
        3. a file of a single benchmark    (./sxm_runs/p600.o500.2025-12-26_07-20-23/convnext_large-tf32-fp16.D0.data)
        4. a glob pattern                  (./sxm_runs/p600.o500.2025-12-26_07-20-23/convnext_large-tf32-fp16.*)
                                           (./sxm_runs/p600.o*)

    Output
    ------

        {'metric': 'gpudata.memory.0'   , 'value': 47281.3125, 'time': 1766765836.8106523, 'unit': None, 'task': 'main', 'bench': 'convnext_large-fp32', 'clock': '1440', 'observation': '350', 'p0': 'g1440.o350.2025-12-26_09-59-43', 'device': '2'}
        {'metric': 'gpudata.memory.1'   , 'value': 143771.0  , 'time': 1766765836.8106523, 'unit': None, 'task': 'main', 'bench': 'convnext_large-fp32', 'clock': '1440', 'observation': '350', 'p0': 'g1440.o350.2025-12-26_09-59-43', 'device': '2'}
        {'metric': 'gpudata.load'       , 'value': 1.0       , 'time': 1766765836.8106523, 'unit': None, 'task': 'main', 'bench': 'convnext_large-fp32', 'clock': '1440', 'observation': '350', 'p0': 'g1440.o350.2025-12-26_09-59-43', 'device': '2'}
        {'metric': 'gpudata.temperature', 'value': 65.0      , 'time': 1766765836.8106523, 'unit': None, 'task': 'main', 'bench': 'convnext_large-fp32', 'clock': '1440', 'observation': '350', 'p0': 'g1440.o350.2025-12-26_09-59-43', 'device': '2'}
        {'metric': 'gpudata.power'      , 'value': 442.059   , 'time': 1766765836.8106523, 'unit': None, 'task': 'main', 'bench': 'convnext_large-fp32', 'clock': '1440', 'observation': '350', 'p0': 'g1440.o350.2025-12-26_09-59-43', 'device': '2'}

    """
    with DataProcessor(MetricExtractor, worker_count=mp.cpu_count(), backend=Multiprocessing) as proc:
        for item in proc(folder):
            yield item


def augment_energy_estimator(metrics, force_sort=True):
    previous = {}

    if force_sort:
        metrics = sorted(metrics, key=lambda item: item.get("time") or 0)

    for metric in metrics:
        if metric["metric"] == "gpudata.power":

            bench = metric["bench"]
            device = metric.get("device", -1)
            p0 = metric["p0"]

            key = (bench, device, p0)

            prev = previous.get(key)
            if prev is not None and prev.get("time") is not None:
                energy_p = prev["value"]
                energy_n = metric["value"]
                elapsed = metric["time"] - prev["time"]

                # Number of GPUs
                count = metric["count"]

                # Riemann sum using midpoint rule
                energy_spent = elapsed * count * (energy_p + energy_n) / 2

                # push the new metric measurements
                newmetric = {**metric}
                newmetric["metric"] = "energy"
                newmetric["value"] = energy_spent
                yield newmetric

            previous[key] = metric

        yield metric


def aggregate(metrics):
    results = {}

    for metric in metrics:

        bench = metric["bench"]
        device = metric.get("device", -1)
        p0 = metric["p0"]

        key = (bench, device, p0)

        data = results.setdefault(key, {})
        data.setdefault(metric["metric"], []).append(metric["value"])

    return results


def accumulate_per_device(aggregated_metric, acc_fun):
    results = {}

    for (bench, device, _), accumulated_metrics in aggregated_metric.items():
        bench_data = results.setdefault((bench, _), {})

        for metric, values in accumulated_metrics.items():
            fun = acc_fun.get(metric)

            if fun is None:
                continue

            per_device = bench_data.setdefault(metric, {})
            per_device[device] = fun(values)

    return results


def accumulate_per_bench(accumulated_metrics, acc_fun):
    results = {}

    for (bench, _), metrics in accumulated_metrics.items():
        bench_data = results.setdefault((bench, _), {})

        for metric, per_device in metrics.items():
            fun = acc_fun.get(metric)

            if fun is None:
                continue

            devices = []
            for d, value in per_device.items():
                devices.append(value)

            if isinstance(fun, dict):
                for k, v in fun.items():
                    bench_data[k] = v(devices)
            else:
                bench_data[metric] = fun(devices)

    for (bench, p0), values in results.items():
        for metric, value in values.items():
            yield {"bench": bench, "p0": p0, "metric": metric, "value": value}


def compute_global_score(metrics, weights, default_weight=0):
    import numpy as np

    sum_score = 0
    total_weight = 0

    for _, v in weights.items():
        total_weight += v.get("weight", default_weight)

    for metric in metrics:

        if metric["metric"] == "score":
            weight = weights.get(metric["bench"]).get("weight", default_weight)

            score = metric.get("value", 0)
            sum_score += np.log(score + 1) * weight

    return np.exp(sum_score / total_weight)



def multi_run_report():

    #
    selected = (
        # "vllm-dense-physics-gpus",
        # "vllm-moe-code-gpus",
        # "whisper-transcribe-single",
        "txt-to-image-gpus",
        # "llm-chat-completion",

        # "vllm-sweep-conc512-mxbt4096-moe",
        # "vllm-sweep-conc64-mxbt4096-moe",
        # "vllm-sweep-conc8-mxbt4096-moe",

        # "vllm-sweep-dense-conc512",
        # "vllm-sweep-dense-conc64",
        # "vllm-sweep-dense-conc8",
        # "llm-lora-mp-gpus",

        # "fp8",
        # "bf16",

        # "bert-tf32-fp16",
        # "diffusion-gpus",
        # "resnet152-ddp-gpus"
        # "llm-lora-ddp-gpus",

        # "llm-lora-mp-gpus",
    )

    # selected = ("fp8",) # "fp8")

    # hgx
    # sxm
    # nvl
    # wvl

    # effect on the higher clock under same power limit
    # effect of same clock with higher power limit
    # then same clock same power limit as base line
    # + max max comme comparaison

    # , '480', '420', '360', '300'

    powers = ('700', '600')
    p1s = ('nvl', 'wvl', 'hgx', 'sxm')
    clocks = ('1980', '1785')
    obss = ("350",)

    def accept_file(file, meta):

        bench = meta["bench"]
        p1 = meta["p1"]
        power = meta.get("power")
        clock = meta.get("clock", "1785")
        obs = meta.get("observation")

        r = (
            bench in selected and
            power in powers and
            p1 in p1s and
            clock in clocks and
            obs in obss
        )

        if r:
            print(meta)

        return r


        if bench in selected:
            print(meta)

        return False
        return bench in selected


    p = "/home/delaunap/workspace/hypertech"

    data = []
    with DataProcessor(MetricExtractor, accept_file=accept_file, backend=Threading) as proc:
        for event in proc(p):
            if event["metric"] in ("rate", "gpudata.temperature", "gpudata.power"):
                data.append(event)

    import pandas as pd

    df = pd.DataFrame(data)

    print(list(df.columns))
    # df = df[df[metric_cols].count(axis=1) >= 10]

    # DROP the run without enough observation
    run_counts = df.groupby('run_id').size()
    valid_runs = run_counts[run_counts >= 10].index
    df = df[df['run_id'].isin(valid_runs)]

    # Use latest run
    latest_runs = (
        df
        .sort_values('date', ascending=False)
        .drop_duplicates(subset=['bench', 'p1', 'clock', 'power', 'observation'], keep='first')
        ['run_id']
    )

    df = df[df['run_id'].isin(latest_runs)]
    df["power"] = df["power"].fillna(600)
    df["clock"] = df["clock"].fillna(1785)

    df["power_clock"] = df["power"].astype(str) + " - " + df["clock"].astype(str)

    df.to_csv("timeseries_raw.csv", index=False)

    # df["time_norm"] = df["time"] - df.groupby(["run_id", "metric"])["time"].transform("min")
    df["time_norm"] = df["time"] - df.groupby(["run_id"])["time"].transform("min")


    print(df["metric"].unique())

    rate_df = (
        df[df["metric"] == "rate"]
        .sort_values(["run_id", "time_norm"])
        .copy()
    )

    rate_df["dvalue_dt"] = (
        rate_df.groupby("run_id")["value"].diff() /
        rate_df.groupby("run_id")["time_norm"].diff()
    )

    rate_df["dvalue_dt_smooth"] = (
    rate_df.groupby("run_id")["dvalue_dt"]
            .transform(lambda x: x.rolling(10, center=True).mean())
    )

    rate_df = rate_df[["run_id", "time_norm", "dvalue_dt_smooth"]].copy()
    df_merged = df.merge(
        rate_df,
        on=["run_id", "time_norm"],
        how="left"  # keeps all original rows, NaN if not "rate"
    )

    # bin_size = 1  # 100ms
    # df["time_bin"] = (df["time_norm"] / bin_size).round() * bin_size
    # df = (
    #     df
    #     .groupby(["time_bin", "power", "clock", "power_clock", "p1", "bench", "metric", "run_id"])
    #     .agg({"value": "mean"})   # or sum / max / median
    #     .reset_index()
    # )

    # t_min, t_max = (
    #     df.loc[df["metric"] == "rate", "time_bin"]
    #     .agg(["min", "max"])
    # )

    # df = df[
    #     (df["time_bin"] >= t_min) &
    #     (df["time_bin"] <= t_max)
    # ]

    machine_map = {"hgx": "SXM", "sxm": "SXM", "wvl": "PCIe", "nvl": "PCIe"}
    cooling_map = {"hgx": "immersion", "sxm": "air", "wvl": "immersion", "nvl": "air"}

    df["machine"] = df["p1"].map(machine_map)
    df["cooling"] = df["p1"].map(cooling_map)

    df["power_clock"] = df["power"].astype(str) + " - " + df["clock"].astype(str) + " - " + df["machine"].astype(str)


    df.to_csv("timeseries.csv", index=False)
    xaxis = "time_norm"

    import altair as alt

    def power_over_time():
        power_over_time = df
        power_over_time = power_over_time[power_over_time["metric"] == "gpudata.power"]
        chart = alt.Chart(power_over_time).mark_line().encode(
            x=alt.X(f"{xaxis}:Q", title="Time (s)"),
            y=alt.Y("value:Q", title="Power (W)", scale=alt.Scale(zero=False)),
            color=alt.Color("power_clock:N", title="Power"),
            # strokeDash=alt.StrokeDash("p1:N", title="Machine")
        ).properties(
            width=500,
            height=500
        ).facet(
            column=alt.Column("bench:N", title="Bench"),
            row=alt.Row("cooling:N", title="Cooling")
        ).resolve_scale(
            x='independent',
            y='independent'
        )
        chart.save("power_evol.png")
        return chart


    def temp_overtime():
        temperature_over_time = df
        temperature_over_time = temperature_over_time[temperature_over_time["metric"] == "gpudata.temperature"]
        chart = alt.Chart(temperature_over_time).mark_line().encode(
            x=alt.X(f"{xaxis}:Q", title="Time (s)"),
            y=alt.Y("value:Q", title="Temperature (C)", scale=alt.Scale(zero=False)),
            color=alt.Color("power_clock:N", title="Power"),
            # strokeDash=alt.StrokeDash()
        ).properties(
            width=500,
            height=500
        ).facet(
            column=alt.Column("bench:N", title="Bench"),
            row=alt.Row("cooling:N", title="Cooling")
        ).resolve_scale(
            x='independent',
            y='independent'
        )
        chart.save("temperature_over_time.png")
        return chart


    def perf_overtime():
        perf = df
        perf = perf[perf["metric"] == "rate"]

        chart = alt.Chart(perf).mark_line().encode(
            x=alt.X(f"{xaxis}:Q", title="Time (s)"),
            y=alt.Y("value:Q", title="Perf (item/s)", scale=alt.Scale(zero=False)),
            color=alt.Color("power_clock:N", title="Power"),
            # strokeDash=alt.StrokeDash()
        ).properties(
            width=500,
            height=500
        ).facet(
            column=alt.Column("bench:N", title="Bench"),
            row=alt.Row("cooling:N", title="Cooling")
        ).resolve_scale(
            x='independent',
            y='independent'
        )

        chart.save("pref_over_time.png")
        return chart

    power_ot = power_over_time()
    temp_ot = temp_overtime()
    perf_ot = perf_overtime()

    (power_ot | temp_ot | perf_ot).configure(
            axis=alt.AxisConfig(
                labelFontSize=16,
                titleFontSize=18
            ),
            legend=alt.LegendConfig(
                labelFontSize=16,
                titleFontSize=18
            )
        ).configure_header(
            titleFontSize=20,   # controls "Bench"
            labelFontSize=18    # controls each facet value
        ).save("all.png")


    # max_per_bench = (
    #     df[df["metric"].isin(["gpudata.power", "gpudata.temperature", "rate"])]
    #     .groupby(["bench", "p1", "power"])["value"]
    #     .median()
    #     .reset_index()
    # )
    # print(max_per_bench)

    # pd.pivot()

    # p = "/home/delaunao/workspace/benchdevenv/projects/hypertec/sxm_runs/"
    # p = "/home/delaunao/workspace/benchdevenv/projects/hypertec/sxm_runs/p600.o500.2025-12-26_07-20-23"
    # # p = "/home/delaunao/workspace/benchdevenv/projects/hypertec/sxm_runs/p600.o*"
    # # p = "/home/delaunao/workspace/benchdevenv/projects/hypertec/sxm_runs/p600.o500.2025-12-26_07-20-23/convnext_large-tf32-fp16.D0.data"
    # # p = "/home/delaunao/workspace/benchdevenv/projects/hypertec/nvl_runs/g1440.o350.2025-12-26_09-59-43/convnext_large-fp32.D*"
    # p = "/home/delaunap/work/milabench_dev/data/A100_mn_run_2b90373c/runs/fafuvegu.2025-10-16_01:37:14.739584"

    # from collections import defaultdict

    # metrics = defaultdict(int)
    # total = 0

    # with DataProcessor(LogExtractor, backend=Threading) as proc:
    #     for i, item in enumerate(proc(p)):
    #         # metrics[item["metric"]] += 1
    #         total += 1
    #         print(item)


    # for k, v in metrics.items():
    #     print(f"{k:>30}: {v:6d}    {v/total:5.2%}")



def single_run_check():
    import pandas as pd 
    import altair as alt

    p = "/opt/milabench/runs/sepaboda.2026-04-01_19-03-47"

    data = []
    with DataProcessor(MetricExtractor, accept_file=lambda *args: True, backend=Threading) as proc:
        for event in proc(p):
            data.append(event)
    
    df = pd.DataFrame(data)
    df.to_csv("check.csv")

    print(list(df["metric"].unique()))

    # Convert epoch time to datetime
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")

    # Normalize time to seconds from start
    df["elapsed_s"] = df["time"] - df["time"].min()
    
    metric_ids = {m: i for i, m in enumerate(df["metric"].unique())}
    df["metric_y"] = df["metric"].map(metric_ids)

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("time:Q", title="Elapsed Time (s)"),
            y=alt.Y("metric_y:Q", title="Value"),
            color=alt.Color("metric:N", title="Metric"),
            # tooltip=["metric", "value", "elapsed_s", "timestamp:T"],
        )
        .properties(width=800, height=400, title="GPU Metrics Over Time")
        .resolve_scale(y="independent")
    )

    chart.save("gpu_metrics.png")



def single_run_check():
    p = "/opt/milabench/runs/sepaboda.2026-04-01_19-03-47/txt-to-image-single.D0.data"

    rates = []
    with open(p, "r") as fp:
        for line in fp.readlines():
            line = json.loads(line)
            if "rate" in line["data"]:
                rates.append(line["data"])

    import pandas as pd 
    import altair as alt

    df = pd.DataFrame(rates)

    t0 = df["schedule_time"].min()
    df["start_s"] = df["schedule_time"] # - t0
    df["end_s"] = df["start_s"] + df["elapsed"]
    df["batch_idx"] = df.index

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("start_s:Q", title="Time (s from start)", scale=alt.Scale(zero=False)),
            x2="end_s:Q",
            y=alt.Y("batch_idx:O", title="Batch Step", sort="ascending", scale=alt.Scale(zero=False)),
            color=alt.Color("rate:Q", scale=alt.Scale(scheme="viridis"), title="Rate (items/s)"),
            tooltip=["batch_idx", "start_s", "end_s", "elapsed", "rate"],
        )
        .properties(width=800, height=max(300, len(df) * 8), title="Batch Work Schedule")
    )

    chart.save("gpu_metrics.png")


def single_run_check():
    p = "/opt/milabench/runs/sepaboda.2026-04-01_19-03-47/txt-to-image-single.D0.data"
    p = "/opt/milabench/runs/konetedo.2026-04-02_14-56-16/"
    p = "/opt/milabench/runs/jifenupu.2026-04-02_15-17-30"
    p = "/opt/milabench/runs/sogorepu.2026-04-02_15-58-15"
    data = []
    with DataProcessor(MetricExtractor, accept_file=lambda *args: True, backend=Threading) as proc:
        for event in proc(p):
            data.append(event)

    import pandas as pd 
    import altair as alt

    df = pd.DataFrame(data)

    print(list(df["metric"].unique()))

    t0 = df["time"].min()
    df["time"] = df["time"] - t0
    # df.loc[df["metric"] == "rate", "time"] = df.loc[df["metric"] == "rate", "time"] + 160

    chart1 = (
        alt.Chart(df[df["metric"] == "rate"])
        .mark_point()
        .encode(
            x=alt.X("time:Q", title="Time", scale=alt.Scale(zero=False)),
            y=alt.Y("value:Q", title="Batch Step", sort="ascending", scale=alt.Scale(zero=False)),
            color=alt.Color("metric:N", scale=alt.Scale(zero=False)),
        )
        
    )
    chart2 = (
        alt.Chart(df[df["metric"] == "gpudata.temperature"])
        .mark_point()
        .encode(
            x=alt.X("time:Q", title="Time", scale=alt.Scale(zero=False)),
            y=alt.Y("value:Q", title="Batch Step", sort="ascending", scale=alt.Scale(zero=False)),
            color=alt.Color("metric:N", scale=alt.Scale(zero=False)),
        )
    )

    (
        (chart1 + chart2)
            .properties(width=800, height=800)
            .resolve_scale(y="independent")
    ).save("gpu_metrics.png")

if __name__ == "__main__":
    single_run_check()
