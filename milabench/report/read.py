
import multiprocessing as mp
import queue
import traceback
import os
import json
import re
import glob
from pathlib import Path


def extract_tags(name, tags):
    for tag, pat in tags.items():
        if m := pat.search(name):
            value = m.group(1)
            yield tag, value
        else:
            print(f"{tag} not found in {name}")
            yield tag, "NA"



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


def worker_fn(i, cls, worker_pack, *args):
    worker = cls(worker_pack, *args)
    worker.run()



class DataProcessor:
    def __init__(self, worker_cls, *args, worker_count=mp.cpu_count()):
        self.work_queue = mp.Queue(maxsize=0)
        self.result_queue = mp.Queue()
        self.error_queue = mp.Queue()
        self.active = mp.Event()
        self.active.set()

        self.jobs = mp.Value("i", 0)
        self.done = mp.Value("i", 0)
        self.results = mp.Value("i", 0)
        self.retrieved = 0

        worker_pack = self.work_queue, self.result_queue, self.error_queue, self.active, self.jobs, self.done, self.results
        args = (worker_cls, worker_pack, *args)
        self.workers = [mp.Process(target=worker_fn, args=(i,) + args) for i in range(worker_count)]
    
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

        for folder in glob.iglob(folderpat, recursive=False):
            folder = Path(folder)
            if folder.is_file():
                meta = extract_meta_from_run_folder(folder.parent, meta)

            readfolder(self, folder, meta)

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


def make_tags(tags_def):
    tags = dict()
    for tag in tags_def:
        name, regex = tag.split("=")
        tags[name] = re.compile(regex)
    return tags


def extract_tags(name, tags):
    for tag, pat in tags.items():
        if m := pat.search(name):
            value = m.group(1)
            yield tag, value



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

bench_tags = make_tags(_bench_tags)


_run_tags = [
    "clock=g([0-9]+)",
    "power=p([0-9]+)",
    "observation=o([0-9]+)",
]
run_tags = make_tags(_run_tags)


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


class EventProcessor(Worker):
    """Base milabench event processor"""
    def __init__(self, worker_pack, accept_pattern=tuple(), ignored=tuple(["$queued"]), *args, **kwargs):
        super().__init__(worker_pack)
        self.ignored = ignored
        self.pattern = accept_pattern

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
        if file.endswith(".data"):
            with open(file, "r") as fp:

                # Build a shared metadata, that events can modify
                file_meta = {**meta}

                for line in fp.readlines():
                    metric_line = json.loads(line)
                    
                    if (stop := self.processline(metric_line, file_meta)) is True:
                        break

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


class MetricExtractor(EventProcessor):
    def data(self, data, meta):
        unit = data.pop("unit", None)
        units = data.pop("units", None)
        time = data.pop("time", None)
        task = data.pop("task", None)
        batch_id = data.pop("batch_id", None)

        for k, v in flatten_values(data, tuple([])):
            if k in self.ignored:
                continue

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

        {'metric': 'gpudata.2.memory.0'   , 'value': 47281.3125, 'time': 1766765836.8106523, 'unit': None, 'task': 'main', 'bench': 'convnext_large-fp32', 'clock': '1440', 'observation': '350', 'p0': 'g1440.o350.2025-12-26_09-59-43', 'device': '2'}
        {'metric': 'gpudata.2.memory.1'   , 'value': 143771.0  , 'time': 1766765836.8106523, 'unit': None, 'task': 'main', 'bench': 'convnext_large-fp32', 'clock': '1440', 'observation': '350', 'p0': 'g1440.o350.2025-12-26_09-59-43', 'device': '2'}
        {'metric': 'gpudata.2.load'       , 'value': 1.0       , 'time': 1766765836.8106523, 'unit': None, 'task': 'main', 'bench': 'convnext_large-fp32', 'clock': '1440', 'observation': '350', 'p0': 'g1440.o350.2025-12-26_09-59-43', 'device': '2'}
        {'metric': 'gpudata.2.temperature', 'value': 65.0      , 'time': 1766765836.8106523, 'unit': None, 'task': 'main', 'bench': 'convnext_large-fp32', 'clock': '1440', 'observation': '350', 'p0': 'g1440.o350.2025-12-26_09-59-43', 'device': '2'}
        {'metric': 'gpudata.2.power'      , 'value': 442.059   , 'time': 1766765836.8106523, 'unit': None, 'task': 'main', 'bench': 'convnext_large-fp32', 'clock': '1440', 'observation': '350', 'p0': 'g1440.o350.2025-12-26_09-59-43', 'device': '2'}

    """
    with DataProcessor(MetricExtractor) as proc:
        for item in proc(folder):
            yield item


if __name__ == "__main__":

    #
    # The data processor can take as argument all those arguments
    #   multi run folder
    #   single run folder
    #   single bench file
    #   glob pattern

    p = "/home/delaunao/workspace/benchdevenv/projects/hypertec/sxm_runs/"
    p = "/home/delaunao/workspace/benchdevenv/projects/hypertec/sxm_runs/p600.o500.2025-12-26_07-20-23"
    # p = "/home/delaunao/workspace/benchdevenv/projects/hypertec/sxm_runs/p600.o*"
    # p = "/home/delaunao/workspace/benchdevenv/projects/hypertec/sxm_runs/p600.o500.2025-12-26_07-20-23/convnext_large-tf32-fp16.D0.data"
    # p = "/home/delaunao/workspace/benchdevenv/projects/hypertec/nvl_runs/g1440.o350.2025-12-26_09-59-43/convnext_large-fp32.D*"

    from collections import defaultdict

    metrics = defaultdict(int)
    total = 0

    with DataProcessor(LogExtractor) as proc:
        for i, item in enumerate(proc(p)):
            # metrics[item["metric"]] += 1
            total += 1
            # print(item)

        
    for k, v in metrics.items():
        print(f"{k:>30}: {v:6d}    {v/total:5.2%}") 
    