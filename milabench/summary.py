from dataclasses import dataclass
from collections import defaultdict
from math import isnan, nan

import numpy as np

from .utils import error_guard
from .syslog import syslog


def node_count(config):
    max_num = config.get("num_machines", 1)
    return max_num


def local_gpu(config):
    method = config.get("plan", {"method": "per_gpu"}).get("method")

    if method == "njobs":
        devices = config.get("devices", [])
        return len(devices)
    else:
        return 1


def ngpu(pack):
    return node_count(pack) * local_gpu(pack)



def aggregate(run_data):
    """Group all the data inside a dictionary of lists"""
    omnibus = defaultdict(list)
    meta = None
    config = None
    start = None
    end = None
    early_stop = False

    for entry in run_data:
        event = entry["event"]

        if event == "config":
            config = entry["data"]
            omnibus["ngpu"] = [ngpu(config)]

        if event == "meta":
            meta = entry["data"]

        elif event == "data":
            data = dict(entry["data"])
            task = data.pop("task", None)
            time = data.pop("time", None)
        
            for k, v in data.items():
                if task is not None and k == "rate":
                    k = f"{task}_{k}"
        
                if time and isinstance(v, dict):
                    v["time"] = time
        
                omnibus[k].append(v)

        elif event == "line":
            omnibus[entry["pipe"]].append(entry["data"])

        elif event == "stop":
            early_stop = True

        elif event == "start":
            assert start is None
            start = entry["data"]

        elif event == "end":
            assert end is None
            end = entry["data"]

    if not config:
        # This is not a run
        return None

    assert config and start and end

    device = config.get("device", None)
    new_gpudata = []
    energy = 0
    last_time = None
    first_time = None

    for entry in omnibus.get("gpudata", []):
        # If a device is specified, filter entries that don't contain it
        now = entry.pop("time", None)

        if first_time is None: 
            first_time = now

        if device is not None:
            dev_key = str(device)

            if dev_key not in entry:
                continue  # skip this entry

            gpu_entry = entry[dev_key]

            # Keep only the requested device
            new_entry = {
                dev_key: gpu_entry,
            }
        else:
            new_entry = entry
        
        new_gpudata.append(new_entry)

        for gpu, data in new_entry.items():
            elasped = 0
            if now is not None and last_time is not None:
                elapsed = now - last_time
                energy += data["power"] * elapsed

        last_time = now

    omnibus["energy"] = [energy/1000]
    omnibus["gpudata"] = new_gpudata
    elasped = 0
    if last_time and first_time:
        elasped = last_time - first_time
    omnibus["elapsed"] = [elasped]
    # omnibus["perf_watt"] = perf / (energy / elapsed)

    if device is not None:
        omnibus["per_gpu"] = [(device, tr) for tr in omnibus["train_rate"]]

    if "loss" in omnibus:
        fl, ll = omnibus["loss"][0], omnibus["loss"][-1]
        omnibus["loss_gain"] = [ll - fl]

    omnibus["walltime"] = [end["time"] - start["time"]]

    success = early_stop or (
        end["return_code"] == 0
        and not any(isnan(loss) for loss in omnibus.get("loss", []))
        and bool(omnibus.get("train_rate", []))
    )

    if "nolog" in config["tag"]:
        success = True

    omnibus["success"] = [success]

    return {
        "config": config,
        "start": start,
        "end": end,
        "data": omnibus,
        "meta": meta
    }


def _classify(all_aggregates):
    """Group data by benchmark names"""
    classified = defaultdict(list)
    for agg in all_aggregates:
        config = agg["config"]
        classified[config["name"]].append(agg)
    return classified


def _merge(aggs):
    """Merge same bench data into a single list of observations"""

    results = {"data": defaultdict(list)}
    for agg in aggs:
        data = agg.pop("data")
        results.update(agg)

        for k, v in data.items():
            results["data"][k].extend(v)

    return results


nans = {
    "min": nan,
    "q1": nan,
    "median": nan,
    "q3": nan,
    "max": nan,
    "mean": nan,
    "std": nan,
    "sem": nan,
}


@error_guard(nans)
def _metrics(xs):
    xs = sorted(x for x in xs if x is not None)
    if len(xs) >= 5:
        xs = xs[1:-1]  # Remove min and max

    if not xs:
        return nans
    
    percentiles = [0, 25, 50, 75, 100]
    percentile_names = ["min", "q1", "median", "q3", "max"]
    metrics = dict(zip(percentile_names, np.percentile(xs, percentiles)))
    metrics["mean"] = np.mean(xs)
    metrics["std"] = np.std(xs)
    metrics["sem"] = np.std(xs) / len(xs) ** 0.5
    return metrics


@error_guard(dict())
def augment(group, query=tuple([])):
    """Optional augmentation steps that will add additional data.
    Usually extracted from the run itself
    """
    data = {}

    if "batch_size" in query:
        from .sizer import get_batch_size

        data["batch_size"] = get_batch_size(group["config"], group["start"])

    if "elapsed" in query:
        start_time = group["start"]["time"]
        end_time = group["end"]["time"]
        data["elapsed"] = end_time - start_time

    return data



@dataclass
class Stats:
    min: float
    q1: float
    median: float
    q3: float
    max: float
    mean: float
    std: float
    sem: float


@dataclass
class Summary:
    name: str  # benchmark name
    n: int  # instance
    successes: int  # number of successful run
    failures: int  # number of failed run
    train_rate: Stats  # train speed
    walltime: Stats
    per_gpu: dict[str, Stats]
    gpu_load: dict[str, dict[str, Stats]]
    weight: float
    enabled: bool
    meta: dict[str]


@error_guard(None)
def _summarize(group, query=tuple([])) -> Summary:
    agg = group["data"]
    gpudata = defaultdict(lambda: defaultdict(list))

    for entry in agg["gpudata"]:
        for device, data in entry.items():
            if data["memory"][0] == 1 or data["load"] == 0:
                continue
            gpudata[device]["memory"].append(data["memory"][0])
            gpudata[device]["load"].append(data["load"])

    per_gpu = defaultdict(list)
    for device, tr in agg["per_gpu"]:
        per_gpu[device].append(tr)

    config = group["config"]
    meta = group["meta"]

    additional = augment(group, query)

    perf =  _metrics(agg["train_rate"])

    return {
        "meta": meta,
        "name": config["name"],
        "group": config["group"],
        "n": len(agg["success"]),
        # In case of failure it is possible ngpu is 0 or 1
        "ngpu": max(agg["ngpu"]),
        "successes": sum(agg["success"]),
        "failures": sum(not x for x in agg["success"]),
        "train_rate": perf,
        "walltime": _metrics(agg["walltime"]),
        "per_gpu": {
            device: _metrics(train_rates) for device, train_rates in per_gpu.items()
        },
        "gpu_load": {
            device: {
                "memory": _metrics(data["memory"]),
                "load": _metrics(data["load"]),
            }
            for device, data in gpudata.items()
        },
        "weight": config.get("weight", 0),
        "extra": additional,
        "enabled": config.get("enabled", False),
        "energy": agg["energy"][0],
        "elpased": agg["elapsed"][0],
        # ---
        "perf_watt": perf["mean"] / (agg["energy"][0] / agg["elapsed"][0])
    }


def make_summary(runs, query=tuple([])) -> dict[str, Summary]:
    aggs = []
    for name, run in runs.items():
        try:
            if agg := aggregate(run):
                aggs.append(agg)
        except AssertionError:
            syslog("Ignoring run {0}: it looks like it did not finish successfully", name)
        except Exception as err:
            import traceback
            traceback.print_exc()
            syslog("Ignoring run {0}: beause of exception: {1}", name, err)

    classified = _classify(aggs)
    merged = {name: _merge(runs) for name, runs in classified.items()}
    summarized = {name: _summarize(agg, query) for name, agg in merged.items()}
    return summarized

