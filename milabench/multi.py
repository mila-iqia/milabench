import json
import subprocess
import time
from copy import deepcopy

from giving import give, given

from .merge import merge

planning_methods = {}


def get_planning_method(name):
    name = name.replace("-", "_")
    return planning_methods[name]


def planning_method(f):
    planning_methods[f.__name__] = f


def clone_with(cfg, new_cfg):
    return merge(deepcopy(cfg), new_cfg)


@planning_method
def per_gpu(cfg):
    import GPUtil as gu

    gpus = gu.getGPUs()

    ids = [gpu.id for gpu in gpus] or [0]

    for gid in ids:
        gcfg = {
            "tag": [f"D{gid}"],
            "device": gid,
            "devices": [gid] if ids else [],
            "env": {"CUDA_VISIBLE_DEVICES": str(gid)},
        }
        yield clone_with(cfg, gcfg)


@planning_method
def njobs(cfg, n):
    for i in range(n):
        gcfg = {
            "tag": [f"{i}"],
            "job-number": i,
        }
        yield clone_with(cfg, gcfg)


def multiread(pipes):
    import os
    import time

    for (_, __, p) in pipes.values():
        os.set_blocking(p.stdout.fileno(), False)

    while pipes:
        for tag, (pack, run, proc) in list(pipes.items()):
            while True:
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        # We do not read a line and the process is over
                        del pipes[tag]
                    break
                try:
                    data = json.loads(line.decode("utf8"))
                except json.JSONDecodeError:
                    data = {"#unknown": line}
                data["#pack"] = pack
                data["#run"] = run
                give(**data)
        time.sleep(0.1)


class MultiPackage:
    def __init__(self, packs):
        self.packs = packs

    def do_install(self, dash):
        with given() as gv, dash(gv):
            for name, pack in self.packs.items():
                pack.do_install()

    def do_run(self, dash, report):
        with given() as gv, dash(gv), report(gv):
            for name, pack in self.packs.items():
                processes = {}
                cfg = pack.config
                plan = deepcopy(cfg["plan"])
                method = get_planning_method(plan.pop("method"))
                for run in method(cfg, **plan):
                    tag = ".".join(run["tag"])
                    give(**{"#pack": pack, "#run": run, "#start": time.time()})
                    process = subprocess.Popen(
                        ["milarun", "job", json.dumps(run)],
                        env=pack._nox_session.env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    assert tag not in processes
                    processes[tag] = (pack, run, process)

                multiread(processes)
