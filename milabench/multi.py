import json
import subprocess
from copy import deepcopy

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
            "tag": [f"X{i}"],
            "job-number": i,
        }
        yield clone_with(cfg, gcfg)


class MultiPackage:
    def __init__(self, packs):
        self.packs = packs

    def do_install(self):
        for name, pack in self.packs.items():
            pack.do_install()

    def do_run(self):
        for name, pack in self.packs.items():
            processes = []
            cfg = pack.config
            plan = deepcopy(cfg["plan"])
            method = get_planning_method(plan.pop("method"))
            for run in method(cfg, **plan):
                process = subprocess.Popen(
                    ["milarun", "job", json.dumps(run)], env=pack._nox_session.env
                )
                processes.append(process)
            for p in processes:
                p.wait()
