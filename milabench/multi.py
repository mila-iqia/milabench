import subprocess
import time
from copy import deepcopy

from giving import give, given
from ovld import ovld
from voir.forward import MultiReader

from .merge import merge
from .utils import give_std

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


@ovld
def _assemble_options(options: list):
    return options


@ovld
def _assemble_options(options: dict):
    args = []
    for k, v in options.items():
        if v is None:
            continue
        if v is True:
            args.append(k)
        elif v is False:
            raise ValueError("Use null to cancel an option, not false")
        else:
            args.append(k)
            args.append(",".join(map(str, v)) if isinstance(v, list) else str(v))
    return args


class MultiPackage:
    def __init__(self, packs):
        self.packs = packs
        (self.rundir,) = {p.dirs.runs for p in packs.values()}

    def do_install(self, dash, force=False, sync=False):
        with given() as gv, dash(gv), give_std():
            for pack in self.packs.values():
                with give.inherit(**{"#pack": pack}):
                    pack.checked_install(force=force, sync=sync)

    def do_prepare(self, dash):
        with given() as gv, dash(gv), give_std():
            for pack in self.packs.values():
                with give.inherit(**{"#pack": pack}):
                    mr = MultiReader()
                    process = pack.prepare()
                    if isinstance(process, subprocess.Popen):
                        mr.add_process(process, info={"#pack": pack})
                        for _ in mr:
                            time.sleep(0.1)

    def do_run(self, dash, report, repeat=1):
        with given() as gv, dash(gv), report(gv, self.rundir):
            for i in range(repeat):
                for pack in self.packs.values():
                    cfg = pack.config
                    plan = deepcopy(cfg["plan"])
                    method = get_planning_method(plan.pop("method"))
                    mr = MultiReader()
                    for run in method(cfg, **plan):
                        if repeat > 1:
                            run["tag"].append(f"R{i}")
                        info = {"#pack": pack, "#run": run}
                        give(**{"#start": time.time()}, **info)
                        give(**{"#config": run}, **info)
                        voirargs = _assemble_options(run.get("voir", {}))
                        args = _assemble_options(run.get("argv", {}))
                        env = run.get("env", {})
                        process = pack.run(args=args, voirargs=voirargs, env=env)
                        mr.add_process(process, info=info)

                    try:
                        for _ in mr:
                            time.sleep(0.1)

                    except BaseException as exc:
                        for (proc, info) in mr.processes:
                            errstring = f"{type(exc).__name__}: {exc}"
                            endinfo = {
                                "#end": time.time(),
                                "#completed": False,
                                "#error": errstring,
                            }
                            give(**endinfo, **info)
                            proc.kill()
                        if not isinstance(exc, KeyboardInterrupt):
                            raise
                        break

    def do_dev(self, dash):
        # TODO: share the common code between do_run and do_dev
        with given() as gv, dash(gv):
            for pack in self.packs.values():
                cfg = pack.config
                plan = {"method": "njobs", "n": 1}
                method = get_planning_method(plan.pop("method"))
                mr = MultiReader()
                for run in method(cfg, **plan):
                    run["tag"] = [run["name"]]
                    give(**{"#pack": pack, "#run": run, "#start": time.time()})
                    voirargs = _assemble_options(run.get("voir", {}))
                    args = _assemble_options(run.get("argv", {}))
                    env = run.get("env", {})
                    process = pack.run(args=args, voirargs=voirargs, env=env)
                    mr.add_process(process, info={"#pack": pack, "#run": run})

                for _ in mr:
                    time.sleep(0.1)
