import asyncio
from copy import deepcopy

from ovld import ovld

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
    from voir.instruments.gpu import get_gpu_info

    gpus = get_gpu_info()["gpus"].values()
    ngpus = len(gpus)
    if not gpus:
        gpus = [{"device": 0, "selection_variable": "CPU_VISIBLE_DEVICE"}]

    for gpu in gpus:
        gid = gpu["device"]
        gcfg = {
            "tag": [*cfg["tag"], f"D{gid}"],
            "device": gid,
            "devices": [gid] if ngpus else [],
            "env": {gpu["selection_variable"]: str(gid)},
        }
        yield clone_with(cfg, gcfg)


@planning_method
def njobs(cfg, n):
    for i in range(n):
        gcfg = {
            "tag": [*cfg["tag"], f"{i}"],
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
        elif v is True:
            args.append(k)
        elif k == "--":
            args.extend(v)
        elif v is False:
            raise ValueError("Use null to cancel an option, not false")
        else:
            args.append(k)
            args.append(",".join(map(str, v)) if isinstance(v, list) else str(v))
    return args


class MultiPackage:
    def __init__(self, packs):
        self.packs = packs

    async def do_install(self):
        for pack in self.packs.values():
            await pack.checked_install()

    async def do_prepare(self):
        for pack in self.packs.values():
            await pack.prepare()

    async def do_run(self, repeat=1):
        for index in range(repeat):
            for pack in self.packs.values():
                cfg = pack.config
                plan = deepcopy(cfg["plan"])
                method = get_planning_method(plan.pop("method"))
                coroutines = []

                for run in method(cfg, **plan):
                    if repeat > 1:
                        run["tag"].append(f"R{index}")
                    run_pack = pack.copy(run)
                    await run_pack.send(event="config", data=run)
                    args = _assemble_options(run.get("argv", {}))
                    coroutines.append(run_pack.run(args=args))

                await asyncio.gather(*coroutines)

    async def do_pin(self, pip_compile_args, constraints: list = tuple()):
        installed_groups = set()
        for pack in self.packs.values():
            if pack.config["group"] in installed_groups:
                continue
            await pack.pin(
                pip_compile_args=pip_compile_args,
                constraints=constraints,
            )
            installed_groups.add(pack.config["group"])
