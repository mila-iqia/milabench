import asyncio
import tempfile
from collections import defaultdict
from copy import deepcopy

from milabench.utils import make_constraints_file

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


class MultiPackage:
    def __init__(self, packs):
        self.packs = packs

    async def do_install(self):
        for pack in self.packs.values():
            try:
                await pack.checked_install()
            except Exception as exc:
                await pack.message_error(exc)

    async def do_prepare(self):
        for pack in self.packs.values():
            try:
                await pack.prepare()
            except Exception as exc:
                await pack.message_error(exc)

    async def do_run(self, repeat=1):
        for index in range(repeat):
            for pack in self.packs.values():
                try:
                    cfg = pack.config
                    plan = deepcopy(cfg["plan"])
                    method = get_planning_method(plan.pop("method"))
                    coroutines = []

                    for run in method(cfg, **plan):
                        if repeat > 1:
                            run["tag"].append(f"R{index}")
                        run_pack = pack.copy(run)
                        await run_pack.send(event="config", data=run)
                        coroutines.append(run_pack.run())

                    await asyncio.gather(*coroutines)
                except Exception as exc:
                    await pack.message_error(exc)

    async def do_pin(self, pip_compile_args, constraints: list = tuple()):
        groups = defaultdict(dict)
        for pack in self.packs.values():
            igrp = pack.config["install_group"]
            base_reqs = pack.requirements_map().keys()
            groups[igrp].update({req: pack for req in base_reqs})

        groups = {
            name: (set(group.keys()), set(group.values()))
            for name, group in groups.items()
        }

        for ig, (reqs, packs) in groups.items():
            if len(packs) < len(reqs):
                raise Exception(
                    f"Install group '{ig}' contains benchmarks that have more than"
                    " one requirements file. Please isolate such benchmarks in their"
                    " own install_group."
                )

        for ig, (reqs, packs) in groups.items():
            packs = list(packs)
            if len(packs) == 1:
                (pack,) = packs
                await pack.pin(
                    pip_compile_args=pip_compile_args,
                    constraints=constraints,
                )
            else:
                pack0 = packs[0]

                constraint_files = make_constraints_file(constraints)
                with tempfile.NamedTemporaryFile() as tf:
                    # Create master requirements
                    await pack0.exec_pip_compile(
                        requirements_file=tf.name,
                        input_files=(*constraint_files, *reqs),
                        argv=pip_compile_args,
                    )

                    # Use master requirements to constrain the rest
                    new_constraints = [tf.name, *constraints]
                    for pack in packs:
                        await pack.pin(
                            pip_compile_args=pip_compile_args,
                            constraints=new_constraints,
                        )
