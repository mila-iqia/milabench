import asyncio
from collections import defaultdict
from copy import deepcopy
import traceback

from voir.instruments.gpu import get_gpu_info

from .executors import NJobs, PerGPU
from .fs import XPath
from .utils import make_constraints_file
from .pack import Package
from .remote import (
    milabench_remote_install,
    milabench_remote_prepare,
    milabench_remote_run,
    is_main_local,
    is_remote,
)

here = XPath(__file__).parent

gpus = get_gpu_info()["gpus"].values()

planning_methods = {}


def get_planning_method(name):
    name = name.replace("-", "_")
    return planning_methods[name]


def planning_method(f):
    planning_methods[f.__name__] = f


def make_execution_plan(pack, step=0, repeat=1):
    cfg = deepcopy(pack.config)
    plan = deepcopy(cfg["plan"])

    if repeat > 1:
        cfg["tag"].append(f"R{step}")

    run_pack = pack.copy(cfg)
    method = plan.pop("method").replace("-", "_")

    # This is wrong because it does not know yet
    # own many GPUs will be used for the GPU
    exec_plan = run_pack.build_run_plan()
    devices = get_gpu_info()["gpus"].values()

    if method == "per_gpu":
        exec_plan = PerGPU(exec_plan, devices)

    elif method == "njobs":
        n = plan.pop("n")
        exec_plan = NJobs(exec_plan, n, devices)

    else:
        raise RuntimeError("Execution plan not specified")

    return exec_plan


class MultiPackage:
    def __init__(self, packs):
        self.packs = packs

    def setup_pack(self) -> Package:
        pack = list(self.packs.values())[0]

        return Package(
            {
                "name": "setup",
                "tag": ["setup"],
                "definition": ".",
                "run_name": pack.config["run_name"],
                "dirs": pack.config["dirs"],
                "config_base": pack.config["config_base"],
                "config_file": pack.config["config_file"],
                "system": pack.config["system"],
            }
        )

    async def do_phase(self, phase_name, remote_plan, method):
        """Run a phase on all the nodes"""
        remote_task = asyncio.create_task(remote_plan.execute())
        pending = [remote_task]

        for pack in self.packs.values():
            pack.phase = phase_name
            try:
                phase_task = asyncio.create_task(getattr(pack, method)())

                coro = [phase_task, *pending]
                done = []

                while phase_task not in done:
                    done, pending = await asyncio.wait(
                        coro, return_when=asyncio.FIRST_COMPLETED
                    )
                    coro = pending

            except Exception as exc:
                traceback.print_exc()
                await pack.message_error(exc)

        if pending:
            await asyncio.wait(pending)

    async def do_install(self):
        setup = self.setup_pack()
        
        if is_remote(setup):
            remote_plan = milabench_remote_install(setup)
            remote_task = asyncio.create_task(remote_plan.execute())
            await asyncio.wait([remote_task])
            return
            
        # Something we could do is run the remote setup first (COPY & install milabench)
        # then later we could remotely do `milabench install|prepare --select {current_pack}`
        # to install & prepare packs in groups
        await self.do_phase(
            "install", milabench_remote_install(setup), "checked_install"
        )

    async def do_prepare(self):
        setup = self.setup_pack()
        
        if is_remote():
            remote_plan = milabench_remote_prepare(setup)
            remote_task = asyncio.create_task(remote_plan.execute())
            await asyncio.wait([remote_task])
            return
            
        await self.do_phase(
            "prepare", milabench_remote_prepare(setup), "prepare"
        )

    async def do_run(self, repeat=1):
        setup = self.setup_pack()

        if is_remote(setup):
            # if we are not on the main node right now
            # ssh to the main node and launch milabench

            remote_plan = milabench_remote_run(setup)
            remote_task = asyncio.create_task(remote_plan.execute())
            await asyncio.wait([remote_task])
            return

        for index in range(repeat):
            for pack in self.packs.values():
                try:

                    def capability_failure():
                        caps = dict(pack.config["capabilities"])
                        for condition in pack.config.get("requires_capabilities", []):
                            if not eval(condition, caps):
                                return condition
                        return False

                    if condition_failed := capability_failure():
                        await pack.message(
                            f"Skip {pack.config['name']} because the following capability is not satisfied: {condition_failed}"
                        )
                        continue

                    exec_plan = make_execution_plan(pack, index, repeat)
                    await exec_plan.execute(timeout=True, timeout_delay=600)

                except Exception as exc:
                    import traceback

                    traceback.print_exc()
                    await pack.message_error(exc)

    async def do_pin(
        self, pip_compile_args, constraints: list = tuple(), from_scratch=False
    ):
        groups = defaultdict(dict)
        for pack in self.packs.values():
            pack.phase = "pin"
            igrp = pack.config["install_group"]
            ivar = pack.config["install_variant"]
            ivar_constraints: XPath = here.parent / "constraints" / f"{ivar}.txt"
            base_reqs = pack.requirements_map().keys()
            if ivar_constraints.exists():
                constraints = {ivar_constraints, *constraints}
            groups[igrp].update({req: pack for req in base_reqs})

        for constraint in constraints:
            print("Using constraint file:", constraint)

        groups = {
            name: (set(group.keys()), set(group.values()))
            for name, group in groups.items()
        }

        for ig, (reqs, packs) in groups.items():
            if len(packs) < len(reqs):
                if len(set(p.config["group"] for p in packs)) > 1:
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
                ivar = pack0.config["install_variant"]

                pindir = XPath(".pin")

                constraint_path = pindir / "tmp-constraints.txt"
                constraint_files = make_constraints_file(constraint_path, constraints)

                ig_constraint_path = pindir / f"constraints-{ivar}-{ig}.txt"
                if ig_constraint_path.exists() and from_scratch:
                    ig_constraint_path.rm()

                # Create master requirements
                await pack0.exec_pip_compile(
                    requirements_file=ig_constraint_path.absolute(),
                    input_files=(*constraint_files, *reqs),
                    argv=pip_compile_args,
                )

                # Use master requirements to constrain the rest
                new_constraints = [ig_constraint_path, *constraints]
                for pack in packs:
                    await pack.pin(
                        pip_compile_args=pip_compile_args,
                        constraints=new_constraints,
                    )
