import asyncio
import traceback
from collections import defaultdict
from copy import deepcopy
from contextlib import contextmanager

from filelock import FileLock, Timeout
from voir.instruments.gpu import get_gpu_info

from .capability import is_system_capable
from .commands import NJobs, PerGPU
from .config import set_run_count
from .fs import XPath
from .config import get_base_folder
from .pack import Package
from .remote import (
    is_main_local,
    is_multinode,
    is_remote,
    milabench_remote_install,
    milabench_remote_prepare,
    milabench_remote_run,
)
from .utils import make_constraints_file

here = XPath(__file__).parent

planning_methods = {}


def lock_file(phase):
    folder = get_base_folder() / "extra" / f"{phase}.lock"
    folder.parent.mkdir(exist_ok=True)
    return folder


@contextmanager
def phase_lock(phase):
    filename = lock_file(phase)
    try:
        yield FileLock(filename, blocking=False)
    except Timeout:
        print("Shared filesystem detected")


async def aprint(pack, msg):
    await pack.send(event="line", data=msg, pipe="stdout")


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

async def copy_base_to_workers(setup):
    # Note: when we use docker we do not need to install
    # so this should be ignored
    if is_main_local(setup) and is_multinode(setup):
        print("Coping main setup from this node to worker")
        # copy the main setup to the workers
        # so it copies the bench venv already, no need for python
        from milabench.remote import copy_folder
        from milabench.system import SystemConfig

        # we copy the entire content of base
        #   FIXME: handle custom (venv, cache, data, etc...) directories
        #  
        copy_plan = copy_folder(setup, SystemConfig().base)
        remote_task = asyncio.create_task(copy_plan.execute())
        await asyncio.wait([remote_task])


class MultiPackage:
    def __init__(self, packs):
        self.packs = packs

    def setup_pack(self) -> Package:
        pack = list(self.packs.values())[0]
        name = "setup"
        if is_remote(pack):
            name = "remote"

        return Package(
            {
                "name": name,
                "tag": [name],
                "definition": ".",
                "run_name": pack.config["run_name"],
                "dirs": pack.config["dirs"],
                "config_base": pack.config["config_base"],
                "config_file": pack.config["config_file"],
                "system": pack.config["system"],
            }
        )

    async def do_phase(self, phase_name, remote_task, method):
        """Run a phase on all the nodes"""

        pending = []
        if remote_task:
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
        remote_task = None

        if is_remote(setup):
            print("Current node is outside of our system")
            # We are outside system, setup the main node first
            remote_plan = milabench_remote_install(setup, setup_for="main")
            remote_task = asyncio.create_task(remote_plan.execute())
            await asyncio.wait([remote_task])

            # We do not install benchmarks on that node
            return

        # elif is_main_local(setup) and is_multinode(setup):
        #     # this was executing install on the remote node but then it needed python to be available
        #     # We are the main node, setup workers
        #     remote_plan = milabench_remote_install(setup, setup_for="worker")
        #     remote_task = asyncio.create_task(remote_plan.execute())

        # do the installation step
        with phase_lock("install"):
            await self.do_phase("install", remote_task, "checked_install")

        await copy_base_to_workers(setup)

    async def do_prepare(self):
        setup = self.setup_pack()
        remote_task = None

        if is_remote(setup):
            remote_plan = milabench_remote_prepare(setup, run_for="main")
            remote_task = asyncio.create_task(remote_plan.execute())
            await asyncio.wait([remote_task])

            return

        # elif is_main_local(setup) and is_multinode(setup):
        #     remote_plan = milabench_remote_prepare(setup, run_for="worker")
        #     remote_task = asyncio.create_task(remote_plan.execute())

        with phase_lock("prepare"):
            await self.do_phase("prepare", remote_task, "prepare")

        # Prepare is done on the main node
        # copy the result there
        await copy_base_to_workers(setup)

    async def do_run(self, repeat=1):
        setup = self.setup_pack()

        if is_remote(setup):
            # if we are not on the main node right now
            # ssh to the main node and launch milabench
            remote_plan = milabench_remote_run(setup)
            remote_task = asyncio.create_task(remote_plan.execute())
            await asyncio.wait([remote_task])
            return

        assert is_main_local(setup), "Running benchmarks only works on the main node"
        await self.count_runs(repeat)
        
        for index in range(repeat):
            for pack in self.packs.values():
                try:
                    if not await is_system_capable(pack):
                        continue

                    exec_plan = make_execution_plan(pack, index, repeat)

                    print(repr(exec_plan))
                    await exec_plan.execute("run", timeout=True, timeout_delay=600)

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

                pindir = here.parent / XPath(".pin")

                constraint_path = pindir / "tmp-constraints.txt"
                constraint_files = make_constraints_file(
                    constraint_path, constraints, str(here.parent)
                )

                ig_constraint_path = pindir / f"constraints-{ivar}-{ig}.txt"
                if ig_constraint_path.exists() and from_scratch:
                    ig_constraint_path.rm()

                # Create master requirements
                await pack0.exec_pip_compile(
                    requirements_file=ig_constraint_path.absolute(),
                    input_files=(*constraint_files, *reqs),
                    argv=pip_compile_args,
                    working_dir=here.parent,
                )

                if not ig_constraint_path.exists():
                    raise RuntimeError("Could not generate main requirements")

                # Use master requirements to constrain the rest
                new_constraints = [ig_constraint_path, *constraints]
                for pack in packs:
                    await pack.pin(
                        pip_compile_args=pip_compile_args,
                        constraints=new_constraints,
                        working_dir=here.parent,
                    )

    async def count_runs(self, repeat):
        total_run = 0
        total_bench = 0
        for index in range(repeat):
            for pack in self.packs.values():
                if not await is_system_capable(pack):
                    continue

                exec_plan = make_execution_plan(pack, index, repeat)

                total_bench += 1

                if isinstance(exec_plan, PerGPU):
                    total_run += len(exec_plan.devices)
                else:
                    total_run += 1

        set_run_count(total_run, total_bench)
        return total_run
