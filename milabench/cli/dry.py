from dataclasses import dataclass, field
from contextlib import contextmanager

from coleo import Option, tooled
import voir.instruments.gpu as voirgpu
import os

from ..common import get_multipack
from ..multi import make_execution_plan

# fmt: off
@dataclass
class Arguments:
    ngpu: int = 8
    capacity: int = 80000
# fmt: on


@tooled
def arguments():
    ngpu = 8
    capacity = 80000
    return Arguments(ngpu, capacity)


class MockDeviceSMI:
    def __init__(self, ngpu, capacity) -> None:
        self.devices = [i for i in range(ngpu)]
        self.used = [1]
        self.total = [capacity]
        self.util = [40]
        self.temp = [35]
        self.power = [225]

    def get_gpu_info(self, device):
        return {
            "device": device,
            "product": "MockDevice",
            "memory": {
                "used": self.used[0] // (1024**2),
                "total": self.total[0] // (1024**2),
            },
            "utilization": {
                "compute": float(self.util[0]) / 100,
                "memory": self.used[0] / self.total[0],
            },
            "temperature": self.temp[0],
            "power": self.power[0],
            "selection_variable": "CUDA_VISIBLE_DEVICES",
        }

    @property
    def arch(self):
        return "mock"

    @property
    def visible_devices(self):
        return os.environ.get("CUDA_VISIBLE_DEVICES", None)

    def get_gpus_info(self, selection=None):
        gpus = dict()
        for device in self.devices:
            if (selection is None) or (selection and str(device) in selection):
                gpus[device] = self.get_gpu_info(device)

        return gpus

    def close(self):
        pass


@contextmanager
def assume_gpu(ngpu=1, capacity=80000, enabled=False):
    if enabled:
        old = voirgpu.DEVICESMI
        voirgpu.DEVICESMI = MockDeviceSMI(ngpu, capacity)
        yield
        voirgpu.DEVICESMI = old
    else:
        yield


class BashGenerator:
    def __init__(self) -> None:
        self.indent = 0
        self.background_mode = False

    def print(self, *args, **kwargs):
        print("  " * self.indent, end="")
        print(*args, **kwargs)

    def comment(self, cmt):
        self.print(f"# {cmt}")

    def env(self, env):
        for k, v in env.items():
            self.print(f'export {k}="{v}"')

    @contextmanager
    def subshell(self):
        self.print("(")
        self.indent += 1
        yield
        self.indent -= 1
        self.print(")")

    @contextmanager
    def background(self):
        self.background_mode = True
        yield
        self.print("wait")
        self.background_mode = False

    def command(self, *args, env=None, **kwargs):
        prefix = []
        if env is not None:
            for k, v in env.items():
                prefix.append(f"{k}={v}")

        prefix = " ".join(prefix)
        sufix = ""
        if True:
            sufix = "&"

        self.print(prefix, " ".join(str(a) for a in args), sufix)


@tooled
def cli_dry(args=None):
    """Generate dry commands to execute the bench standalone"""

    if args is None:
        args = arguments()

    with assume_gpu(args.ngpu, args.capacity, enabled=True):
        repeat = 1
        mp = get_multipack(run_name="dev")
        gen = BashGenerator()

        first_pack = True
        for index in range(repeat):
            for pack in mp.packs.values():
                if first_pack:
                    first_pack = False
                    gen.comment("Virtual Env")
                    gen.env(pack.core._nox_session.env)
                    gen.comment("Milabench")
                    gen.env(pack.make_env())

                exec_plan = make_execution_plan(pack, index, repeat)

                with gen.subshell():
                    gen.comment("Benchmark")
                    with gen.background():
                        for pack, argv, _ in exec_plan.commands():
                            # gen.env(pack.config.get("env", {}))
                            gen.command(*argv, env=pack.config.get("env", {}))

                print()
