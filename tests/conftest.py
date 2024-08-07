import os
import stat
from pathlib import Path


from milabench.testing import official_config
import pytest
import voir.instruments.gpu as voirgpu

here = Path(__file__).parent


if "MILABENCH_CONFIG" in os.environ:
    del os.environ["MILABENCH_CONFIG"]


@pytest.fixture
def runs_folder():
    return here / "runs"


@pytest.fixture
def config():
    def get_config(name):
        return here / "config" / f"{name}.yaml"
    return get_config


@pytest.fixture(scope='session')
def module_tmp_dir():
    import tempfile
    yield tempfile.mkdtemp()


@pytest.fixture(scope='session')
def standard_config():
    return official_config("standard")


@pytest.fixture
def replayfolder():
    return here / "replays"


class MockDeviceSMI:
    def __init__(self) -> None:
        self.devices = [0]
        self.used = [1]
        self.total = [8000000]
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
            "selection_variable": "MOCK_VISIBLE_DEVICES",
        }

    @property
    def arch(self):
        return "mock"

    @property
    def visible_devices(self):
        return os.environ.get("MOCK_VISIBLE_DEVICES", None)

    def get_gpus_info(self, selection=None):
        gpus = dict()
        for device in self.devices:
            if (selection is None) or (selection and str(device) in selection):
                gpus[device] = self.get_gpu_info(device)

        return gpus

    def close(self):
        pass

class MockDeviceModule:
    @staticmethod
    def is_installed():
        return True

    DeviceSMI = MockDeviceSMI


@pytest.fixture(scope="session", autouse=True)
def set_env():
    voirgpu.BACKENDS["mock"] = MockDeviceModule

    try:
        backend = voirgpu.deduce_backend()
    except Exception:
        backend = "mock"

    if backend == "cpu":
        backend = "mock"

    os.environ["MILABENCH_CONFIG"] = "config/ci.yaml"
    os.environ["MILABENCH_BASE"] = "output"
    os.environ["MILABENCH_DASH"] = "no"
    os.environ["MILABENCH_GPU_ARCH"] = backend

    #
    # milabench expects voir to be installed in the bench venv
    # we fake one to use the one we have in the current env
    os.makedirs("output/venv/benchio/bin/", exist_ok=True)
    voirexec = "output/venv/benchio/bin/voir"
    with open(voirexec, "w") as fp:
        fp.write("#!/bin/bash\n")
        fp.write('python -m voir "$@"')

    current_permissions = stat.S_IMODE(os.lstat(voirexec).st_mode)
    os.chmod(
        voirexec, current_permissions | (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    )

    if backend == "mock":
        oldsmi = voirgpu.DEVICESMI
        voirgpu.DEVICESMI = MockDeviceSMI()

    yield

    if backend == "mock":
        voirgpu.DEVICESMI = oldsmi

    # --
    # --


@pytest.fixture
def multipack(config, tmp_path):
    from milabench.common import _get_multipack, arguments

    args = arguments()
    args.config = config("benchio")
    args.system = config("system")
    args.base = tmp_path
    args.use_current_env = True
    args.select = None
    args.exclude = None
    run_name = "test"
    overrides = {}

    return _get_multipack(
        args=args,
        run_name=run_name,
        overrides=overrides,
    )
