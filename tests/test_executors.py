import asyncio
import os

import pytest

import milabench.commands.executors
from milabench.alt_async import proceed
from milabench.commands import (
    NJobs,
    PackCommand,
    PerGPU,
    SingleCmdCommand,
    TorchRunCommand,
    VoirCommand,
)
from milabench.common import _get_multipack, arguments


class ExecMock1(SingleCmdCommand):
    def __init__(self, pack_or_exec, *exec_argv, **kwargs) -> None:
        super().__init__(pack_or_exec, **kwargs)
        self.exec_argv = exec_argv

    def _argv(self, **kwargs):
        return [
            f"cmd{self.__class__.__name__}",
            *self.exec_argv,
            *[f"{self.__class__.__name__}[arg{i}]" for i in range(2)],
            *[f"{self.__class__.__name__}{k}:{v}" for k, v in kwargs.items()],
        ]


class ExecMock2(ExecMock1):
    pass


TEST_FOLDER = os.path.dirname(__file__)


def benchio():
    args = arguments()
    args.config = os.path.join(TEST_FOLDER, "config", "benchio.yaml")
    args.base = "/tmp"
    args.use_current_env = True

    packs = _get_multipack(args)

    _, pack = packs.packs.popitem()
    return pack


@pytest.fixture
def noexecute(monkeypatch):
    async def execute(pack, *args, **kwargs):
        return [*args, *[f"{k}:{v}" for k, v in kwargs.items()]]

    monkeypatch.setattr(milabench.commands.executors, "execute", execute)


def mock_pack(pack):
    async def execute(*args, **kwargs):
        return [*args, *[f"{k}:{v}" for k, v in kwargs.items()]]

    mock = pack
    mock.execute = execute
    return mock


def test_executor_argv():
    submock = ExecMock1(mock_pack(benchio()), "a1", "a2")
    wrapmock = ExecMock2(submock, "a3")

    assert wrapmock.argv() == [
        "cmdExecMock2",
        "a3",
        "ExecMock2[arg0]",
        "ExecMock2[arg1]",
        "cmdExecMock1",
        "a1",
        "a2",
        "ExecMock1[arg0]",
        "ExecMock1[arg1]",
    ]

    submock = ExecMock2(mock_pack(benchio()))
    wrapmock = ExecMock1(submock)

    assert wrapmock.argv(k1="v1") == [
        "cmdExecMock1",
        "ExecMock1[arg0]",
        "ExecMock1[arg1]",
        "ExecMock1k1:v1",
        "cmdExecMock2",
        "ExecMock2[arg0]",
        "ExecMock2[arg1]",
        "ExecMock2k1:v1",
    ]


def test_executor_kwargs():
    submock = ExecMock1(mock_pack(benchio()), selfk1="sv1", selfk2="sv2")
    wrapmock = ExecMock2(submock, selfk1="sv1'", selfk3="sv3")
    kwargs = {"selfk2": "v2''", "selfk3": "v3''", "k4": "v4"}

    assert sorted(wrapmock.kwargs().keys()) == ["selfk1", "selfk2", "selfk3"]
    assert sorted(wrapmock.kwargs().values()) == ["sv1'", "sv2", "sv3"]


def test_executor_execute(noexecute):
    submock = ExecMock1(mock_pack(benchio()), "a1", selfk1="sv1")
    wrapmock = ExecMock2(submock, "a2", selfk2="sv2")

    result = asyncio.run(wrapmock.execute(k3="v3"))
    expected = [
        [
            "cmdExecMock2",
            "a2",
            "ExecMock2[arg0]",
            "ExecMock2[arg1]",
            "cmdExecMock1",
            "a1",
            "ExecMock1[arg0]",
            "ExecMock1[arg1]",
            "selfk1:sv1",
            "selfk2:sv2",
            "k3:v3",
        ]
    ]
    print(result)
    print(expected)
    assert result == expected


def test_pack_executor():
    # voir is not setup so we are not receiving anything
    executor = PackCommand(benchio(), "--start", "2", "--end", "20")

    acc = 0
    for r in proceed(executor.execute()):
        print(r)
        acc += 1

    assert acc >= 4, "Only 4 message received (config, meta, start, end)"


def test_voir_executor():
    executor = PackCommand(benchio(), "--start", "2", "--end", "20")
    voir = VoirCommand(executor)

    acc = 0
    for r in proceed(voir.execute()):
        print(r)
        acc += 1

    assert acc == 72


def test_timeout():
    executor = PackCommand(benchio(), "--start", "2", "--end", "20", "--sleep", 20)
    voir = VoirCommand(executor)

    acc = 0
    for r in proceed(voir.execute(timeout=True, timeout_delay=1)):
        print(r)
        acc += 1

    assert acc > 2 and acc < 72


def test_njobs_executor():
    executor = PackCommand(benchio(), "--start", "2", "--end", "20")
    voir = VoirCommand(executor)
    njobs = NJobs(voir, 5)

    acc = 0
    for r in proceed(njobs.execute()):
        print(r)
        acc += 1

    assert acc == 72 * 5


def test_njobs_gpus_executor():
    """Two GPUs so torch run IS used"""
    devices = mock_gpu_list()

    from importlib.util import find_spec

    if find_spec("torch") is None:
        pytest.skip("Pytorch is not installed")

    executor = PackCommand(benchio(), "--start", "2", "--end", "20")
    voir = VoirCommand(executor)
    torchcmd = TorchRunCommand(voir, use_stdout=True)
    njobs = NJobs(torchcmd, 1, devices)

    acc = 0
    for r in proceed(njobs.execute()):
        if r.event == "start":
            assert r.data["command"][0].endswith("torchrun")
        acc += 1
        print(r)

    assert acc == len(devices) * 72


def test_njobs_gpu_executor():
    """One GPU, so torch run is not used"""
    devices = [mock_gpu_list()[0]]

    executor = PackCommand(benchio(), "--start", "2", "--end", "20")
    voir = VoirCommand(executor)
    torch = TorchRunCommand(voir, use_stdout=True)
    njobs = NJobs(torch, 1, devices)

    acc = 0
    for r in proceed(njobs.execute()):
        print(r)

        if r.event == "start":
            assert r.data["command"][0].endswith("voir")

        acc += 1

    assert acc == len(devices) * 72


def test_njobs_novoir_executor():
    executor = PackCommand(benchio(), "--start", "2", "--end", "20")
    njobs = NJobs(executor, 5)

    acc = 0
    for r in proceed(njobs.execute()):
        print(r)
        acc += 1

    assert acc >= 2 * 10


def mock_gpu_list():
    return [
        {"device": 0, "selection_variable": "CUDA_VISIBLE_DEVICE"},
        {"device": 1, "selection_variable": "CUDA_VISIBLE_DEVICE"},
    ]


def test_per_gpu_executor():
    devices = mock_gpu_list()

    executor = PackCommand(benchio(), "--start", "2", "--end", "20")
    voir = VoirCommand(executor)
    plan = PerGPU(voir, devices)

    acc = 0
    for r in proceed(plan.execute()):
        print(r)
        acc += 1

    assert acc == len(devices) * 72


def test_void_executor():
    from milabench.commands import VoidCommand

    plan = VoirCommand(VoidCommand(benchio()))

    for _ in proceed(plan.execute()):
        pass
