import asyncio
import os

from milabench.executors import Executor, PackExecutor, VoirExecutor, NJobs, TimeOutExecutor
from milabench.pack import Package
from milabench.cli import _get_multipack
from milabench.alt_async import proceed

class MockPack(Package):
    async def execute(self, *args, **kwargs):
        return (
            *args,
            *[f"{k}:{v}" for k, v in kwargs.items()]
        )

class ExecMock1(Executor):
    def __init__(
            self,
            pack_or_exec,
            *exec_argv,
            **kwargs
    ) -> None:
        super().__init__(
            pack_or_exec,
            **kwargs
        )
        self.exec_argv = exec_argv

    def _argv(self, **kwargs):
        return [
            f"cmd{self.__class__.__name__}",
            *self.exec_argv,
            *[f"{self.__class__.__name__}[arg{i}]" for i in range(2)],
            *[f"{self.__class__.__name__}{k}:{v}" for k, v in kwargs.items()]
        ]

class ExecMock2(ExecMock1):
    pass

TEST_FOLDER = os.path.dirname(__file__)


def benchio():
    packs = _get_multipack(
        os.path.join(TEST_FOLDER, 'config', 'benchio.yaml'),
        base='/tmp',
        use_current_env=True,
    )

    _, pack = packs.packs.popitem()
    return pack


def mock_pack(pack):
    async def execute(*args, **kwargs):
        return [
            *args,
            *[f"{k}:{v}" for k, v in kwargs.items()]
        ]
    mock = pack
    mock.execute = execute
    return mock


def test_executor_argv():
    submock = ExecMock1(None, "a1", "a2")
    wrapmock = ExecMock2(submock, "a3")

    assert (
        wrapmock.argv() ==
        ["cmdExecMock2", "a3", "ExecMock2[arg0]", "ExecMock2[arg1]",
         "cmdExecMock1", "a1", "a2", "ExecMock1[arg0]", "ExecMock1[arg1]"]
    )

    submock = ExecMock2(None)
    wrapmock = ExecMock1(submock)

    assert (
        wrapmock.argv(k1="v1") ==
        ["cmdExecMock1", "ExecMock1[arg0]", "ExecMock1[arg1]", "ExecMock1k1:v1",
         "cmdExecMock2", "ExecMock2[arg0]", "ExecMock2[arg1]", "ExecMock2k1:v1"]
    )


def test_executor_kwargs():
    submock = ExecMock1(None, selfk1="sv1", selfk2="sv2")
    wrapmock = ExecMock2(submock, selfk1="sv1'", selfk3="sv3")
    kwargs = {"selfk2":"v2''", "selfk3":"v3''", "k4":"v4"}

    assert (
        sorted(wrapmock.kwargs().keys()) == ["selfk1", "selfk2", "selfk3"]
    )
    assert (
        sorted(wrapmock.kwargs().values()) == ["sv1'", "sv2", "sv3"]
    )
    assert (
        sorted(
            wrapmock.kwargs(**kwargs).keys()
        ) == ["k4", "selfk1", "selfk2", "selfk3"]
    )
    assert (
        sorted(
            wrapmock.kwargs(**kwargs).values()
        ) == ["sv1'", "v2''", "v3''", "v4"]
    )


def test_executor_execute():
    submock = ExecMock1(mock_pack(benchio()), "a1", selfk1="sv1")
    wrapmock = ExecMock2(submock, "a2", selfk2="sv2")

    assert (
        asyncio.run(wrapmock.execute(k3="v3")) ==
        [["cmdExecMock2", "a2", "ExecMock2[arg0]", "ExecMock2[arg1]",
          "cmdExecMock1", "a1", "ExecMock1[arg0]", "ExecMock1[arg1]",
          "selfk1:sv1", "selfk2:sv2", "k3:v3"]]
    )


def test_pack_executor():
    # voir is not setup so we are not receiving anything
    executor = PackExecutor(benchio(), "--start", "2", "--end", "20")

    acc = 0
    for r in proceed(executor.execute()):
        print(r)
        acc += 1

    assert acc == 2, "Only two message received"


def test_voir_executor():
    executor = PackExecutor(benchio(), "--start", "2", "--end", "20")
    voir = VoirExecutor(executor)

    acc = 0
    for r in proceed(voir.execute()):
        print(r)
        acc += 1

    assert  acc == 70


def test_timeout():
    executor = PackExecutor(benchio(), "--start", "2", "--end", "20", "--sleep", 20)
    voir = VoirExecutor(executor)
    timed = TimeOutExecutor(voir, delay=1)

    acc = 0
    for r in proceed(timed.execute()):
        print(r)
        acc += 1

    assert acc > 2 and acc < 70 


def test_njobs_executor():
    executor = PackExecutor(benchio(), "--start", "2", "--end", "20")
    voir = VoirExecutor(executor)
    njobs = NJobs(voir, 5)

    acc = 0
    for r in proceed(njobs.execute()):
        print(r)
        acc += 1

    assert acc == 70 * 5


def test_njobs_novoir_executor():
    executor = PackExecutor(benchio(), "--start", "2", "--end", "20")
    njobs = NJobs(executor, 5)

    acc = 0
    for r in proceed(njobs.execute()):
        print(r)
        acc += 1

    assert acc == 2 * 5


def test_void_executor():
    from milabench.executors import VoidExecutor
    
    plan = TimeOutExecutor(VoirExecutor(VoidExecutor()))
    
    for _ in proceed(plan.execute()):
        pass
