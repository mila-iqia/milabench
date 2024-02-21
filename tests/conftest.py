import os
from pathlib import Path

from milabench.cli.dry import assume_gpu
import voir.instruments.gpu as voirgpu

import pytest


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


@pytest.fixture
def replayfolder():
    return here / "replays"



@pytest.fixture(scope="session", autouse=True)
def set_env():
    backend = voirgpu.deduce_backend()
    if backend == "cpu":
        backend = "mock"

    os.environ["MILABENCH_CONFIG"] = "config/ci.yaml"
    os.environ["MILABENCH_BASE"] = "output"
    os.environ["MILABENCH_DASH"] = "no"
    os.environ["MILABENCH_GPU_ARCH"] = backend

    mock = False
    if backend == "mock":
        mock = True

    with assume_gpu(enabled=mock):
        yield

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
