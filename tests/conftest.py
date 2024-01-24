import os
from pathlib import Path

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
    os.environ["MILABENCH_CONFIG"] = "config/ci.yaml"
    os.environ["MILABENCH_BASE"] = "output"
    os.environ["MILABENCH_GPU_ARCH"] = "cuda"
    os.environ["MILABENCH_DASH"] = "no"


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
