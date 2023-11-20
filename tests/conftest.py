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


@pytest.fixture
def multipack(config, tmp_path):
    from milabench.cli import _get_multipack

    bench_config = config("benchio")
    system_path = config("system")
    base = tmp_path

    use_current_env = True
    select = None
    exclude = None
    run_name = "test"
    overrides = {}

    return _get_multipack(
        bench_config,
        system_path,
        base,
        use_current_env,
        select,
        exclude,
        run_name=run_name,
        overrides=overrides,
    )
