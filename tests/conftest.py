from pathlib import Path

import pytest

here = Path(__file__).parent


@pytest.fixture
def runs_folder():
    return here / "runs"


@pytest.fixture
def config():
    def get_config(name):
        return here / "config" / f"{name}.yaml"

    return get_config
