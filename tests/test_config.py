import os

from milabench.testing import config


def test_config():
    s = config("argerror")
    assert isinstance(s, str)
    assert os.path.exists(s)
