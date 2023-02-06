import pytest

from milabench.testing import milabench_cmd, config


expected_output = """"""


def test_error_reporting(capsys):

    with pytest.raises(SystemExit) as err:
        milabench_cmd("run", config("oom"))

    assert err.type is SystemExit
    assert err.value.code == -1

    captured = capsys.readouterr()
    assert captured.out == expected_output
