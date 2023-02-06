import pytest

from milabench.testing import milabench_cmd, config


expected_out = """
Error Report
------------

hf_t5.D0
^^^^^^^^
    ValueError: invalid literal for int() with base 10: 'xyz'

Summary
-------
    Success: 0
    Failures: 1
""".strip()


def test_error_reporting(capsys):

    with pytest.raises(SystemExit) as err:
        milabench_cmd("run", config("argerror"))

    assert err.type is SystemExit
    assert err.value.code == -1

    captured = capsys.readouterr()
    nchar = len(expected_out)
    ouput = captured.out.strip()[-nchar:]

    assert ouput[-nchar:] == expected_out
    assert captured.err == ""
