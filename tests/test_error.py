import re
import pytest

from milabench.testing import milabench_cmd, config


def short_matcher(output):
    """Check that the last line of the traceback is found"""
    return "ValueError: invalid literal for int() with base 10: 'xyz'" in output


def long_matcher(output):
    """Check that at least another line of the traceback is printed as well"""
    return short_matcher(output) and "result = type_func(arg_string)" in output


error_cases = [([], short_matcher), (["--fulltrace"], long_matcher)]


@pytest.mark.parametrize("args,matcher", error_cases)
def test_error_reporting_short(capsys, args, matcher):

    with pytest.raises(SystemExit) as err:
        milabench_cmd("run", config("argerror"), *args)

    assert err.type is SystemExit
    assert err.value.code == -1

    captured = capsys.readouterr()
    assert matcher(captured.out), "The traceback need to be printed"
    assert captured.err == ""
