import pytest

from milabench.cli import main


def short_matcher(output):
    """Check that the last line of the traceback is found"""
    return "ZeroDivisionError: division by zero" in output


def long_matcher(output):
    """Check that at least another line of the traceback is printed as well"""
    return short_matcher(output) and "give(loss=1 / x)" in output


error_cases = [([], short_matcher), (["--fulltrace"], long_matcher)]


@pytest.mark.parametrize("args,matcher", error_cases)
def test_error_reporting_short(capsys, args, matcher, config):
    with pytest.raises(SystemExit) as err:
        main(["run", "--config", config("argerror"), *args])

    assert err.type is SystemExit
    assert err.value.code == -1

    captured = capsys.readouterr()
    assert matcher(captured.out), "The traceback need to be printed"
