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


    captured = capsys.readouterr()
    print("==")
    print(captured.out)
    print("==")

    print("==")
    print(captured.err)
    print("==")
    
    assert err.type is SystemExit
    assert err.value.code != 0

    assert matcher(captured.out), "The traceback need to be printed"
