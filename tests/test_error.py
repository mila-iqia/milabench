import re
import pytest

from milabench.testing import milabench_cmd, config


expected_out_short = """
Error Report
------------
(.*)
hf_t5.D0
^^^^^^^^
    ValueError: invalid literal for int() with base 10: 'xyz'
(.*)
Summary
-------
    Success: 0
    Failures: 1
""".strip()

expected_out_long = """
Error Report
------------
(.*)
hf_t5.D0
^^^^^^^^
    Traceback (most recent call last):
    File "(.*)", line 2476, in _get_value
        result = type_func(arg_string)
    ValueError: invalid literal for int() with base 10: 'xyz'
(.*)
Summary
-------
    Success: 0
    Failures: 1
""".strip()

error_cases = [([], expected_out_short), (["--fulltrace"], expected_out_long)]


def make_regex(pattern):
    frags = []

    for pat in pattern.strip().split("(.*)"):
        frags.append(re.escape(pat))

    return "(.*)" + "(.*)".join(frags) + "(.*)"


@pytest.mark.parametrize("args,expected_out", error_cases)
def test_error_reporting_short(capsys, args, expected_out):

    with pytest.raises(SystemExit) as err:
        milabench_cmd("run", config("argerror"), *args)

    assert err.type is SystemExit
    assert err.value.code == -1

    expected_pat = re.compile(
        make_regex(expected_out),
        flags=re.MULTILINE | re.DOTALL,
    )

    captured = capsys.readouterr()
    assert re.search(expected_pat, captured.out)
    assert captured.err == ""
