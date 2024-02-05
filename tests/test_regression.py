import os
from pathlib import Path

import pytest

from milabench.cli.dry import cli_dry, arguments

milabench_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
milabench_base = os.path.join(milabench_src, "output")
user_home = str(Path.home())


@pytest.mark.skipif(not Path(milabench_base).exists(), reason="Milabench is not installed")
def test_command_regression(capsys, file_regression):
    
    old = os.getenv("MILABENCH_BASE")
    os.environ["MILABENCH_BASE"] = milabench_base
    
    args = arguments()
    args.withenv = False
    cli_dry(args)
    
    all = capsys.readouterr()
    stdout = all.out
    assert stdout != ""

    stdout = stdout.replace(str(milabench_base), "$MILABENCH_BASE")
    stdout = stdout.replace(str(milabench_src), "$MILABENCH_SRC")
    stdout = stdout.replace(str(user_home), "$HOME")
    
    file_regression.check(stdout)
    os.environ["MILABENCH_BASE"] = old
