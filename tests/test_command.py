import os
import re

from milabench.cli import main
import milabench.multi

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

VOIR_MD5_REGEX = r"(\.D?\d-)([a-z0-9]*)(\.json)"


def mock_gpu_list():
    return [
        {
            "device": 0,
            "selection_variable": "CUDA_VISIBLE_DEVICE"
        },
        {
            "device": 1,
            "selection_variable": "CUDA_VISIBLE_DEVICE"
        }
    ]


def test_command_regression(standard, capsys, file_regression, monkeypatch):
    monkeypatch.setattr(milabench.multi, "gpus", mock_gpu_list())
    
    try:
        main(["dry", "--config", str(standard), "--base", "/opt/milabench"])
    except SystemExit as exc:
        assert not exc.code

    output = capsys.readouterr().out
    output = (
        # VOIR_MD5_REGEX has 3 capturing group
        # The MD5 is the 2 second one which we replace
        # we put back the other 2
        re.sub(VOIR_MD5_REGEX, r"\1XXX\3", output)
        .replace(root, '/ROOT')
    )
    
    file_regression.check(output)
