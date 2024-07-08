
import milabench.commands.executors

import traceback
from pytest import fixture


@fixture
def args(standard_config, tmp_path):
    return [
        "--base", str(tmp_path),
        "--config", str(standard_config)
    ]


async def mock_exec(command, phase="run", timeout=False, timeout_delay=600, **kwargs):
    return [0]


def run_cli(*args):
    from milabench.cli import main

    print(" ".join(args))
    try:
        main(args)
    except SystemExit as exc:
        assert not exc.code


def test_milabench(monkeypatch, args):
    monkeypatch.setenv("MILABENCH_GPU_ARCH", "cuda")
    monkeypatch.setattr(milabench.commands, "execute_command", mock_exec)

    run_cli("install", *args)

    run_cli("prepare", *args)

    #
    # use Mock GPU-SMI
    #
    monkeypatch.setenv("MILABENCH_GPU_ARCH", "mock")
    from milabench.cli.dry import assume_gpu
    with assume_gpu(8):
        run_cli("run", *args, "--no-report")
