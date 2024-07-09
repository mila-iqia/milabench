from contextlib import contextmanager
import os

import milabench.alt_async
import milabench.commands.executors
from milabench.testing import resolved_config

import pytest


def run_cli(*args, expected_code=0, msg=None):
    from milabench.cli import main

    print(" ".join(args))
    try:
        return main(args)
    except SystemExit as exc:
        assert exc.code == expected_code, msg


def benchlist(enabled=True):
    standard = resolved_config("standard")

    for key, value in standard.items():
        if value.get("enabled", False):
            if key[0] != "_":
                yield key


def mock_voir_run(argv, info, timeout=None, constructor=None, env=None, **options):
    from voir.proc import Multiplexer
    mp = Multiplexer(timeout=timeout, constructor=constructor)
    mp.start(["sleep", "1"], info=info, env=env, **options)
    mp.buffer.append(
        # add perf data
        constructor(
            event="data",
            pipe="data",
            data={
                "rate": 10,
                "units": "item/s",
                "task": "train"
            },
            **info
        )
    )
    return mp


def count_file_like(path, name):
    try:
        acc = 0
        for file in os.listdir(path + "/runs"):
            if file.startswith(name):
                acc += 1
        return acc
    except FileNotFoundError:
        return 0


@contextmanager
def filecount_inc(path, name):
    """Check that a new file was created after running"""
    old = count_file_like(path, name)
    yield
    new = count_file_like(path, name)

    assert new == old + 1


@pytest.mark.parametrize("bench", benchlist())
def test_milabench(monkeypatch, bench, module_tmp_dir, standard_config):
    from milabench.cli.dry import assume_gpu

    args= [
        "--base", str(module_tmp_dir),
        "--config", str(standard_config)
    ]

    monkeypatch.setenv("MILABENCH_GPU_ARCH", "cuda")
    
    with filecount_inc(module_tmp_dir, "install"):
        run_cli("install", *args, "--select", bench)

    with filecount_inc(module_tmp_dir, "prepare"):
        run_cli("prepare", *args, "--select", bench)

    #
    # use Mock GPU-SMI
    #
    with monkeypatch.context() as ctx:
        ctx.setattr(milabench.alt_async, "voir_run", mock_voir_run)
        ctx.setenv("MILABENCH_GPU_ARCH", "mock")

        with filecount_inc(module_tmp_dir, bench):
            with assume_gpu(8):
                run_cli("run", *args, "--no-report", "--select", bench, "--run-name", str(bench))




ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
def cleanpath(out):
    return out.replace(str(ROOT), "$SRC")



def test_milabench_bad_install(monkeypatch, tmp_path, config, capsys, file_regression):
    args= [
        "--base", str(tmp_path),
        "--config", str(config("benchio_bad"))
    ]

    monkeypatch.setenv("MILABENCH_GPU_ARCH", "bad")
    
    # check that the return code is an error
    with filecount_inc(str(tmp_path), "install"):
        run_cli("install", *args, expected_code=1)

    # Check that the error was extracted
    all = capsys.readouterr()
    stdout = cleanpath(all.out)
    stdout = "\n".join(stdout.split("\n")[-15:-2])
    file_regression.check(stdout)


def test_milabench_bad_prepare(monkeypatch, tmp_path, config, capsys, file_regression):
    monkeypatch.setenv("MILABENCH_GPU_ARCH", "mock")
    args= [
        "--base", str(tmp_path),
        "--config", str(config("benchio_bad"))
    ]
    
    # check that the return code is an error
    with filecount_inc(str(tmp_path), "prepare"):
        run_cli("prepare", *args, expected_code=1)

    # Check that the error was extracted
    all = capsys.readouterr()
    stdout = cleanpath(all.out)
    stdout = "\n".join(stdout.split("\n")[-12:-2])
    file_regression.check(stdout)


def test_milabench_bad_run(monkeypatch, tmp_path, config, capsys, file_regression):
    from milabench.cli.dry import assume_gpu
    
    # Do a valid install to test a bad run
    with monkeypatch.context() as m:
        m.setenv("MILABENCH_GPU_ARCH", "ok")
        args = [
            "--base", str(tmp_path),
            "--config", str(config("benchio"))
        ]
        with filecount_inc(str(tmp_path), "install"):
            run_cli("install", *args)

    #
    # Do a bad run
    #
    monkeypatch.setenv("MILABENCH_GPU_ARCH", "mock")
    args = [
        "--base", str(tmp_path),
        "--config", str(config("benchio_bad"))
    ]
    
    # check that the return code is an error
    with filecount_inc(str(tmp_path), "run"):
        with assume_gpu(8):
            run_cli("run", *args, "--run-name", "run", expected_code=2)

    # Check that the error was extracted
    all = capsys.readouterr()
    stdout = cleanpath(all.out)
    stdout = "\n".join(stdout.split("\n")[-53:-2])
    file_regression.check(stdout)



def test_milabench_bad_run_before_install(monkeypatch, tmp_path, config, capsys, file_regression):
    from milabench.cli.dry import assume_gpu
    
    #
    # Do a bad run
    #
    monkeypatch.setenv("MILABENCH_GPU_ARCH", "mock")
    args = [
        "--base", str(tmp_path),
        "--config", str(config("benchio_bad"))
    ]
    
    #
    # NOTE: the exception happened IN milabench but in the asyncio part
    # which runs the benchmark
    #
    # Here milabench eat the exceptions and simply reports it as an error at the end of the run
    # so all the benchmarks might try to run as well

    # check that the return code is an error
    run_cli("run", *args, "--run-name", "run", "--no-report", expected_code=2)

    # Check that the error was extracted
    all = capsys.readouterr()
    stdout = cleanpath(all.out)
    stdout = "\n".join(stdout.split("\n")[-53:-2])
    file_regression.check(stdout)
