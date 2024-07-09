from contextlib import contextmanager
import os

import milabench.alt_async
import milabench.commands.executors
from milabench.testing import resolved_config

import pytest


def run_cli(*args):
    from milabench.cli import main

    print(" ".join(args))
    try:
        return main(args)
    except SystemExit as exc:
        assert not exc.code


def benchlist(enabled=True):
    standard = resolved_config("standard")

    for key, value in standard.items():
        if value.get("enabled", False):
            if key[0] != "_":
                yield key


# We want to reuse this fixtures for each bench
# so we do not run some steps multiple times
@pytest.fixture(scope='session')
def args(standard_config, module_tmp_dir):
    return [
        "--base", str(module_tmp_dir),
        "--config", str(standard_config)
    ]


def mock_voir_run(argv, info, timeout=None, constructor=None, env=None, **options):
    from voir.proc import Multiplexer
    mp = Multiplexer(timeout=timeout, constructor=constructor)
    mp.start(["sleep", "1"], info=info, env=env, **options)
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
def test_milabench(monkeypatch, args, bench, module_tmp_dir):
    #
    #   How do we check for a good run ?
    #
    from milabench.cli.dry import assume_gpu

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
