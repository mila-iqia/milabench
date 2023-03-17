from argparse import Namespace
import os

import pytest

import milabench.evalctx as evalctx
from milabench.evalctx import ArgumentResolver


@pytest.fixture
def fake_run():
    return {
        "name": "fake_bench",
        "mem": dict(
            slope=2,
            intercept=10,
            multiple=8,
        ),
    }


@pytest.fixture
def fake_run_2():
    return {
        "name": "fake_bench",
        "mem": dict(
            slope=20,
            intercept=40,
            multiple=4,
        ),
    }


@pytest.fixture
def fetch_gpu_config(monkeypatch):
    def mock(*args):
        return {"mem": 12000, "count": 2}

    monkeypatch.setattr(evalctx, "fetch_gpu_configuration", mock)


def test_eval_context(fetch_gpu_config, fake_run):
    args = {"a": "$(bs(gpu, mem))"}
    resolver = ArgumentResolver(fake_run)
    resolver.resolve_arguments(args)

    assert int(args["a"]) % 8 == 0, "Batch size is a multiple of 8"
    assert int(args["a"]) == 5992


def test_eval_context_multi_gpu(fetch_gpu_config, fake_run):
    args = {"a": "$(bs(gpu, mem, multi_gpu=True))"}
    resolver = ArgumentResolver(fake_run)
    resolver.resolve_arguments(args)

    assert int(args["a"]) % 8 == 0, "Batch size is a multiple of 8"
    assert int(args["a"]) == 5992 * 2


def test_eval_context_2(fetch_gpu_config, fake_run_2):
    args = {"a": "$(bs(gpu, mem))"}
    resolver = ArgumentResolver(fake_run_2)
    resolver.resolve_arguments(args)

    assert int(args["a"]) % 4 == 0, "Batch size is a multiple of 4"
    assert int(args["a"]) == 596


def test_eval_context_env_override(fake_run):
    args = {"a": "$(bs(gpu, mem))"}

    os.environ["MILABENCH_GPU_MEM_LIMIT"] = "6000"

    resolver = ArgumentResolver(fake_run)
    resolver.resolve_arguments(args)

    assert int(args["a"]) % 8 == 0, "Batch size is a multiple of 8"
    assert int(args["a"]) == 2992


def test_eval_context_missing(fetch_gpu_config):
    args = {"a": "$(bs(gpu, mem))"}

    with pytest.raises(AttributeError) as err:
        resolver = ArgumentResolver(dict(name="fake-run"))
        resolver.resolve_arguments(args)

    assert (
        str(err.value)
        == "fake-run configuration does not provide a `mem` configuration"
    )
