import json
import os

import voir.instruments.gpu

from milabench.utils import validation_layers, multilogger
from milabench.structs import BenchLogEntry
from milabench.pack import BasePackage


def replay(filename):
    with open(filename, "r") as f:
        for line in f.readlines():
            entry = json.loads(line)

            if entry["event"] == "config":
                pack = BasePackage(entry["data"])
                continue

            yield BenchLogEntry(pack, **entry)


def interleave(*filenames):
    """Interleaves replay lines to simulate multiple bench sending events in parallel"""
    generators = [replay(file) for file in filenames]

    while len(generators) > 0:
        finished = []

        for gen in generators:
            try:
                yield next(gen)
            except StopIteration:
                finished.append(gen)

        for gen in finished:
            generators.remove(gen)


def replay_scenario(folder, name, filename=None):
    """Replay events from a data file or folder"""
    gen = None

    path = folder / f"{filename or name}"
    file = str(path) + ".txt"

    if os.path.isdir(path):
        files = [path / f for f in os.scandir(path)]
        gen = interleave(*files)

    if os.path.isfile(file):
        gen = replay(file)

    with multilogger(*validation_layers(name)) as log:
        for entry in gen:
            log(entry)

    return log


def test_error_layer(replayfolder):
    log = replay_scenario(replayfolder, "error")
    assert log.result() != 0


def test_error_layer_early_stop(replayfolder):
    log = replay_scenario(replayfolder, "error", "early_stop")
    assert log.result() == 0


def test_error_layer_early_stop_per_gpu(replayfolder):
    log = replay_scenario(replayfolder, "error", "early_stopping_per_gpu")
    assert log.result() == 0


def test_nan_layer(replayfolder):
    log = replay_scenario(replayfolder, "nan")
    assert log.result() != 0


def test_usage_layer_no_usage(replayfolder):
    log = replay_scenario(replayfolder, "usage", "no_usage")
    assert log.result() != 0


def test_usage_layer_usage(replayfolder):
    log = replay_scenario(replayfolder, "usage")
    assert log.result() == 0


def test_rate_layer(replayfolder):
    log = replay_scenario(replayfolder, "ensure_rate")
    assert log.result() != 0


def test_planning_layer_njobs_good(replayfolder):
    # Expected 3 jobs got 3 jobs
    log = replay_scenario(replayfolder, "planning", "planning_njobs_good")
    assert log.result() == 0


def test_planning_layer_njobs_bad(replayfolder):
    # Expected 3 jobs got 1 job
    log = replay_scenario(replayfolder, "planning", "planning_njobs_bad")
    assert log.result() != 0


def mock_gpu_info():
    return {
        "arch": "rocm",
        "gpus": [0, 1],
    }


def test_planning_layer_per_gpu_good(replayfolder, monkeypatch):
    # 2 GPU detected; expected 2 jobs got 2 jobs
    monkeypatch.setattr(voir.instruments.gpu, "get_gpu_info", mock_gpu_info)

    log = replay_scenario(replayfolder, "planning", "planning_per_gpu_good")
    assert log.result() == 0


def test_planning_layer_per_gpu_bad(replayfolder, monkeypatch):
    # 2 GPU detected; expected 2 jobs got 1 job
    monkeypatch.setattr(voir.instruments.gpu, "get_gpu_info", mock_gpu_info)

    log = replay_scenario(replayfolder, "planning", "planning_per_gpu_bad")
    assert log.result() != 0
