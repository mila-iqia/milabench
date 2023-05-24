import json

from milabench.utils import validation, multilogger
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


def replay_scenario(folder, name, filename=None):
    with multilogger(*validation(name)) as log:
        for entry in replay(folder / f"{filename or name}.txt"):
            log(entry)
    return log


def test_nan_layer(replayfolder):
    log = replay_scenario(replayfolder, "nan")
    assert log.result() != 0


def test_planning_layer(replayfolder):
    log = replay_scenario(replayfolder, "planning")
    assert log.result() != 0


def test_usage_layer_no_usage(replayfolder):
    log = replay_scenario(replayfolder, "usage", "no_usage")
    assert log.result() != 0


def test_usage_layer_usage(replayfolder):
    log = replay_scenario(replayfolder, "usage")
    assert log.result() == 0
