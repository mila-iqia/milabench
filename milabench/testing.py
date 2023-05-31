from collections import defaultdict
import json
import os
from pathlib import Path

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


def replay_run(folder):
    """Replay a run folder"""
    folder = Path(folder)

    files = sorted([folder / f for f in os.scandir(folder)])
    benches = defaultdict(list)

    for file in files:
        tags = str(file).split(".", maxsplit=1)
        name = tags[0]
        benches[name].append(file)

    for benchfiles in benches.values():
        yield from interleave(*benchfiles)
