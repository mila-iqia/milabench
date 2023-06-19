from copy import deepcopy
from collections import defaultdict
import json
import os
from pathlib import Path

from milabench.structs import BenchLogEntry
from milabench.pack import BasePackage


class ReplayPackage(BasePackage):
    """Disable some folder creation for replay purposes"""

    def __init__(self, config, core=None):
        self.core = core
        self.config = config
        self.phase = None
        self.processes = []


def replay(filename):
    pack = None
    with open(filename, "r") as f:
        for line in f.readlines():
            try:
                entry = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Could not read `{line}` from {filename}") from e

            if entry["event"] == "config":
                pack = ReplayPackage(entry["data"])
                continue

            if pack is not None:
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


def show_diff(a, b, depth=0, path=None):
    indent = " " * depth
    acc = 0

    if depth == 0:
        print()

    if path is None:
        path = []

    for k in a.keys():
        p = deepcopy(path) + [k]

        v1 = a.get(k)
        v2 = b.get(k)

        if v1 is None or v2 is None:
            print(f"Missing key {k}")
            continue

        if isinstance(v1, dict):
            acc += show_diff(v1, v2, depth + 1, p)
            continue

        if v1 != v2:
            print(f'{indent} {".".join(p)} {v1} != {v2}')
            acc += 1

    return acc
