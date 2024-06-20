import json
import os
import time
import zipfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from milabench.pack import BasePackage
from milabench.structs import BenchLogEntry
from milabench.utils import multilogger, validation_layers


class ReplayPackage(BasePackage):
    """Disable some folder creation for replay purposes"""

    def __init__(self, config, core=None):
        self.core = core
        self.config = config
        self.phase = None
        self.processes = []


def replay(filename, open_fun=open):
    pack = None
    with open_fun(filename, "r") as f:
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


def interleave(*filenames, open_fun=open):
    """Interleaves replay lines to simulate multiple bench sending events in parallel"""
    generators = [replay(file, open_fun=open_fun) for file in filenames]

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
    files = []

    for file in os.scandir(folder):
        if file.name.endswith(".data"):
            files.append(folder / file)
        else:
            print(f"Skiping {file.name}")

    files = sorted(files)
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


def replay_zipfile(path, *validation, sleep=0):
    with zipfile.ZipFile(path, "r") as archive:
        data = defaultdict(list)
        total = 0

        for member in archive.namelist():
            if member.endswith(".data"):
                filename = member.split("/")[-1]
                benchname = filename.split(".", maxsplit=1)[0]
                data[benchname].append(member)
                total += 1

        from milabench.config import set_run_count

        set_run_count(total)

        with multilogger(*validation) as log:
            for _, streams in data.items():
                gen = interleave(*streams, open_fun=archive.open)

                for entry in gen:
                    time.sleep(sleep)
                    log(entry)
                    # callback(entry)

        return log


def replay_validation_scenario(folder, *validation, filename=None):
    """Replay events from a data file or folder"""
    gen = None

    path = folder / filename
    file = str(path) + ".txt"

    if os.path.isdir(path):
        files = [path / f for f in os.scandir(path)]
        gen = interleave(*files)

    if os.path.isfile(file):
        gen = replay(file)

    with multilogger(*validation) as log:
        for entry in gen:
            log(entry)

    return log


def replay_scenario(folder, name, filename=None):
    """Replay events from a data file or folder"""
    return replay_validation_scenario(
        folder, *validation_layers(name), filename=filename or name
    )
