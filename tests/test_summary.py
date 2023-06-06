from milabench.validation.validation import Summary
from milabench.cli import main


def test_report(runs_folder, capsys, file_regression):
    folder = runs_folder / "rijubigo.2023-03-24_13:45:27.512446"
    try:
        main(["report", "--runs", folder])
    except SystemExit as exc:
        assert not exc.code

    output = capsys.readouterr().out
    output = output.replace(str(folder), "XXX")
    file_regression.check(output)


def test_summary(file_regression):
    benchs = ["matmult", "matsub"]
    points = [
        "1. Errors stuff happened",
        "2. Errors stuff happened",
        "3. Errors stuff happened",
    ]
    report = Summary()

    with report.section("Errors"):
        for bench in benchs:
            with report.section(bench):
                for p in points:
                    report.add(p)

    output = ""

    def get_output(data):
        nonlocal output
        output = data

    report.show()
    report.show(get_output)
    file_regression.check(output)


def test_empty_summary():
    points = [
        "1. Errors stuff happened",
        "2. Errors stuff happened",
        "3. Errors stuff happened",
    ]
    report = Summary()

    with report.section("Errors"):
        with report.section("Bench"):
            pass

    output = ""

    def get_output(data):
        nonlocal output
        output = data

    report.show(get_output)

    assert output.strip() == ""


def test_report_folder_does_average(runs_folder, capsys, file_regression):
    try:
        main(["report", "--runs", runs_folder])
    except SystemExit as exc:
        assert not exc.code

    output = capsys.readouterr().out
    output = output.replace(str(runs_folder), "XXX")
    file_regression.check(output)


def test_compare(runs_folder, capsys, file_regression):
    try:
        main(["compare", runs_folder])
    except SystemExit as exc:
        assert not exc.code

    output = capsys.readouterr().out
    output = output.replace(str(runs_folder), "XXX")
    file_regression.check(output)


# ---
import json
from milabench.structs import BenchLogEntry
from milabench.pack import BasePackage


class ReplayPackage(BasePackage):
    def __init__(self, config, core=None):
        self.core = core
        self.config = config
        self.phase = None
        self.processes = []


def replay(filename):
    with open(filename, "r") as f:
        pack = None

        for line in f.readlines():
            entry = json.loads(line)

            if entry["event"] == "config":
                pack = ReplayPackage(entry["data"])
                continue

            if pack is not None:
                yield BenchLogEntry(pack, **entry)


from collections import defaultdict
import os
from pathlib import Path


def replay_run(folder):
    """Replay a run folder"""
    folder = Path(folder)

    files = sorted([folder / f for f in os.scandir(folder)])
    benches = defaultdict(list)

    for file in files:
        if str(file).endswith(".data"):
            tags = str(file).split(".", maxsplit=1)
            name = tags[0]
            benches[name].append(file)

    for benchfiles in benches.values():
        for bench in benchfiles:
            yield from replay(bench)


from copy import deepcopy


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


def test_make_summary(runs_folder):
    from milabench.reports.report import ReportMachinePerf, ReportGPUPerf
    import json

    run = runs_folder / "MI250.2023-05-08_17_54_51.224604"

    with ReportMachinePerf() as log:
        for event in replay_run(run):
            log(event)
            
    new = log.summary()
    
    
    from milabench.cli import _read_reports, make_summary
    runs = [run]
    reports = _read_reports(*runs)
    summary = make_summary(reports.values())
    
    show_diff(new, summary)
    
    with open('new.json', 'w') as f:
        json.dump(new, f, indent=2)

    with open('old.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    # print(json.dumps(log.summary(), indent=2))

    # with ReportGPUPerf() as log:
    #     for event in replay_run(run):
    #         log(event)

    # print(json.dumps(log.summary(), indent=2))
