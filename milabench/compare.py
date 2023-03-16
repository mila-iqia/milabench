import os
from dataclasses import dataclass
from datetime import datetime


@dataclass
class _Output:
    path: str
    name: str
    date: datetime
    summary: dict = None


def fetch_runs(folder):
    runs = []
    for run in os.listdir(folder):
        try:
            split = run.split(".", maxsplit=1)

            if len(split) == 2:
                name, date = split
            else:
                name = run
                date = datetime.fromtimestamp(
                    os.stat(os.path.join(folder, run)).st_mtime
                )

            out = _Output(
                os.path.join(folder, run),
                name,
                datetime.strptime(date, "%Y-%m-%d_%H:%M:%S.%f"),
            )
            runs.append(out)
        except ValueError:
            continue

    runs.sort(key=lambda out: out.date)
    return runs


def _print_headers(runs, sep):
    # Print Header
    times = [
        f"{'bench':<20}",
        f"{'metric':>15}",
    ]
    dates = [" " * (35 + len(sep))]
    header = [" " * (35 + len(sep))]
    for run in runs:
        if not run.summary:
            continue

        header.append(f"{run.name:>10}")
        dates.append(f"{str(run.date.date()):>10}")
        times.append(f"{str(run.date.time().strftime('%H:%M:%S')):>10}")

    line = sep.join(header)
    print(line)
    print(sep.join(dates))
    print(sep.join(times))
    print("-" * len(line))


def getmetric(bench, key):
    keys = key.split(".")
    parent = bench

    def maybeint(k):
        try:
            return int(k)
        except Exception:
            return k

    assert keys[0] in bench.keys(), f"{keys[0]} Choose from {list(bench.keys())}"

    for key in keys:
        if key in parent or (key := maybeint(key) and key in parent):
            parent = parent.get(key, dict())

    return parent


def compare(runs, last, metric, stat):
    sep = " | "

    if last is not None:
        runs = runs[-last:]

    benches = set()
    for run in runs:
        if run.summary:
            benches.update(run.summary.keys())

    benches = sorted(list(benches))

    _print_headers(runs, sep)

    for bench in benches:
        line = [
            f"{bench:<20}",
            f"{metric:>15}",
        ]

        for run in runs:
            if not run.summary:
                continue

            value = getmetric(run.summary.get(bench, {}), metric).get(
                stat, float("NaN")
            )

            line.append(f"{value:10.2f}")

        print(sep.join(line))
