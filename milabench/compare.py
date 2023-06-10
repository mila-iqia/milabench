import os
from dataclasses import dataclass
from datetime import datetime


@dataclass
class _Output:
    path: str
    name: str
    date: datetime
    summary: dict = None


def retrieve_datetime_from_name(date):
    """Windows does not support : in folders/filenames"""
    formats = ["%Y-%m-%d_%H:%M:%S.%f", "%Y-%m-%d_%H_%M_%S.%f"]
    for fmt in formats:
        try:
            return datetime.strptime(date, fmt)
        except ValueError:
            pass


def fetch_runs(folder):
    runs = []
    for run in os.listdir(folder):
        pth = os.path.join(folder, run)
        if not os.path.isdir(pth):
            continue
        if "." in run:
            name, date = run.split(".", maxsplit=1)
            date = retrieve_datetime_from_name(date)
        else:
            name = run

        if date is None:
            date = datetime.fromtimestamp(os.path.getmtime(pth))

        out = _Output(pth, name, date)
        runs.append(out)

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
        header.append(f"{run.name:>10}")
        dates.append(f"{str(run.date.date()):>10}")
        times.append(f"{str(run.date.time().strftime('%H:%M:%S')):>10}")

    line = sep.join(header)
    print(line)
    print(sep.join(dates))
    print(sep.join(times))
    print("-" * len(line))


def compare(runs, last, metric, stat):
    sep = " | "

    if last is not None:
        runs = runs[-last:]

    benches = set()
    for run in runs:
        benches.update(run.summary.keys())

    benches = sorted(list(benches))

    _print_headers(runs, sep)

    for bench in benches:
        line = [
            f"{bench:<20}",
            f"{metric:>15}",
        ]

        for run in runs:
            value = run.summary.get(bench, {}).get(metric, {}).get(stat, float("NaN"))
            line.append(f"{value:10.2f}")

        print(sep.join(line))
