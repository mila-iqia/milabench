from collections import defaultdict
from dataclasses import dataclass, field

from ..validation.validation import ValidationLayer

import numpy as np


@dataclass
class MetricAcc:
    name: str = None
    metrics: defaultdict = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    started: int = 0
    finished: int = 0
    shown: bool = False


def get_benchname(config):
    return config.get("name")


def get_per_gpu_key(config):
    jobid = config.get("job-number", "X")
    device = config.get("device", "Y")
    return f"N{jobid}-D{device}"


class Report(ValidationLayer):
    """Generate the report live"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.ignored_metrics = {"task", "progress", "units"}
        self.accumulator = defaultdict(MetricAcc)
        self.config = None
        self.stats = {
            "min": lambda xs: np.percentile(xs, 0),
            "q1": lambda xs: np.percentile(xs, 25),
            "median": lambda xs: np.percentile(xs, 50),
            "q3": lambda xs: np.percentile(xs, 75),
            "max": lambda xs: np.percentile(xs, 100),
            "mean": np.mean,
            "std": np.std,
            "sem": lambda xs: np.std(xs) / len(xs) ** 0.5,
        }

    def __exit__(self, *args, **kwargs):
        for acc in self.accumulator.values():
            self.show_bench(acc)

    def benchname(self):
        return self.config["name"]

    def on_start(self, entry):
        name = get_benchname(entry.pack.config)
        acc = self.accumulator[name]
        acc.started += 1

        if acc.started == acc.finished:
            self.show_bench(acc)

    def on_config(self, entry):
        self.config = entry.data

    def on_end(self, entry):
        name = get_benchname(entry.pack.config)
        acc = self.accumulator[name]
        acc.finished += 1

    def groupkey(self, config):
        return get_per_gpu_key(config)

    def group_reduce(self, metric, stat, xs):
        if stat == "std":
            return sum(np.power(xs, 2)) ** 0.5

        return sum(xs)

    def add_metric(self, bench, group, metric, value):
        acc = self.accumulator[bench]

        acc.name = bench
        acc.metrics[metric][group].append(value)

    def reduce(self, acc: MetricAcc):
        reduced = dict()

        for metric, groups in acc.metrics.items():
            reduced[metric] = dict()

            group_values = defaultdict(list)
            for _, values in groups.items():
                for stat, statfun in self.stats.items():
                    group_values[stat].append(statfun(values))

            for stat in self.stats:
                reduced[metric][stat] = self.group_reduce(
                    metric, stat, group_values[stat]
                )

        return reduced

    def show_bench(self, acc: MetricAcc):
        if acc.shown:
            return

        acc.shown = True
        print(acc.name)

        reduced = self.reduce(acc)

        for metric, stats in reduced.items():
            print(f"    {metric}")
            for stat, value in stats.items():
                print(f"        {stat:>10}: {value}")

    def on_data(self, entry):
        name = get_benchname(entry.pack.config)
        group = self.groupkey(entry.pack.config)

        for metric, v in entry.data.items():
            if metric in self.ignored_metrics:
                continue

            if metric == "gpudata":
                for _, data in v.items():
                    for m, v in data.items():
                        if m == "memory":
                            v = v[0]

                        self.add_metric(name, group, m, v)
            else:
                self.add_metric(name, group, metric, v)
