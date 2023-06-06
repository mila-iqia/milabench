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
    successes: int = 0
    failures: int = 0
    early_stop: bool = False
    times: dict = field(default_factory=dict)


def get_benchname(config):
    return config.get("name")


def get_per_gpu_key(config):
    jobid = config.get("job-number", "X")
    device = config.get("device", "Y")
    return f"N{jobid}-D{device}"


def drop_min_max(xs):
    xs = sorted(x for x in xs if x is not None)
    if len(xs) >= 5:
        xs = xs[1:-1]  # Remove min and max
    return xs


class ReportMachinePerf(ValidationLayer):
    """Generate the report live"""

    def __init__(self, print_live=False, **kwargs) -> None:
        super().__init__(**kwargs)

        self.ignored_metrics = {"task", "progress", "units"}
        self.accumulator = defaultdict(MetricAcc)
        self.config = None
        self.header_shown = False
        self.print_live = print_live
        self.preprocessor = drop_min_max
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
        if self.print_live:
            for acc in self.accumulator.values():
                self.show_bench(acc)

    def benchname(self):
        return self.config["name"]

    def on_start(self, entry):
        name = get_benchname(entry.pack.config)
        acc = self.accumulator[name]
        acc.started += 1
        acc.times[entry.tag] = entry.data["time"]

        if self.print_live and acc.started == acc.finished:
            self.show_bench(acc)

    def on_config(self, entry):
        self.config = entry.data

    def on_end(self, entry):
        name = get_benchname(entry.pack.config)
        group = self.groupkey(entry.pack.config)

        acc = self.accumulator[name]
        acc.finished += 1

        # Compute walltime
        start = acc.times[entry.tag]
        walltime = entry.data["time"] - start
        self.add_metric(name, group, "walltime", walltime)

        good = entry.data["return_code"] == 0 or acc.early_stop
        acc.successes += int(good)
        acc.failures += int(not good)

    def on_stop(self, entry):
        name = get_benchname(entry.pack.config)
        acc = self.accumulator[name]
        acc.early_stop = True

    def groupkey(self, config):
        """Key used to group observation of the same benchmark"""
        return get_per_gpu_key(config)

    def group_reduce(self, metric, stat, xs):
        """Combine group metrics to form an overal perf score for a given benchmark"""
        if stat in ("std", "sem"):
            return sum(np.power(xs, 2)) ** 0.5

        if metric in ("temperature", "memory", "loss", "load", "walltime"):
            return np.mean(xs)

        return sum(xs)

    def add_metric(self, bench, group, metric, value: float):
        """Add a single metric value"""
        acc = self.accumulator[bench]

        acc.name = bench
        acc.metrics[metric][group].append(value)

    def reduce(self, acc: MetricAcc):
        """Compute the score of a benchmark"""
        reduced = dict()

        for metric, groups in acc.metrics.items():
            reduced[metric] = dict()

            group_values = defaultdict(list)
            for _, values in groups.items():
                for stat, statfun in self.stats.items():
                    group_values[stat].append(statfun(self.preprocessor(values)))

            for stat in self.stats:
                reduced[metric][stat] = self.group_reduce(
                    metric, stat, group_values[stat]
                )

        return reduced

    def _backward_compat(self, summary):
        for k, metrics in summary.items():
            metrics["name"] = k
            metrics["train_rate"] = metrics.pop("rate", {})

    def summary(self):
        summary = dict()

        for k, acc in self.accumulator.items():
            result = self.reduce(acc)
            result["successes"] = acc.successes
            result["failures"] = acc.failures
            result["n"] = acc.successes + acc.failures

            summary[k] = result

        self._backward_compat(summary)
        return summary

    def show_bench(self, acc: MetricAcc, show_header=True):
        """Show metrics as a table"""
        if acc.shown:
            return

        acc.shown = True
        print(acc.name)

        reduced = self.reduce(acc)
        ordered = sorted(reduced.keys())

        header = []
        lines = []
        for metric in ordered:
            stats = reduced[metric]

            line = [f"{metric:>20}"]
            header = [f"{'name':>20}"]

            for stat, value in stats.items():
                line.append(f"{value:10.2f}")
                header.append(f"{stat:>10}")

            lines.append(" | ".join(line))

        if show_header and not self.header_shown:
            self.header_shown = True
            print(" | ".join(header))

        print("\n".join(lines))

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


class ReportGPUPerf(ReportMachinePerf):
    """Report performance per GPU"""

    def groupkey(self, config):
        return "all"
