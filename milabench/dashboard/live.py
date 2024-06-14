import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..validation.validation import ValidationLayer


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


def _get(dictionary, *path, default=float("nan")):
    s = dictionary
    for p in path[:-1]:
        s = s.get(p, {})
    return s.get(path[-1], default)


class ReportMachinePerf(ValidationLayer):
    """Generate the report live"""

    def __init__(self, print_live=False, count=None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.ignored_metrics = {"task", "progress", "units"}
        self.metrics = {
            "perf": lambda stats: _get(stats, "rate", "mean"),
            "std%": lambda stats: _get(stats, "rate", "std")
            * 100
            / _get(stats, "rate", "mean"),
            "sem%": lambda stats: _get(stats, "rate", "sem")
            * 100
            / _get(stats, "rate", "mean"),
            "peak_memory": lambda stats: _get(stats, "memory", "max"),
        }
        self.accumulator = defaultdict(MetricAcc)

        self.current_bench = None
        self.current_group = None
        self.current_acc = None
        self.prev_acc = None

        self.update_line = False
        self.config = None
        self.header_shown = False
        self.bench_count = None
        self.print_live = print_live
        self.time = time.time()
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
        print()
        print(self.show_header())

    def show_pending(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass
        # if self.print_live:
        #     for acc in self.accumulator.values():
        #         self.show_bench(acc)

    def benchname(self):
        return self.config["name"]

    def on_start(self, entry):
        acc = self.current_acc
        acc.started += 1
        acc.times[entry.tag] = entry.data["time"]

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

        if acc.finished == acc.started:
            self.update_line = False
            self.bench_finished(name, acc)

    def bench_finished(self, name, acc):
        print("\r", self.bench_line(acc), " " * 10)
        # self.dump_metric_table(acc)
        del self.accumulator[name]

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

    def show_header(self):
        header = [f' {"name":>30}', "  n", "fail"]

        for m in self.metrics:
            header.append(f"{m:>10}")

        return " | ".join(header)

    def bench_line(self, acc: MetricAcc):
        reduced = self.reduce(acc)
        line = [f"{acc.name:>30}", f"{acc.started:3d}", f"{acc.failures:4d}"]

        for metric, fun in self.metrics.items():
            value = fun(reduced)
            line.append(f"{value:10.2f}")

        return " | ".join(line)

    def dump_metric_table(self, acc: MetricAcc, show_header=True):
        """Show metrics as a table"""
        if acc.shown:
            return

        acc.shown = True

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

    def on_event(self, entry):
        self.prev_acc = self.current_acc
        self.current_bench = get_benchname(entry.pack.config)
        self.current_acc = self.accumulator[self.current_bench]
        self.current_acc.name = self.current_bench
        self.current_group = self.groupkey(entry.pack.config)
        self.update_line = True

        super().on_event(entry)

        if self.update_line:
            self.show_progress_line()

    def show_progress_line(self):
        now = time.time()
        if now - self.time > 0.1:
            line = self.bench_line(self.current_acc)
            print(f"\r{line}", end="")
            self.time = now

    def on_data(self, entry):
        name = self.current_bench
        group = self.current_group

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

    def report(self, summary):
        return 0


class ReportGPUPerf(ReportMachinePerf):
    """Report performance per GPU"""

    def groupkey(self, config):
        return "all"


class LivePrinter:
    pass
