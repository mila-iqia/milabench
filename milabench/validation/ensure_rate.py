from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

from .validation import ValidationLayer, BenchLogEntry


class _Layer(ValidationLayer):
    """Makes sure the training rate is generated for each benchmarks"""

    def __init__(self, **kwargs) -> None:
        self.rates = defaultdict(float)
        self.errors = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def on_event(self, entry: BenchLogEntry):
        tag = entry.tag

        if tag not in self.rates:
            self.rates[tag] = 0

        if entry.event == "data":
            self.rates[tag] += entry.data.get("rate", 0)

        if entry.event == "end":
            self.errors += self.rates[tag] <= 0

    def report(self, summary, short=True, **kwargs):
        for tag, rate in self.rates.items():
            if rate > 0:
                continue

            with summary.section(tag):
                summary.add("* no training rate retrieved")

        self.set_error_code(self.errors)
        return self.errors
