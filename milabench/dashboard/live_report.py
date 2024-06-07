import json
import os
import pprint
import shlex
import time
from collections import defaultdict
from datetime import datetime

from blessed import Terminal
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from .fs import XPath
from ..summary import make_summary_from_aggregates, Aggregator

T = Terminal()
color_wheel = [T.cyan, T.magenta, T.yellow, T.red, T.green, T.blue]


class LiveReportFormatter(DashFormatter):
    def __init__(self):
        super().__init__()
        self.running_set = defaultdict(int)
        self.aggregator_set = defaultdict(lambda: defaultdict(Aggregator))
        self.prune_delay = 0

    def __call__(self, entry):
        benchname = entry.tag.split('.')[0]

        agg = self.aggregator_set[benchname][entry.tag]
        agg.event_aggregator(entry)

        super().__call__(entry)

    def on_start(self, entry, data, row):
        self.running_set[benchname] += 1

    def on_end(self, entry, data, row):
        super().on_end(entry, data, row)

        benchname = entry.tag.split('.')[0]
        self.running_set[benchname] -= 1

        if self.running_set[benchname] == 0:
            self.produce_report_line(benchname)

    def produce_report_line(self, benchname):
        aggregators = self.aggregator_set.pop(benchname)
        summary = make_summary_from_aggregates([agg.group_by() for agg in aggregators])
        df = make_dataframe(summary, None, None)
        print(x.to_string(formatters=_formatters))