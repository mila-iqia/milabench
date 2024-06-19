import argparse
import os
import re
from dataclasses import dataclass, field

import pandas as pd

from ..common import _read_reports
from ..report import make_dataframe, pandas_to_string
from ..summary import make_summary


def default_tags():
    return [
        "worker=w([a-z0-9]*)",
        "multiple=m([0-9]*)",
        "power=p([0-9]*)",
        "capacity=c([A-Za-z0-9]*(Go)?)",
    ]


# fmt: off
@dataclass
class Arguments:
    runs: str
    tags: list = field(default_factory=default_tags)
# fmt: on


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=str,
        help="Run folder",
        default="/home/mila/d/delaunap/batch_x_worker/",
    )
    parser.add_argument(
        "--tags",
        type=str,
        help="Tags defined in run names",
        default=default_tags(),
    )
    return parser.parse_args()  # Arguments()


def get_config(reports):
    k = list(reports.keys())[0]
    config = None
    for line in reports[k]:
        if line["event"] == "config":
            config = line["data"]
            break
    return config


def extract_tags(name, tags):
    for tag, pat in tags.items():
        if m := pat.search(name):
            value = m.group(1)
            yield tag, value
        else:
            print(f"{tag} not found in {name}")
            yield tag, "NA"


def gather_cli(args=None):
    """Gather metrics from runs inside a folder in a neat format.
    It can extract tags/flags from the runname and create new columns to uniquely identify runs.

    Examples
    --------

    >>> python -m milabench.cli.gather --runs /home/mila/d/delaunap/batch_x_worker/
    bench  | fail |   n |       perf |   sem% |   std% | peak_memory |      score | weight | elapsed | name | worker | multiple | power | capacity
    brax   |    0 |   1 |  722480.33 |   0.7% |   5.2% |        6448 |  722480.33 |   1.00 |    94 | w16-m8-c4Go | 16 | 8 | NA | 4Go
    dlrm   |    0 |   1 |  350641.30 |   0.6% |   4.6% |        7624 |  350641.30 |   1.00 |   124 | w16-m8-c4Go | 16 | 8 | NA | 4Go
    ....
    brax   |    0 |   1 |  723867.42 |   0.6% |   4.5% |        6448 |  723867.42 |   1.00 |    94 | w2-m8-c8Go | 2 | 8 | NA | 8Go
    dlrm   |    0 |   1 |  403113.36 |   0.7% |   5.1% |        7420 |  403113.36 |   1.00 |   258 | w2-m8-c8Go | 2 | 8 | NA | 8Go
    bf16   |    0 |   8 |     293.08 |   0.3% |   7.5% |        5688 |    2361.09 |   0.00 |    18 | w2-m8-c8Go | 2 | 8 | NA | 8Go
    fp16   |    0 |   8 |     290.58 |   0.2% |   4.9% |        5688 |    2335.63 |   0.00 |    29 | w2-m8-c8Go | 2 | 8 | NA | 8Go

    """
    if args is None:
        args = arguments()

    runs = []
    for folder in os.listdir(args.runs):
        if folder.startswith("prepare"):
            continue

        if folder.startswith("install"):
            continue

        path = f"{args.runs}/{folder}"
        if os.path.isdir(path):
            runs.append(path)

    tags = dict()
    for tag in args.tags:
        name, regex = tag.split("=")
        tags[name] = re.compile(regex)

    query = ("batch_size", "elapsed")
    data = []
    for run in runs:
        reports = _read_reports(run)
        summary = make_summary(reports.values(), query=query)
        df = make_dataframe(summary, None, None, query=query)

        name = run.split("/")[-1]
        df["name"] = name.split(".", maxsplit=1)[0]
        for tag, value in extract_tags(name, tags):
            df[tag] = value

        data.append(df)

    gathered = pd.concat(data)
    print(pandas_to_string(gathered))


if __name__ == "__main__":
    gather_cli()
