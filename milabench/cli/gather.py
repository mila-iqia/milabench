import argparse
import os
import re
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import pandas as pd

from ..common import _read_reports, _parse_report
from ..report import make_dataframe, pandas_to_string
from ..summary import make_summary
from ..fs import XPath

def default_tags():
    return [
        # Run Tgs
        "clock=g([0-9]+)",
        "power=p([0-9]+)",
        "observation=o([0-9]+)",

        # Bench Tags
        "concurrency=conc([0-9]+)",
        "max_context=mxctx([0-9]+)",
        "max_batch_token=mxbt([0-9]+)",
        "worker=w([0-9]+)",
        "multiple=m([0-9]+)",
        "batch_power=bp([0-9]+)",
        "capacity=c([0-9]+(Go)?)",
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
        nargs="+",
        default=default_tags(),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.csv"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--more",
        type=str,
        help="Tags defined in run names",
        nargs="+",
        default="",
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


def extract_tags(name, tags, found):
    for tag, pat in tags.items():
        if m := pat.search(name):
            value = m.group(1)
            found[tag] += 1
            yield tag, value
        # else:
        #     print(f"{tag} not found in {name}")
        #     yield tag, "NA"



def make_tags(tags_def):
    tags = dict()
    for tag in tags_def:
        name, regex = tag.split("=")
        tags[name] = re.compile(regex)
    return tags


def matrix_run():
    runs = []
    for folder in os.listdir(args.runs):
        if folder.startswith("prepare"):
            continue

        if folder.startswith("install"):
            continue

        path = f"{args.runs}/{folder}"
        if os.path.isdir(path):
            runs.append(path)

    found_tags = defaultdict(int)
    tags = make_tags(args.args)


    query = ("batch_size", "elapsed")
    data = []
    for run in runs:
        reports = _read_reports(run)
        summary = make_summary(reports, query=query)
        df = make_dataframe(summary, None, None, query=query)

        name = run.split("/")[-1]
        df["name"] = name.split(".", maxsplit=1)[0]
        for tag, value in extract_tags(name, tags, found_tags):
            if value != "NA":
                df[tag] = value

        data.append(df)

    gathered = pd.concat(data)
    print(pandas_to_string(gathered))


def flatten_values(payload, namespace=None):
    if namespace is None:
        namespace = tuple()

    if isinstance(payload, list):
        for i, val in enumerate(payload):
            yield from flatten_values(val, namespace + (str(i),))

    elif isinstance(payload, dict):
        for k, v in payload.items():
            nspace = namespace + (str(k),)
            yield from flatten_values(v, nspace)

    else:
        yield ".".join(namespace), payload   


def process_file(parent, file, tags, more, skip=None):
    if not file.endswith(".data"):
        return []

    if skip is None:
        skip = set()

    parent = XPath(parent)
    pth =   parent / file
    found_tags = defaultdict(int)

    filename_tags = { k: v
        for k, v in extract_tags(file, tags, found_tags)
    }

    runname_tags = { k: v
        for k, v in extract_tags(parent.name, tags, found_tags)
    }

    data = {
        **runname_tags,
        **filename_tags
    }

    name, _, device = file[:-5].partition('.')

    data["bench"] = name
    data["device"] = device
    data["run_id"] = hashlib.sha256(str(pth).encode("utf-8")).hexdigest()[:8]

    report = _parse_report(pth, shared=data)   

    lines = []
    for line in report:
        match line["event"]:
            case "data":
                payload = line["data"]

                unit = payload.pop("unit", None)
                units = payload.pop("units", None)
                time = payload.pop("time", None)
                task = payload.pop("task", None)

                for k, v in flatten_values(payload, tuple([])):
                    if k.startswith("progress"):
                        continue

                    line = {
                        "metric": k, 
                        "value": v, 
                        "time": time, 
                        "unit": units or unit, 
                        "task": task, 
                        **data, 
                        **more
                    }
                    lines.append(line)
    return lines



def gather_matrix_cli(args=None):
    if args is None:
        args = arguments()

    tags = dict()
    for tag in args.tags:
        name, regex = tag.split("=")
        tags[name] = re.compile(regex)

    all_data = []
    from milabench.common import _parse_report, XPath

    metrics = [
        "rate",
        "output_tok",
        "input_tok",
        "tpot",
        "itl",
        "e2els",
        "ttfts",
    ]

    for parent, _, filenames in os.walk(args.runs):
        for file in filenames:
            if not file.endswith(".data"):
                continue
    
            pth = XPath(parent) / file
            tag_values = dict(extract_tags(file, tags))

            for event in _parse_report(pth):
                match event["event"]:
                    case "data": 
                        for m in metrics:
                            if m in event["data"]:
                                all_data.append({
                                    "bench": str(file),
                                    **tag_values,
                                    "value": event["data"][m],
                                    "name": m
                                })

    dump = pandas_to_string(pd.DataFrame(all_data))

    with open("result.txt", "w") as fp:
        fp.write(dump)

    print(dump)



def gather_multi_run_cli(args=None):
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

    print(runs)
    found_tags = defaultdict(int)
    tags = dict()
    for tag in args.tags:
        name, regex = tag.split("=")
        tags[name] = re.compile(regex)

    more = dict(more.split("=") for more in args.more)
    lines = []
    folders = os.listdir(args.runs)

    if len(folders) == 0:
        folders = [""]

    lines = []
    for folder in folders:
        lines.extend(process_file(args.runs, folder, tags, more))

        for parent, _, filenames in os.walk(os.path.join(args.runs, folder)):
            for file in filenames:
                lines.extend(process_file(parent, file, tags, more))


    # print(lines)

    df = pd.DataFrame(lines)

    metric = df["metric"]

    gpu_match = metric.str.extract(r'^gpudata\.(\d+)\.(.*)')

    df["gpu"] = gpu_match[0].astype("Int64")
    df["metric"] = (
        gpu_match[1]
        .where(gpu_match[1].notna(), df["metric"])
    )

    df.loc[df["gpu"].notna(), "metric"] = "gpu." + df["metric"]

    df["time_norm"] = df["time"] - df.groupby("run_id")["time"].transform("min")

    options = {
        "mode": "a" if args.append else "w",
        "header": not os.path.exists(args.output) if args.append else True
    }
    df.to_csv(args.output, index=False, **options)

    # power_over_time(df)



def power_over_time(df):
    df = df[df["clock"] == 1785]
    df["metric"] = df["metric"].str.replace(r"gpudata\.\d+\.power", "power", regex=True)
    df = df[df["metric"] == "power"]
    
    df.to_csv("power_only.csv")

    import altair as alt

    chart = alt.Chart(df).mark_point().encode(
        x=alt.X("time_norm:Q", title="Time since start (s)"),
        y=alt.Y("value:Q", title="Metric Value"),
        color=alt.Color("power:O", title="Power"), 
    ).properties(
        width=500,
        height=500
    ).facet(
        column=alt.Column("bench:N", title="Bench")
    ).resolve_scale(
        x='independent',
        y='independent'
    )

    chart.save("power_evol.png")


if __name__ == "__main__":
    gather_multi_run_cli()
    
    # gather_matrix_cli()
    # gather_cli()

    
    # import pandas as pd
    # df = pd.read_csv("output.csv")
    # power_over_time(df)
    pass