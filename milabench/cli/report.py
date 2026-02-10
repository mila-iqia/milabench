import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from coleo import Option, config as configuration, tooled

from ..common import _error_report, _get_multipack, _read_reports
from ..report import make_report
from ..summary import make_summary


# fmt: off
@dataclass
class Arguments:
    runs:        list = field(default_factory=list)
    config      : str = os.getenv("MILABENCH_CONFIG")
    compare     : str = None
    compare_gpus: bool = False
    html        : str = None
    price       : int = None
# fmt: on


@tooled
def arguments():
    # Runs directory
    # [action: append]
    runs: Option = []

    config: Option & str = os.getenv("MILABENCH_CONFIG")

    # Comparison summary
    compare: Option & configuration = None

    # Compare the GPUs
    compare_gpus: Option & bool = False

    # HTML report file
    html: Option = None

    # Price per unit
    price: Option & int = None

    return Arguments(runs, config, compare, compare_gpus, html, price)


@tooled
def cli_report(args=None):
    """Generate a report aggregating all runs together into a final report."""
    if args is None:
        args = arguments()

    # Examples
    # --------
    # >>> milabench report --runs results/
    # Source: /home/newton/work/milabench/milabench/../tests/results
    # =================
    # Benchmark results
    # =================
    #                    n fail       perf   perf_adj   std%   sem%% peak_memory
    # bert               2    0     201.06     201.06  21.3%   8.7%          -1
    # convnext_large     2    0     198.62     198.62  19.7%   2.5%       29878
    # td3                2    0   23294.73   23294.73  13.6%   2.1%        2928
    # vit_l_32           2    1     548.09     274.04   7.8%   0.8%        9771
    # <BLANKLINE>
    # Errors
    # ------
    # 1 errors, details in HTML report.

    reports = None
    if args.runs:
        reports = _read_reports(*args.runs)
        summary = make_summary(reports)

    if args.config:
        from milabench.common import arguments as multipack_args

        margs = multipack_args()
        args.config = _get_multipack(margs, return_config=True)

    make_report(
        summary,
        compare=args.compare,
        weights=args.config,
        html=args.html,
        compare_gpus=args.compare_gpus,
        price=args.price,
        title=None,
        sources=args.runs,
        errdata=reports and _error_report(reports),
        stream=sys.stdout,
    )



def make_report_for_single_run(run_folder, output=sys.stdout):
    reports = _read_reports(run_folder)
    summary = make_summary(reports)

    # FIXME
    from milabench.common import arguments as multipack_args
    margs = multipack_args()
    config = _get_multipack(margs, return_config=True)

    df = make_report(
        summary,
        weights=config,
        title=None,
        errdata=reports and _error_report(reports),
        stream=output,
    )

    return df


def gather_run_folders(folder, runs=None):
    if runs is None:
        runs = []
    
    for folder_name in os.listdir(folder):
        full_pth = os.path.join(folder, folder_name)

        if os.path.isfile(full_pth):
            if full_pth.endswith(".data"):
                runs.append(Path(full_pth).parent)
        else:
            runs.extend(gather_run_folders(full_pth))

    return runs
        

def report_combine():
    from argparse import ArgumentParser
    from collections import defaultdict

    import pandas as pd

    from ..report import pandas_to_string
    from .gather import default_tags, extract_tags, make_tags
    from collections import defaultdict

    parser = ArgumentParser()
    parser.add_argument("--folder", type=str, help="run folder")
    args = parser.parse_args()

    found_tags = defaultdict(int)
    tags = make_tags(default_tags())
    reports = []

    run_folders = list(set(gather_run_folders(args.folder)))

    for folder in run_folders:
        full_pth = Path(folder)

        if (full_pth.is_file()):
            continue
        
        columns = {
            "power": "600",
            "clock": "1785",
        }

        # Tag Extraction from the run name
        for tag, value in extract_tags(full_pth.name, tags, found_tags):
            if value != "NA":
                columns[tag] = value
        
        # ---

        # Report Generation
        with open(os.devnull, "w") as devnull:
            df = make_report_for_single_run(
                str(full_pth),
                output=devnull
            )

        print(full_pth, columns)

        # Insert columns to the data frame
        for key, value in columns.items():
            df[key] = value

        df = df.rename_axis("bench").reset_index()
        df["bench"] = df["bench"].astype("string")
        #
        reports.append(df)

    # Print the full df
    all_reports = pd.concat(reports, ignore_index=True)

    all_reports.to_csv("big_beautiful_report.csv", index=False)

    print(pandas_to_string(all_reports))



if __name__ == "__main__":
    report_combine()
