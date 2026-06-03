import os
import shutil
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
    runs:            list = field(default_factory=list)
    config         : str  = os.getenv("MILABENCH_CONFIG")
    compare        : str  = None
    compare_gpus   : bool = False
    html           : str  = None
    price          : int  = None
    filter_failures: bool = False
    latest         : bool = False
    publish        : str  = os.getenv("MILABENCH_PUBLISH_KEY", None)
    dashboard_url  : str  = os.getenv("MILABENCH_DASHBOARD_URL", None)
    consolidate    : bool = False
    run_name       : str  = None
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

    # Filter out failed runs
    filter_failures: Option & bool = False

    # Only consider the latest run for each benchmark
    latest: Option & bool = False

    # Push key to publish results to the dashboard
    publish: Option & str = os.getenv("MILABENCH_PUBLISH_KEY", None)

    # Dashboard URL to publish results to
    dashboard_url: Option & str = os.getenv("MILABENCH_DASHBOARD_URL", None)

    # Consolidate run folders (remove install/prepare, merge into single folder)
    consolidate: Option & bool = False

    # Name for the consolidated run folder (e.g. commit hash)
    run_name: Option & str = None

    return Arguments(runs, config, compare, compare_gpus, html, price, filter_failures, latest, publish, dashboard_url, consolidate, run_name)


def consolidate_runs(runs_dir, run_name=None):
    """Consolidate multiple run folders into a single clean folder.

    Removes install/prepare folders, merges remaining run folders
    (latest file wins on conflicts), and optionally renames to run_name.

    Returns the path to the consolidated folder.
    """
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        return runs_dir

    subdirs = [d for d in runs_path.iterdir() if d.is_dir()]

    # Remove install and prepare folders
    for d in subdirs:
        if d.name.startswith("install") or d.name.startswith("prepare"):
            print(f"[consolidate] Removing {d.name}/")
            shutil.rmtree(d)

    # Remaining run folders sorted by mtime (oldest first)
    run_dirs = sorted(
        [d for d in runs_path.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
    )

    if not run_dirs:
        return runs_dir

    # The latest folder is the target
    target = run_dirs[-1]

    # Merge older folders into the target (skip files that already exist in target)
    for source in run_dirs[:-1]:
        for f in source.iterdir():
            if f.is_file():
                dest = target / f.name
                if not dest.exists():
                    shutil.move(str(f), str(dest))
        # Remove the now-empty source folder
        shutil.rmtree(source)
        print(f"[consolidate] Merged {source.name}/ into {target.name}/")

    # Rename to run_name if provided
    if run_name:
        final = runs_path / run_name
        if final != target:
            if final.exists():
                shutil.rmtree(final)
            target.rename(final)
            target = final
            print(f"[consolidate] Renamed to {run_name}/")

    print(f"[consolidate] Result: {target}")
    return str(target)


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

    if args.consolidate and args.runs:
        args.runs = [consolidate_runs(r, run_name=args.run_name) for r in args.runs]

    reports = None
    if args.runs:
        reports = _read_reports(*args.runs)
        summary = make_summary(
            reports,
            filter_failures=args.filter_failures,
            latest_only=args.latest,
        )

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

    if args.publish and args.runs:
        from .push_results import publish_results

        success = publish_results(
            args.runs,
            push_key=args.publish,
            dashboard_url=args.dashboard_url,
        )
        if not success:
            sys.exit(1)



def make_report_for_single_run(run_folder, output=sys.stdout, filter_failures=False, latest_only=False):
    reports = _read_reports(run_folder)
    summary = make_summary(
        reports,
        filter_failures=filter_failures,
        latest_only=latest_only,
    )

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

    parser = ArgumentParser()
    parser.add_argument("--folder", type=str, help="run folder")
    parser.add_argument("--filter-failures", action="store_true", help="filter out failed benchmarks")
    parser.add_argument("--latest", action="store_true", help="only keep the latest run per benchmark")
    args = parser.parse_args()

    found_tags = defaultdict(int)
    tags = make_tags(default_tags())
    reports = []

    run_folders = sorted(
        set(gather_run_folders(args.folder)),
        key=lambda f: os.path.getmtime(f),
    )

    for folder in run_folders:
        full_pth = Path(folder)

        if full_pth.is_file():
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
                output=devnull,
                filter_failures=args.filter_failures,
            )

        print(full_pth, columns)

        # Insert columns to the data frame
        for key, value in columns.items():
            df[key] = value

        df = df.rename_axis("bench").reset_index()
        df["bench"] = df["bench"].astype("string")
        reports.append(df)

    # Print the full df
    all_reports = pd.concat(reports, ignore_index=True)

    if args.filter_failures:
        all_reports = all_reports[all_reports["fail"] <= 0]

    if args.latest:
        all_reports = all_reports.drop_duplicates(subset=["bench"], keep="last")

    all_reports.to_csv("big_beautiful_report.csv", index=False)

    print(pandas_to_string(all_reports))



if __name__ == "__main__":
    report_combine()
