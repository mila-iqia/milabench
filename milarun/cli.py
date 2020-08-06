from coleo import auto_cli, config, Argument, ConfigFile, default, tooled
from datetime import datetime
from types import SimpleNamespace as NS
from .lib.experiment import Experiment
from .lib.helpers import resolve
from .lib.report import make_report, summarize
import os
import blessed
import json
import sys
import shutil
import pkg_resources
import time
import traceback
import subprocess


def _split_args(argv):
    try:
        idx = argv.index("--")
        myargs, others = argv[:idx], argv[idx + 1:]
    except ValueError:
        myargs, others = argv, []
    return myargs, others


def _get_entries():
    for entry_point in pkg_resources.iter_entry_points("milarun.run"):
        _entries[entry_point.name]["run"] = entry_point
    for entry_point in pkg_resources.iter_entry_points("milarun.download"):
        _entries[entry_point.name]["download"] = entry_point
    for entry_point in pkg_resources.iter_entry_points("milarun.dataset"):
        _entries[entry_point.name]["dataset"] = entry_point


def command_dataset(subargv):
    """Download a dataset.

    Positional argument must point to a Python function in
    a module, using the entry point syntax.
    """
    # Name(s) of the dataset(s) to download (e.g. milarun.datasets:mnist)
    # [positional: +]
    name: Argument & resolve

    # Root directory for datasets (default: $MILARUN_DATAROOT)
    # [metavar: PATH]
    # [alias: -d]
    dataroot: Argument = default(os.getenv("MILARUN_DATAROOT"))

    if not dataroot:
        print(
            "milarun: error: no dataroot specified, ",
            "please use --dataroot/-d or set $MILARUN_DATAROOT",
            file=sys.stderr
        )
        sys.exit(1)

    dataroot = os.path.realpath(os.path.expanduser(dataroot))

    for dataset_gen in name:
        dataset = dataset_gen(dataroot)
        dataset.download()


def _compute_outdir(out):
    outroot = os.getenv("MILARUN_OUTROOT")
    if out is None and outroot is not None:
        now = datetime.now()
        dirname = now.strftime("results-%Y-%m-%d.%H:%M:%S")
        out = os.path.join(outroot, dirname)

    return out and os.path.realpath(os.path.expanduser(out))


def command_run(subargv):
    """Run a benchmark.

    Positional argument must point to a Python function in
    a module, using the entry point syntax.
    """
    # [positional]
    # Name of the experiment to run (e.g. milarun.models.polynome:main)
    function: Argument

    # File/directory where to put the results. Assumed to be a directory
    # unless the name ends in .json
    # [metavar: PATH]
    # [alias: -o]
    out: Argument = default(None)
    out = _compute_outdir(out)

    # Name of the experiment (optional)
    experiment_name: Argument = default(None)

    # ID of the job (optional)
    job_id: Argument = default(None)

    # Extra information (optional)
    extra: Argument = default("{}")
    extra = json.loads(extra)

    # Root directory for datasets (default: $MILARUN_DATAROOT)
    # [metavar: PATH]
    # [alias: -d]
    dataroot: Argument = default(os.getenv("MILARUN_DATAROOT"))

    if not dataroot:
        print(
            "milarun: error: no dataroot specified, ",
            "please use --dataroot/-d or set $MILARUN_DATAROOT",
            file=sys.stderr
        )
        sys.exit(1)

    dataroot = os.path.realpath(os.path.expanduser(dataroot))

    run = resolve(function)

    experiment = Experiment(
        name=experiment_name or function,
        job_id=job_id,
        dataroot=dataroot,
        outdir=out,
    )
    experiment.set_fields(extra)
    experiment["call"] = {
        "function": function,
        "argv": subargv,
    }

    with experiment.time("program"):
        experiment.execute(lambda: run(experiment, subargv))

    experiment.write(out)


def command_rerun(subargv):
    """Re-run a benchmark, using the JSON output of a previous run."""
    # JSON results file
    # [positional]
    job: Argument & config

    # File/directory where to put the results. Assumed to be a directory
    # unless the name ends in .json
    # [alias: -o]
    out: Argument = default(None)

    argmap = {
        "dataroot": "--dataroot",
        "name": "--experiment-name",
    }

    jc = job["call"]
    cmd = [
        "milarun",
        "run",
    ]
    if out:
        cmd += ["--out", out]
    for k, v in argmap.items():
        if job[k] is not None:
            cmd += [v, job[k]]
    cmd += [
        jc["function"],
        "--",
        *jc["argv"],
        *subargv
    ]

    print("=== re-run command ===")
    for k, v in job["environ"].items():
        print(f"{k} = {v}")
    print(" ".join(cmd))
    print("======================")
    subprocess.run(cmd, env={**os.environ, **job["environ"]})


def _launch_job(jobdata, definition, cgexec, subargv):
    psch = definition.get("partition_scheme", None)
    psch_type = psch and psch["type"]

    process_data = []
    run_id = jobdata["run"]

    for cmd in definition.get("prepare", []):
        print("+", cmd)
        subprocess.run(cmd, shell=True)

    if psch_type == "per-gpu":
        import torch
        device_count = max(torch.cuda.device_count(), 1)
        for device_id in range(device_count):
            partial_args = {
                "--job-id": f"{run_id}.{device_id}",
                "--extra": json.dumps({
                    "job": definition,
                    "device": device_id,
                    "devices": [device_id],
                }),
            }
            env = {
                "CUDA_VISIBLE_DEVICES": str(device_id),
            }
            process_data.append((partial_args, env, "Popen"))

    elif psch_type == "gpu-progression":
        import torch
        device_count = max(torch.cuda.device_count(), 1)
        for device_id in range(device_count):
            device_list = list(range(device_id + 1))
            partial_args = {
                "--job-id": f"{run_id}.{device_id}",
                "--extra": json.dumps({
                    "job": definition,
                    "devices": device_list,
                }),
            }
            env = {
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, device_list)),
            }
            process_data.append((partial_args, env, "run"))

    elif psch_type is None or psch_type == "normal":
        process_data.append(({}, {}, "Popen"))

    else:
        raise Exception(f"Unknown partition_scheme: {psch_type}")

    processes = []

    for partial_args, env, exec_type in process_data:
        args = {
            "--experiment-name": f"{jobdata['suite']}.{jobdata['name']}",
            "--job-id": run_id,
            "--out": jobdata["out"],
            "--dataroot": jobdata["dataroot"],
            **partial_args,
            "--": True,
            **definition["arguments"],
        }
        args = {k: v for k, v in args.items() if v is not None}

        cmd = []
        cgroup = cgexec and psch and psch.get("cgroup", None)
        if cgroup:
            cmd += ["cgexec", "-g", cgroup.format(**env)]
        cmd += definition.get("exec_prefix", [])

        cmd += ["milarun", "run", definition['experiment']]
        for k, v in args.items():
            if isinstance(v, bool):
                cmd.append(k)
            else:
                cmd.extend((k, str(v)))
        cmd.extend(subargv)

        print("Running:", " ".join(cmd))
        if exec_type == "Popen":
            processes.append(
                subprocess.Popen(
                    cmd,
                    env={**os.environ, **env},
                )
            )
        else:
            subprocess.run(
                cmd,
                env={**os.environ, **env},
            )

    for process in processes:
        try:
            return_code = process.wait()
        except Exception as e:
            process.kill()


def command_jobs(subargv):
    """Run benchmarks defined in JSON jobs file."""
    # [positional]
    # File containing the job definitions
    jobs: Argument & ConfigFile

    # Number of times to repeat the suite of jobs
    repeat: Argument & int = default(1)

    # Use cgroups to execute each partition
    cgexec: Argument & bool = default(True)

    # Only download the dataset for the experiment
    download: Argument & bool = default(False)

    # Directory where to put the results
    # [alias: -o]
    out: Argument & os.path.abspath = default(None)
    out = _compute_outdir(out)

    # Name(s) of the job to run
    # [nargs: +]
    name: Argument = default([])

    # Root directory for datasets (default: $MILARUN_DATAROOT)
    # [metavar: PATH]
    # [alias: -d]
    dataroot: Argument = default(os.getenv("MILARUN_DATAROOT"))

    if not dataroot:
        print(
            "milarun: error: no dataroot specified, ",
            "please use --dataroot/-d or set $MILARUN_DATAROOT",
            file=sys.stderr
        )
        sys.exit(1)

    dataroot = os.path.realpath(os.path.expanduser(dataroot))

    jobs_data = jobs.read()

    # Merge the common fields
    common = jobs_data.pop("*", None)
    if common:
        jobs_data = {
            name: {**common, **job}
            for name, job in jobs_data.items()
        }

    name = set(name)
    not_found = name and name - set(jobs_data.keys())
    if not_found:
        print(f"Could not find job(s): {not_found}", file=sys.stderr)
        sys.exit(1)

    start = time.time()
    for i in range(repeat):
        for jobname, definition in jobs_data.items():
            if not name or jobname in name:
                jobdata = {
                    "name": jobname,
                    "suite": os.path.splitext(os.path.basename(jobs.filename))[0],
                    "run": i,
                    "out": out,
                    "dataroot": dataroot,
                }
                _launch_job(jobdata, definition, cgexec, subargv)
    end = time.time()
    print(f"Total time: {end - start:.2f}s")


def _filter(x, path):
    return x.get("job_id", None) is not None and not x.get("sub_job", False)


def _group(x):
    scheme = x["job"]["partition_scheme"]
    if scheme["type"] == "per-gpu":
        return x["name"]
    elif scheme["type"] == "gpu-progression":
        return f"{x['name']}.{x['device_count']}"
    else:
        return x["name"]


def command_summary(subargv):
    """Output a JSON summary of the results of the jobs command."""

    # Directory in which the reports are
    # [positional]
    reports: Argument

    # JSON file where to output the summary
    # [alias: -o]
    out: Argument = default(None)

    reports = os.path.realpath(os.path.expanduser(reports))
    out = out and os.path.realpath(os.path.expanduser(out))

    results = summarize(reports, filter=_filter, group=_group)

    if out:
        json.dump(results, open(out, "w"), indent=4)
    else:
        print(json.dumps(results, indent=4))


def command_report(subargv):
    """Output a report from the results of the jobs command."""

    # Directory in which the reports are
    # [positional]
    reports: Argument

    # Comparison file
    compare: Argument & config = default(None)

    # Weights file to compute the score
    weights: Argument & config = default(None)

    # Path to the HTML file to generate
    html: Argument = default(None)

    # Compare the configuration's individual GPUs
    compare_gpus: Argument & bool = default(False)

    # Whether to penalize variance in the score (defaults to False)
    penalize_variance: Argument & bool = default(False)

    # Price of the configuration, to compute score/price ratio
    price: Argument & float = default(None)

    # Title to give to the report
    title: Argument = default(None)

    reports = os.path.realpath(os.path.expanduser(reports))

    if os.path.isdir(reports):
        results = summarize(reports, filter=_filter, group=_group)
    else:
        results = json.load(open(reports))

    make_report(
        results,
        compare=compare,
        weights=weights,
        html=html,
        compare_gpus=compare_gpus,
        price=price,
        title=title,
        penalize_variance=penalize_variance,
    )


def main():
    argv, subargv = _split_args(sys.argv)
    _get_entries()
    commands = {}
    for name, value in globals().items():
        parts = name.split("_")
        if parts[0] == "command":
            assert len(parts) > 1
            curr = commands
            for part in parts[1:-1]:
                curr = curr.setdefault(part, {})
            curr[parts[-1]] = value
    auto_cli(commands, [subargv], argv=argv[1:])
