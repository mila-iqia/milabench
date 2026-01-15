import os
import re
from pathlib import Path

from ..syslog import syslog
from ..capability import is_system_capable


def _expected_logfiles(packs, repeat):
    from ..multi import make_execution_plan

    files = []

    for index in range(repeat):
        for pack in packs.values():
            # System is not capable of running benchmarks
            if not is_system_capable(pack):
                continue

            plan = make_execution_plan(pack, index, repeat)
            plan_files = []

            for exec in plan.executors:
                epack = exec.pack
                plan_files.append(epack.logfile("data").name)
            
            files.append((plan, plan_files))

    return files


def is_run_complete(runfolder, logfiles):
    for logfile in logfiles:
        full_path = os.path.join(runfolder, logfile)

        # Check if the log exists
        if not os.path.exists(full_path):
            return False

        # Check if the logs are valid


    return True


def resume_from_files(packs, runfolder, repeat):
    if repeat > 1:
        print("repeat > 1 is not supported")

    expected = _expected_logfiles(packs, repeat)

    missing = []
    for plan, logfiles in expected:
        if not is_run_complete(runfolder, logfiles):
            missing.append(plan)
        
    return missing


def full_run(packs, repeat):
    """"Build run plans for the entire benchmarking suite"""
    expected = _expected_logfiles(packs, repeat)

    missing = []
    for plan, _ in expected:
        missing.append(plan)
        
    return missing


def find_matching_runfolder(base, run_name):
    """Find run folder that match the run name pattern"""

    pattern = re.sub(r"\{[^}]+\}", "*", run_name)

    base = Path(base)

    matches = list(base.glob(pattern))

    if len(matches) > 1:
        raise RuntimeError("Cannot reusme, found multiple matching runs")

    return matches[0].name


def resume_as_bench_selector(packs, base, run_name):
    from ..multi import make_execution_plan
    run_folder = os.path.join(base, run_name)

    filtered_packs = {}
    for name, pack in packs.items():
        # System is not capable of running benchmarks
        if not is_system_capable(pack):
            continue

        plan = make_execution_plan(pack, 0, 1)
        logfiles = []
        for exec in plan.executors:
            epack = exec.pack
            logfiles.append(epack.logfile("data").name)
        
        if not is_run_complete(run_folder, logfiles):
            filtered_packs[name] = pack

    return filtered_packs


def schedule_run(packs, base, run_name, repeat, resume):
    if "{"  in run_name and "}" in run_name:
        run_pat = run_name
        run_name = find_matching_runfolder(base, run_name)
        syslog("Found run to resume {run_pat} => {run_name}", run_pat=run_pat, run_name=run_name)

    runfolder = os.path.join(base, run_name)

    if resume and not os.path.exists(runfolder):
        syslog("Requested resume on an non existing folder {runfolder}", runfolder=runfolder)

    if resume:
        return resume_from_files(packs, runfolder, repeat)
    else:
        return full_run(packs, repeat)


def main():
    from milabench.common import assemble_config, get_pack
    import tqdm

    config = "/home/delaunap/work/milabench_dev/milabench/config/vllm.yaml"

    p = "/home/delaunap/work/milabench_dev/projects/hypertec/nvl/p600.o350.2026-01-07_14-27-23"

    run_name =  os.path.basename(p)
    # run_name = "p600.o350.{time}" 
    base_path = os.path.dirname(p)

    config = assemble_config(run_name, config, base_path)

    packs = {name: get_pack(defn) for name, defn in config.items()}

    for i in tqdm.tqdm(schedule_run(packs, base_path, run_name, repeat=1, resume=True)):
        # print(i)
        pass

if __name__ == "__main__":
    main()