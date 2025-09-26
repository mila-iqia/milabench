
import os
import json
from collections import defaultdict

import altair as alt
import pandas as pd

from ..constant import JOBRUNNER_LOCAL_CACHE

# Force early copy straight from the compute node
# scp cn-d004.server.mila.quebec:/tmp/7793606/cuda/results/runs/stats.jsonl /home/delaunap/work/milabench_dev/data/A100_run_f8fead63/runs/
#
#                           

def beegfs_iops(job_name="A100_run_f8fead63") -> alt.Chart:
    """Display distributed IOPS used during the run"""
    lines = []

    with open(os.path.join(JOBRUNNER_LOCAL_CACHE, job_name, "runs", "stats.jsonl"), "r") as fp:
        for line in fp.readlines():
            lines.append(json.loads(line))

    df = pd.DataFrame(filter(lambda row: row["elapsed"] != 0, lines))

    storage = (
        alt.Chart(df)
            .mark_line().encode(
                x="elapsed",
                y="sum",
                color="source"
            ).resolve_scale(y="independent")
    )

    storage.save(f"iops.png")
    return storage



def job_inspection():
    """This is used to inspect a finished job"""

    def list_runs(job_name):
        runs_folder = os.path.join(JOBRUNNER_LOCAL_CACHE, job_name, "runs")
        runs = []
        for dir in os.listdir(runs_folder):
            if os.path.isdir(dir):
                runs.append(dir)
        return runs
    
    def list_bench(job_name, run):
        run_folder = os.path.join(JOBRUNNER_LOCAL_CACHE, job_name, "runs", run)
        bench = defaultdict(list)
        for filename in os.listdir(run_folder):
            bench[filename.split(".")[0]].append(filename)
        return bench

    def replay_run(job_name, run):
        bench_instances = list_bench(job_name, run)

        # Replay to websocket
        for bench, instances in bench_instances.items():
            replay_bench(job_name, run, bench, -1)

    def replay_bench(job_name, run, bench, idx=-1):
        bench_instances = list_bench(job_name, run).get(bench, [])

        if idx >= 0:
            bench_instances = [bench_instances[idx]]

        # Replay to websocket

    

if __name__ == "__main__":
    beegfs_iops()
