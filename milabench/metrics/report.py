from milabench.metrics.sqlalchemy import Pack, Exec, Metric

import numpy as np

from bson.json_util import dumps as to_json
from bson.json_util import loads as from_json
import pandas as pd
import sqlalchemy
from sqlalchemy.orm import Session


def fetch_data(client, runame):
    stmt = (
        sqlalchemy.select(
            Exec.name.label("run"),
            Pack.name.label("bench"),
            Metric.name.label("metric"),
            Metric.value,
            Metric.gpu_id,
        )
        .join(Exec, Metric.exec_id == Exec._id)
        .join(Pack, Metric.pack_id == Pack._id)
        .where(Exec.name.startswith(runame))
    )

    results = []
    with Session(client) as sess:
        cursor = sess.execute(stmt)
        columns = list(cursor.keys())

        for row in cursor:
            row = list(row)
            row[0] = row[0].split(".")[0]
            results.append(row)

    df_post = pd.DataFrame(results, columns=columns)
    return df_post


def sem(xs):
    return np.std(xs) / len(xs) ** 0.5


def min(xs):
    return np.percentile(xs, 0)


def q1(xs):
    return np.percentile(xs, 25)


def median(xs):
    return np.percentile(xs, 50)


def q3(xs):
    return np.percentile(xs, 75)


def max(xs):
    return np.percentile(xs, 100)


def merge(*args):
    merged = dict()
    count = len(args)

    for k in args[0].keys():
        acc = 0
        for d in args:
            acc += d[k]

        merged[k] = acc / count
    return merged


default_metrics = {
    "min": min,
    "q1": q1,
    "median": median,
    "q3": q3,
    "max": max,
    "mean": np.mean,
    "std": np.std,
    "sem": sem,
}


def make_pivot_summary(runame, df, metrics=None):
    benchmarks = df["bench"].unique()
    gpu = df["gpu_id"].unique()

    if metrics is None:
        metrics = default_metrics

    stats = pd.pivot_table(
        df,
        index=["run", "bench", "gpu_id"],
        columns="metric",
        aggfunc=list(metrics.values()),
        dropna=True,
    )

    def _get(bench, name, gpu_id, k):
        try:
            return stats.loc[(runame, bench, gpu_id)][(k, "value", name)]
        except:
            # this happens if dropna=true, STD == NA if there is only one observation
            print(f"{bench}.{name}.{k} missing")
            return 0

    def _metric(bench, name, gpu_id):
        return {k: _get(bench, name, gpu_id, k) for k in metrics.keys()}

    def bench(name):
        return {
            "name": name,
            "n": 0,
            "successes": 0,
            "failures": 0,
            "train_rate": merge(*(_metric(name, "rate", g) for g in gpu)),
            "gpu_load": {
                g: {
                    "memory": _metric(name, "gpu.memory", g),
                    "load": _metric(name, "gpu.load", g),
                }
                for g in gpu
            },
        }

    return {name: bench(name) for name in benchmarks}
