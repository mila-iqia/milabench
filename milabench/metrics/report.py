import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy.orm import Session

from milabench.metrics.sqlalchemy import Exec, Metric, Pack


def fetch_data(client, run_name):
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
        .where(Exec.name.startswith(run_name))
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


def dropminmax(xs):
    xs = sorted(x for x in xs if x is not None)
    if len(xs) >= 5:
        xs = xs[1:-1]
    return xs


def sem(xs):
    xs = dropminmax(xs)
    return np.std(xs) / len(xs) ** 0.5


def min(xs):
    xs = dropminmax(xs)
    return np.percentile(xs, 0)


def q1(xs):
    xs = dropminmax(xs)
    return np.percentile(xs, 25)


def median(xs):
    xs = dropminmax(xs)
    return np.percentile(xs, 50)


def q3(xs):
    xs = dropminmax(xs)
    return np.percentile(xs, 75)


def max(xs):
    xs = dropminmax(xs)
    return np.percentile(xs, 100)


def mean(xs):
    xs = dropminmax(xs)
    return np.mean(xs)


def std(xs):
    xs = dropminmax(xs)
    return np.std(xs)


def count(xs):
    return len(xs)


default_metrics = {
    "min": min,
    "q1": q1,
    "median": median,
    "q3": q3,
    "max": max,
    "mean": mean,
    "std": std,
    "sem": sem,
}


def make_pivot_summary(runame, df: pd.DataFrame, metrics=None):
    benchmarks = df["bench"].unique()
    gpu = df["gpu_id"].unique()

    if metrics is None:
        metrics = default_metrics

    print(df)

    # Per-GPU
    stats = pd.pivot_table(
        df,
        index=["run", "bench", "gpu_id"],
        columns="metric",
        aggfunc=list(metrics.values()),
        dropna=True,
    )

    overall = pd.pivot_table(
        df.drop(columns=["gpu_id"], inplace=False),
        index=["run", "bench"],
        columns="metric",
        aggfunc=list(metrics.values()),
        dropna=True,
    )

    def _get(df, bench, name, gpu_id, k):
        try:
            if gpu_id is None:
                return df.loc[(runame, bench)][(k, "value", name)]
            return df.loc[(runame, bench, gpu_id)][(k, "value", name)]
        except:
            # this happens if dropna=true, STD == NA if there is only one observation
            print(f"{bench}.{name}.{k} missing")
            return 0

    def _metric(df, bench, name, gpu_id=None):
        return {k: _get(df, bench, name, gpu_id, k) for k in metrics.keys()}

    def bench(name):
        return_codes = df[df["bench"] == name][df["metric"] == "return_code"]
        total = len(return_codes)
        success = sum([int(r == 0) for r in return_codes["value"]])

        return {
            "name": name,
            "n": total,
            "successes": success,
            "failures": total - success,
            "train_rate": _metric(overall, name, "rate"),
            "walltime": _metric(overall, name, "walltime"),
            "per_gpu": {},
            "gpu_load": {
                g: {
                    "memory": _metric(stats, name, "gpu.memory", g),
                    "load": _metric(stats, name, "gpu.load", g),
                }
                for g in gpu
            },
        }

    return {name: bench(name) for name in benchmarks}
