import math
from io import StringIO
import os

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy import cast, Integer
from sqlalchemy import select, func, cast, TEXT, Float, and_

from milabench.metrics.sqlalchemy import Exec, Metric, Pack, Weight


def base_report_view(*columns, profile="default", visibility=0):
    weight_total = select(func.sum(Weight.weight * Weight.enabled.cast(Integer))).where(Weight.profile == profile).scalar_subquery()

    # Why not all Weight.pack are included?
    return (
        sqlalchemy.select(
            func.coalesce(Exec.name, "").label("run"),
            Weight.pack.label("bench"),
            func.coalesce(Metric.name, "rate").label("metric"),
            func.coalesce(Metric.value, 0).label("value"),
            func.coalesce(Metric.gpu_id, "").label("gpu_id"),
            Weight.weight.label("weight"),
            Weight.priority.label("priority"),
            cast(Weight.enabled, Integer).label("enabled"),
            weight_total.label("weight_total"),
            *columns
        )
        .select_from(Metric)
        .join(Pack, Metric.pack_id == Pack._id)
        .join(Exec, Metric.exec_id == Exec._id)
        .outerjoin(Weight, Weight.pack == Pack.name)
        .where(Weight.profile == profile)
        .where(Exec.visibility == visibility)
        .order_by(Weight.priority)
    )



# Check how to make that query
# def select_gpu(view, gpu_name):
#     view.

def fetch_data(client, run_name, profile="default"):
    stmt = (base_report_view(profile=profile)
            .where(
                Exec.name.startswith(run_name),
                Metric.name.in_(["gpu.memory", "gpu.load", "status", "walltime", "rate"]))
    )
    return fetch_data_by_query(client, stmt)


def fetch_data_by_id(client, run_id, profile="default"):
    stmt = (base_report_view(profile=profile)
            .where(
                Exec._id == run_id,
                Metric.name.in_(["gpu.memory", "gpu.load", "status", "walltime", "rate"])
            )
    )
    return fetch_data_by_query(client, stmt)


def fetch_data_by_query(client, stmt):
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
    old = len(xs)
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

def debug_count(xs):
    xs = dropminmax(xs)
    return len(xs)


def no_nan(fun):
    """NaN are not json serializable"""
    def wrapped(*args):
        return fun(*args)
    return fun


default_metrics = {
    "min": no_nan(min),
    "q1": no_nan(q1),
    "median": no_nan(median),
    "q3": no_nan(q3),
    "max": no_nan(max),
    "mean": no_nan(mean),
    "std": no_nan(std),
    "sem": no_nan(sem),
    "count": no_nan(count),
    "debug_count": no_nan(debug_count),
}


def make_pivot_summary(runame, df: pd.DataFrame, metrics=None):
    benchmarks = df["bench"].unique()
    gpu = df["gpu_id"].unique()
    gpu = [g for g in gpu if ',' not in g]

    if metrics is None:
        metrics = default_metrics

    stats = pd.pivot_table(
        df.drop(columns=["weight", "priority", "enabled", "weight_total"], inplace=False),
        index=["run", "bench", "gpu_id"],
        columns="metric",
        aggfunc=list(metrics.values()),
        dropna=True,
    )

    # Only works for multi-gpu runs
    overall = pd.pivot_table(
        df.drop(columns=["gpu_id"], inplace=False),
        index=["run", "bench"],
        columns="metric",
        aggfunc=list(metrics.values()),
        dropna=True,
    )

    def _get(self_df, bench, name, gpu_id, k):
        try:
            if gpu_id is None:
                return self_df.loc[(runame, bench)][(k, "value", name)]
            return self_df.loc[(runame, bench, gpu_id)][(k, "value", name)]
        except KeyError:
            # this happens if dropna=true, STD == NA if there is only one observation
            # print(f"{bench}.{name}.{k} missing")
            return 0

    def _metric(self_df, bench, name, gpu_id=None):
        return {k: _get(self_df, bench, name, gpu_id, k) for k in metrics.keys()}

    def bench(name):
        bench_df = df[df["bench"] == name]

        return_codes = bench_df[bench_df["metric"] == "status"]
        total = len(return_codes)

        success = sum([int(r == 0) for r in return_codes["value"]])

        # FIXME: Does not work for multi-node
        ngpu = (return_codes['gpu_id'].astype(str).apply(lambda x: len(x.split(',')))).mean()

        per_gpu = {
            g: _metric(stats, name, "rate", gpu_id=g)
                for g in gpu
        }

        # This is not correct because we should do a sum of means of per_gpu
        train_rate = _metric(overall, name, "rate")
        
        if True:
            train_rate["mean_original"] = train_rate["mean"]
            sum_of_means = sum(stat["mean"] for g, stat in per_gpu.items()) / len(per_gpu)

            if math.isnan(sum_of_means):
                train_rate["mean"] = train_rate["mean_original"]
            else:
                train_rate["mean"] = sum_of_means
                train_rate["debug_count"] = sum(stat["debug_count"] for g, stat in per_gpu.items())

        entry = {
            "name": name,
            "n": total,
            "successes": success,
            "failures": total - success,
            "train_rate": train_rate,

            "weight": bench_df["weight"].iloc[0],
            "enabled": bench_df["enabled"].iloc[0],
            "priority": bench_df["priority"].iloc[0],
            "weight_total": bench_df["weight_total"].iloc[0],

            # "walltime": _metric(overall, name, "walltime"),
            "ngpu": ngpu,
            "per_gpu": per_gpu,
            "gpu_load": {
                g: {
                    "memory": _metric(stats, name, "gpu.memory", g),
                    "load": _metric(stats, name, "gpu.load", g),
                }
                for g in gpu
            },
        }

        return entry
    r = {name: bench(name) for name in benchmarks}

    return r


def make_pandas_report(db_session, exec_id):
    from milabench.report import make_report

    df_post = fetch_data_by_id(db_session, exec_id)
   
    run_name = df_post["run"].iloc[0]
    replicated = make_pivot_summary(run_name, df_post)
    stream = StringIO()
    
    with open(os.devnull, "w") as devnull:
        df_report = make_report(replicated, stream=devnull, html=stream, weights=replicated)

    return df_report, replicated
