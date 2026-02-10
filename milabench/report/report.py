from .read import extract_milabench_metrics, augment_energy_estimator
from ..metrics.report import default_metrics, mean as dropminmax_mean

import pandas as pd


SELECTED_METRICS = [
    "rate",
    "gpudata.memory.0",
    "gpudata.power",
    "gpudata.load",
    "gpudata.temperature",
    "energy"
]


def report(folder, selected_metrics=None):
    if selected_metrics is None:
        selected_metrics = SELECTED_METRICS

    df = pd.DataFrame(augment_energy_estimator(extract_milabench_metrics(folder)))

    # Drop unnecessary index if possible
    index = ["p0", "bench", "device"]
    if df["p0"].nunique() == 1:
        df = df.drop(columns=["p0"])
        index = ["bench", "device"]

    # Drop unselected metrics
    df = df[df["metric"].isin(selected_metrics)]

    # Pivot
    stats = pd.pivot_table(
        df.drop(columns=["time", "unit", "task"], inplace=False),
        index=index,
        columns="metric",
        aggfunc=list(default_metrics.values()),
        dropna=True,
    )

    select = [
        ("mean", "value", "rate"),
        ("max" , "value", "gpudata.memory.0"),
        ("mean", "value", "gpudata.power"),
        ("sum" , "value", "energy"),
    ]

    # print(stats)
    # print(stats.columns)

    # Per GPU dat
    stats = stats.loc[:, select]
    return stats


def custom(folder):
    from .read import accumulate_per_bench, accumulate_per_device, aggregate
    from ..metrics.report import mean as dropminmax_mean
    import numpy as np

    metric_stream = extract_milabench_metrics(folder)

    augmented_stream = augment_energy_estimator(metric_stream)

    agg = aggregate(augmented_stream)

    per_device_acc = {
        "rate": dropminmax_mean,
        "gpu.memory.0": max,
        "energy": sum,
        "ngpu": np.mean,
        "elapsed": np.mean,
        "success": np.sum,
    }
    per_device = accumulate_per_device(agg, per_device_acc)

    device_combine = {
        "rate": {"score": sum, "rate": np.mean},
        "gpu.memory.0": max,
        "energy": np.mean,
        "elapsed": np.mean,
        "success": {"n": len, "success": np.sum},
        "ngpu": np.mean,
    }

    data = accumulate_per_bench(per_device, device_combine)

    df = pd.DataFrame(data)

    stats = pd.pivot_table(
        df,
        index=("p0", "bench"),
        columns="metric",
        aggfunc="sum",
        dropna=True,
    )


    return stats


if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.2f}".format)

    # p = "/home/delaunap/work/milabench_dev/data/A100_mn_run_2b90373c/runs/fafuvegu.2025-10-16_01:37:14.739584/bert-tf32-fp16.*"
    # df = report(p)

    # print(df)
    # print(df.columns)

    p = "/home/delaunap/work/milabench_dev/projects/hypertec/nvl/p600*"
    df = custom(p)

    print(df)