import os
import yaml

import altair as alt
import pandas as pd

from milabench.sizer import to_octet


HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")


folder_path = os.path.join(ROOT, "config", "scaling")


def read_config(filename, output=None, benchmarks=None):
    if output is None:
        output = []

    with open(os.path.join(folder_path, filename), "r") as fp:
        data = yaml.safe_load(fp)

        for bench, rows in data.items():
            if bench == "version":
                continue
            
            if benchmarks is not None:
                benchmarks.add(bench)

            for obs in rows.get("observations", []):
                obs["gpu"] = filename.split(".")[0]
                obs["bench"] = bench
                obs["memory"] = to_octet(obs["memory"]) / (1024 ** 2)
                
                output.append(obs)

    return output


def main():
    benchmarks = set()

    output = []
    read_config("L40S.yaml", output, benchmarks)
    read_config("H100.yaml", output, benchmarks)
    read_config("MI325.yaml", output, benchmarks)

    df = pd.DataFrame(output)

    def perf_scaling():
        for bench in benchmarks:

            title = alt.TitleParams(bench, anchor='middle')

            chart = (
                alt.Chart(df[df["bench"] == bench], title=title)
                    .mark_point().encode(
                        x="memory",
                        y="perf",
                        shape="gpu"
                    )
            )

            chart.save(os.path.join(HERE, "plots", f"{bench}.png"))


    for bench in benchmarks:
        title = alt.TitleParams(bench, anchor='middle')

        chart = (
            alt.Chart(df[df["bench"] == bench], title=title)
                .mark_point().encode(
                    x="batch_size",
                    y="memory",
                    shape="gpu",
                    color="gpu",
                    size="perf",
                )
        )

        chart.save(os.path.join(HERE, "plots", f"{bench}.png"))
