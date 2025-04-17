import os
import yaml

import altair as alt
import pandas as pd

from milabench.sizer import to_octet


HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "..")


folder_path = os.path.join(ROOT, "config", "scaling")
benchmarks = set()

def read_config(filename, output=None):
    if output is None:
        output = []

    with open(os.path.join(folder_path, filename), "r") as fp:
        data = yaml.safe_load(fp)

        for bench, rows in data.items():
            if bench == "version":
                continue
            
            benchmarks.add(bench)
            for obs in rows.get("observations", []):
                obs["gpu"] = filename.split(".")[0]
                obs["bench"] = bench
    
                output.append(obs)

    return output

output = []
read_config("L40S.yaml", output)
read_config("H100.yaml", output)

df = pd.DataFrame(output)

df['memory'] = df['memory'].apply(lambda x: to_octet(x) / (1024 ** 2))

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
