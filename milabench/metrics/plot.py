import altair as alt
from altair import datum

import pandas as pd
import sqlite3

con = sqlite3.connect("sqlite.db")

data = pd.read_sql_query(
    """
SELECT 
    execs.name as run,
    json_extract(execs.meta, '$.login') as login, 
    packs.name as bench,
    metrics.name as metric, 
    metrics.value as value 
FROM 
    metrics, 
    execs,
    packs
WHERE 
    metrics.exec_id = execs._id AND
    metrics.pack_id = packs._id
""",
    con,
)

print(data)


def compare_all_runs(metric="train_rate"):
    compare = (
        alt.Chart(data)
        .transform_filter((datum.metric == metric))
        .mark_bar()
        .encode(
            x="run:N",
            y=alt.Y("value", type="quantitative", aggregate="average"),
            color="run:N",
            column="bench:N",
        )
    )

    return compare


def show_run(runname, metric="train_rate"):
    base = (
        alt.Chart(data)
        .transform_filter((datum.metric == metric) and (datum.run == runname))
        .mark_bar()
        .encode(
            x="run:N",
            y=alt.Y("value", type="quantitative", aggregate="average"),
        )
    )
    err = base.mark_rule().encode(y="ci0(value)", y2="ci1(value)")

    return base + err


import numpy as np

values = pd.pivot_table(
    data,
    values="value",
    index=["run", "bench"],
    columns="metric",
    aggfunc=np.mean,
)
print(values)
print(values.describe())

print("Saving")
comp = compare_all_runs()
comp.save("compare.html")
