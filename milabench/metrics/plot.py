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

chart = (
    alt.Chart(data).mark_bar().encode(
        x='run:N',
        y='value:Q',
        color='run:N',
        column='bench:N'
    ).transform_filter(
        (datum.metric == "train_rate")
    )
)

print('Saving')
chart.save('chart.html')