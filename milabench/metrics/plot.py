from functools import partial

import altair as alt
from altair import datum

import numpy as np
import pandas as pd
import sqlite3


raw_data = """
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
"""


class Report:
    def __init__(self, sqlite) -> None:
        con = sqlite3.connect(sqlite)
        self.data = pd.read_sql_query(
            raw_data,
            con,
        )
        
    def normalized_metric(self, metric, baseline):
        normalized = self.data.copy()
        
        extracted = self.data[self.data['metric'] == metric]
        baseline = extracted[extracted['run'] == baseline].groupby(['bench']).median()
        
        for bench in baseline.index:
            selection = (normalized['metric'] == metric) & (normalized['bench'] == bench)
            
            normalized.loc[selection, 'value'] = \
                normalized.loc[selection, 'value'] / baseline.loc[bench]['value']
            
        return normalized
    
    def plot_runs(self, metric, baseline):
        data = self.normalized_metric(metric, baseline)
        
        mn = data[data['metric'] == metric].min().value
        mx = data[data['metric'] == metric].max().value
        
        bars = (alt.Chart()
            .transform_filter((datum.metric == metric))
            .mark_boxplot()
            .encode(
                x='run:N',
                y=alt.Y('value:Q', title="Speed Up", scale=alt.Scale(domain=[mn, mx])),
                color='run:N',
            )
        )
        return alt.layer(bars, data=data).facet(
            column='bench:N'
        )
        
    def compute_stats(self, metric='train_rate'):
        stats = dict(
            median=np.median, 
            q25=partial(np.quantile, q=0.25), 
            q75=partial(np.quantile, q=0.75),
            sd=np.std,
        )
        
        data = self.data[self.data['metric'] == metric]
        result = None
        
        for name, stat in stats.items():
            values = pd.pivot_table(
                data,
                values="value",
                index=["bench", "run"],
                columns="metric",
                aggfunc=stat,
            ).rename(columns={metric: name})
            
            if result is None:
                result = values
            else:
                result = result.join(values)
        
        print(pd.concat({metric: result}))
    

rep = Report("sqlite.db")
plot = rep.plot_runs('train_rate', 'zijudibi')
plot.save('com.html')
rep.compute_stats()
