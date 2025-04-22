from io import StringIO
import os
from contextlib import contextmanager
from collections import defaultdict

import pandas as pd
from flask import Flask, jsonify
import sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy import select, func, cast, TEXT

from milabench.metrics.sqlalchemy import Exec, Metric, Pack
from milabench.metrics.sqlalchemy import SQLAlchemy
from milabench.metrics.report import fetch_data, make_pivot_summary
from milabench.report import make_report
from .utils import database_uri, page, make_selection_key, make_filter, cursor_to_json, cursor_to_dataframe


def view_server(config):
    """Display milabench results"""

    DATABASE_URI = database_uri()

    app = Flask(__name__)
    app.config.update(config)

    @contextmanager
    def sqlexec():
        with SQLAlchemy(DATABASE_URI) as logger:
            with Session(logger.client) as sess:
                yield sess

    @app.route('/api/summary/<runame>')
    def api_summary(runame):
        with SQLAlchemy(DATABASE_URI) as logger:
            df_post = fetch_data(logger.client, runame)

        multirun = {}
        for real_runname in df_post["run"].unique():
            multirun[real_runname] = make_pivot_summary(real_runname, df_post)

        return jsonify(multirun)

    @app.route('/report/<runame>')
    def report(runame):
        with SQLAlchemy(DATABASE_URI) as logger:
            df_post = fetch_data(logger.client, runame)

        names = list(df_post["run"].unique())

        if len(names) > 0:
            print("multiple run report") 
        
        full_name = names[0]

        replicated = make_pivot_summary(full_name, df_post)

        stream = StringIO()
        
        with open(os.devnull, "w") as devnull :
            make_report(replicated, stream=devnull, html=stream)

        return stream.getvalue()
    
    @app.route('/api/ls/exec')
    def api_exec_list():
        stmt = sqlalchemy.select(
            Exec._id,
            Exec.name,
        )

        results = []
        with sqlexec() as sess:
            cursor = sess.execute(stmt)
            columns = list(cursor.keys())
            for row in cursor:
                results.append({k: v for k, v in zip(columns, row)})

        return results
    
    @app.route('/api/exec/<exec_id>/packs')
    def api_packs_show(exec_id):
        stmt = sqlalchemy.select(Pack).where(Pack.exec_id == exec_id)

        results = []
        with sqlexec() as sess:
            cursor = sess.execute(stmt)
            for row in cursor:
                for col in row:
                    results.append(col.as_dict())
    
        return results

    @app.route('/api/exec/<exec_id>/packs/<pack_id>/metrics')
    def api_metrics_show(exec_id, pack_id):
        stmt = sqlalchemy.select(Metric).where(Metric.exec_id == exec_id, Metric.pack_id == pack_id)

        results = []
        with sqlexec() as sess:
            cursor = sess.execute(stmt)
            for row in cursor:
                for col in row:
                    results.append(col.as_dict())
    
        return results


    @app.route('/api/keys')
    def api_ls_keys():
        pass

    @app.route('/pivot')
    def api_pivot():
        from milabench.metrics.report import base_report_view
        selected_keys = [ 
            make_selection_key(key) for key in [
                "Exec:meta.accelerators.gpus.0.product as gpu",
                "Exec:meta.accelerators.gpus.0.memory.total as vram"
            ]
        ]

        table = base_report_view(*selected_keys).where(Metric.exec_id == 12)

        with sqlexec() as sess:
            cursor = sess.execute(table)

            results = cursor_to_json(cursor)

            # df = cursor_to_dataframe(cursor)
        
        pivot_spec = {
            "rows": [
                "run", "gpu", "bench",
            ],
            "cols": [
                "metric"
            ],
            "values": {
                "value": ["mean", "max"],
            },
            # "filters": [
            #     "gpu == 'L40S'"
            # ]
        }

        df = pd.DataFrame(results)

        filtered = df
        for filter in pivot_spec.get("filters", []):
            filtered = filtered.query(filter)

        overall = pd.pivot_table(
            filtered,
            values=pivot_spec["values"].keys(),
            index=pivot_spec["rows"],
            columns=pivot_spec["cols"],
            aggfunc=pivot_spec["values"],
            dropna=True,
        )

        formatters = {
            ("value", 'mean', "gpu.load"): "{:.2%}".format,
            ("value", 'mean', "gpu.memory"): "{:.2%}".format,
            ("value", 'mean', "gpu.power"): "{:.2f}".format, 
            ("value", 'mean', "gpu.temperature"): "{:.2f}".format, 
            ("value", 'mean', "loss"): "{:.2f}".format,
            ("value", 'mean', "walltime"): "{:.2f}".format,
            ("value", 'mean', "rate"): "{:.2f}".format,
            ("value", 'mean', "return_code"): "{:.0f}".format,
            ("value", 'mean', "memory_peak"): "{:.0f}".format,
        }

        column_order = [
            ('value', 'mean',            'rate'),
            ('value', 'mean',        'gpu.load'),
            ('value', 'mean',      'gpu.memory'),
            ('value', 'mean',       'gpu.power'),
            ('value', 'mean', 'gpu.temperature'),
            ('value', 'mean',        'walltime'),
            # Not as important
            ('value', 'mean',            'loss'),
            ('value', 'mean',     'memory_peak'),
            ('value', 'mean',     'return_code'),
        ]
    
        print(overall.columns)

        df = overall
        df = df[column_order]
        df = df.reset_index()

        priority_map = defaultdict(int)
        for i, k in enumerate(sorted(list(set(df['bench'])))):
            priority_map[k] = i

        priority_map.update({
            "fp16": -1,
            "bf16": -2,
            "tf32": -3,
            "fp32": -4,
        })

        df['_priority'] = df['bench'].map(priority_map)
        df = df.sort_values('_priority').drop(columns=['_priority'])

        df = df.set_index(["run", "gpu", "bench"])
        
        return page("", df.to_html(formatters=formatters, classes=["table", "table-striped", "table-hover", "table-sm"], na_rep=""))
        
    @app.route('/api/ls/gpu')
    def api_ls_gpu():
        """Fetch a list of gpus milabench ran on"""

        # Note that assumes all gpus are the same model which should be fine
        stmt = select(func.distinct(cast(Exec.meta["accelerators"]["gpus"]["0"]["product"], TEXT)))

        with sqlexec() as sess:
            return sess.execute(stmt).scalars().all()
        
    @app.route('/api/ls/metrics')
    def api_ls_metrics():
        """Fetch a list of all saved up metrics"""

        # Note that assumes all gpus are the same model which should be fine
        stmt = select(func.distinct(Metric.name))

        with sqlexec() as sess:
            return sess.execute(stmt).scalars().all()
        
    @app.route('/api/ls/pytorch')
    def api_ls_pytorch():
        """Fetch a list of pytorch version milabench ran on"""
        stmt = select(func.distinct(cast(Exec.meta["pytorch"]["torch"], TEXT)))

        with sqlexec() as sess:
            return sess.execute(stmt).scalars().all()
        
    @app.route('/api/ls/milabench')
    def api_ls_milabench():
        """Fetch a list of milabench version"""
        stmt = select(func.distinct(cast(Exec.meta["milabench"]["tag"], TEXT)))

        with sqlexec() as sess:
            return sess.execute(stmt).scalars().all()

    @app.route('/api/exec/<id>')
    def api_exec_show(id):
        stmt = sqlalchemy.select(Exec).where(Exec._id == id)

        results = []
        with sqlexec() as sess:
            cursor = sess.execute(stmt)
            for row in cursor:
                result = row[0]
                return result.as_dict()
    
        return results
    
    @app.route('/index')
    def index():
        parts = []

        parts = "".join(parts)

        body = f"""
            <h1>U</h1>
            {parts}
        """
        return page("Milabench", body)
        
    return app


def main():
    # flask --app milabench.web.view:main run
    app = view_server({})
    return app


if __name__ == "__main__":
    main()
