from io import StringIO
import os
from contextlib import contextmanager
from collections import defaultdict

import pandas as pd
from flask import Flask, jsonify, render_template_string
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

    @app.route('/html/report/<runame>')
    def html_report(runame):
        with SQLAlchemy(DATABASE_URI) as logger:
            df_post = fetch_data(logger.client, runame)

        names = list(df_post["run"].unique())
        full_name = names[0]
        replicated = make_pivot_summary(full_name, df_post)

        stream = StringIO()
        with open(os.devnull, "w") as devnull:
            make_report(replicated, stream=devnull, html=stream)

        return stream.getvalue()

    @app.route('/api/exec/list')
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

        return jsonify(results)

    @app.route('/api/exec/<int:exec_id>/packs')
    def api_packs_show(exec_id):
        stmt = sqlalchemy.select(Pack).where(Pack.exec_id == exec_id)

        results = []
        with sqlexec() as sess:
            cursor = sess.execute(stmt)
            for row in cursor:
                for col in row:
                    results.append(col.as_dict())

        return jsonify(results)

    @app.route('/html/exec/<int:exec_id>/packs/<int:pack_id>/metrics')
    def html_pack_metrics(exec_id, pack_id):
        import altair as alt
        from .utils import plot

        chart = alt.Chart(f"/api/exec/{exec_id}/packs/{pack_id}/metrics").mark_line().encode(
            x=alt.X("order", type="quantitative", scale=alt.Scale(zero=False), title="Time"),
            y=alt.Y("value", type="quantitative", scale=alt.Scale(zero=False)),
            color=alt.Color("gpu_id", type="ordinal"),
            tooltip=[
                alt.Tooltip("unit:N", title="Unit"),
            ]
        ).facet(
            facet=alt.Facet("name:N", title="Metric"),
            columns=4
        ).resolve_scale(y='independent', x='independent')

        return plot(chart.to_json())

    @app.route('/api/exec/<int:exec_id>/packs/<int:pack_id>/metrics')
    def api_pack_metrics(exec_id, pack_id):
        stmt = sqlalchemy.select(Metric).where(Metric.exec_id == exec_id, Metric.pack_id == pack_id)

        results = []
        with sqlexec() as sess:
            cursor = sess.execute(stmt)
            for row in cursor:
                for col in row:
                    results.append(col.as_dict())

        return jsonify(results)

    @app.route('/api/exec/<int:exec_id>/packs/<string:pack_name>/metrics')
    def api_pack_summary_metrics(exec_id, pack_name):
        stmt = sqlalchemy.select(Metric).where(Metric.exec_id == exec_id, Pack.name.startswith(pack_name)).join(Pack, Metric.pack_id == Pack._id)

        results = []
        with sqlexec() as sess:
            cursor = sess.execute(stmt)
            for row in cursor:
                for col in row:
                    results.append(col.as_dict())

        return jsonify(results)

    @app.route('/api/keys')
    def api_ls_keys():
        here = os.path.dirname(__file__)
        with open(os.path.join(here, "template", "key.txt"), "r") as fp:
            return jsonify(fp.readlines())

    @app.route('/html/pivot')
    def html_pivot():
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

        pivot_spec = {
            "rows": ["run", "gpu", "bench"],
            "cols": ["metric"],
            "values": {
                "value": ["mean", "max"],
            },
        }

        df = pd.DataFrame(results)
        filtered = df

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
            ('value', 'mean', 'rate'),
            ('value', 'mean', 'gpu.load'),
            ('value', 'mean', 'gpu.memory'),
            ('value', 'mean', 'gpu.power'),
            ('value', 'mean', 'gpu.temperature'),
            ('value', 'mean', 'walltime'),
            ('value', 'mean', 'loss'),
            ('value', 'mean', 'memory_peak'),
            ('value', 'mean', 'return_code'),
        ]

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

        return df.to_html(formatters=formatters, classes=["table", "table-striped", "table-hover", "table-sm"], na_rep="")

    @app.route('/api/gpu/list')
    def api_ls_gpu():
        stmt = select(func.distinct(cast(Exec.meta["accelerators"]["gpus"]["0"]["product"], TEXT)))
        with sqlexec() as sess:
            return jsonify(sess.execute(stmt).scalars().all())

    @app.route('/api/metrics/list')
    def api_ls_metrics():
        stmt = select(func.distinct(Metric.name))
        with sqlexec() as sess:
            return jsonify(sess.execute(stmt).scalars().all())

    @app.route('/api/pytorch/list')
    def api_ls_pytorch():
        stmt = select(func.distinct(cast(Exec.meta["pytorch"]["torch"], TEXT)))
        with sqlexec() as sess:
            return jsonify(sess.execute(stmt).scalars().all())

    @app.route('/api/milabench/list')
    def api_ls_milabench():
        stmt = select(func.distinct(cast(Exec.meta["milabench"]["tag"], TEXT)))
        with sqlexec() as sess:
            return jsonify(sess.execute(stmt).scalars().all())

    @app.route('/api/exec/<id>')
    def api_exec_show(id):
        stmt = sqlalchemy.select(Exec).where(Exec._id == id)
        with sqlexec() as sess:
            cursor = sess.execute(stmt)
            for row in cursor:
                result = row[0]
                return jsonify(result.as_dict())

        return jsonify({})

    @app.route('/')
    def index():
        return page("Milabench", render_template_string(open(os.path.join(os.path.dirname(__file__), "template", "main.html")).read()))

    return app


def main():
    app = view_server({})
    return app


if __name__ == "__main__":
    main()
