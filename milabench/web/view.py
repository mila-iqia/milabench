from io import StringIO
import os
import json
from contextlib import contextmanager
from collections import defaultdict
import base64

import pandas as pd
from flask import Flask, jsonify, render_template_string, render_template
import sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy import select, func, cast, TEXT

from milabench.metrics.sqlalchemy import Exec, Metric, Pack
from milabench.metrics.sqlalchemy import SQLAlchemy
from milabench.metrics.report import fetch_data, make_pivot_summary, fetch_data_by_id
from milabench.report import make_report
from .utils import database_uri, page, make_selection_key, make_filters, cursor_to_json, cursor_to_dataframe


class MultiIndexFormater:
    """Format a dataframe using the last element on a multi index"""
    def __init__(self, df):
        self.df = df
        self.style = {
            "gpu.load": "{:.2%}".format,
            "gpu.memory": "{:.2%}".format,
            "gpu.power": "{:.2f}".format,
            "gpu.temperature": "{:.2f}".format,
            "loss": "{:.2f}".format,
            "walltime": "{:.2f}".format,
            "rate": "{:.2f}".format,
            "return_code": "{:.0f}".format,
            "memory_peak": "{:.0f}".format,
        }

    def __len__(self): 
        return len(self.df.columns)
    
    def get(self, item, default=None):
        for col in item:
            if col in self.style:
                return self.style.get(col)
            
        return default


def pandas_to_html(df):
    fmt = MultiIndexFormater(df)

    return df.to_html(
        formatters=fmt, 
        classes=["table", "table-striped", "table-hover", "table-sm"], 
        na_rep=""
    )


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

    #
    # AI routes
    #

    @app.route('/api/summary/<runame>')
    def api_summary(runame):
        with SQLAlchemy(DATABASE_URI) as logger:
            df_post = fetch_data(logger.client, runame)

        multirun = {}
        for real_runname in df_post["run"].unique():
            multirun[real_runname] = make_pivot_summary(real_runname, df_post)

        return jsonify(multirun)

    @app.route('/api/exec/list')
    def api_exec_list():
        stmt = sqlalchemy.select(Exec)

        results = []
        with sqlexec() as sess:
            cursor = sess.execute(stmt)
            columns = list(cursor.keys())
            for row in cursor:
                for col in row:
                    results.append(col.as_dict())

        return results

    @app.route('/api/exec/<int:exec_id>/packs')
    def api_packs_show(exec_id):
        stmt = sqlalchemy.select(Pack).where(Pack.exec_id == exec_id)

        results = []
        with sqlexec() as sess:
            cursor = sess.execute(stmt)
            for row in cursor:
                for col in row:
                    results.append(col.as_dict())

        # group packs by benchmarks
        grouped = defaultdict(list)
        for row in results:
            grouped[row["name"]] = row

        return jsonify(results)

    @app.route('/api/exec/<int:exec_id>/packs/<int:pack_id>/metrics')
    def api_pack_metrics(exec_id, pack_id):
        stmt = sqlalchemy.select(Metric).where(Metric.exec_id == exec_id, Metric.pack_id == pack_id)

        results = []
        with sqlexec() as sess:
            cursor = sess.execute(stmt)
            for row in cursor:
                for col in row:
                    results.append(col.as_dict())

        return results

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
        with open(os.path.join(here, "template", "keys.txt"), "r") as fp:
            return jsonify([line.strip() for line in fp.readlines()])

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


    #
    # html routes
    #

    def fetch_data_type(client, run_id):
        if isinstance(run_id, str):
            return fetch_data(client, run_id)
        else:
            return fetch_data_by_id(client, run_id)

    def report(run_id):
        with SQLAlchemy(DATABASE_URI) as logger:
            df_post = fetch_data_type(logger.client, run_id)

        names = list(df_post["run"].unique())

        if len(names) > 0:
            print("multiple run report")

        full_name = names[0]
        replicated = make_pivot_summary(full_name, df_post)

        stream = StringIO()
        with open(os.devnull, "w") as devnull:
            make_report(replicated, stream=devnull, html=stream)

        return stream.getvalue()

    @app.route('/html/report/<string:runame>')
    def html_report_name(runame):
        return report(runame)

    @app.route('/html/report/<int:run_id>')
    def html_report(run_id):
        return report(run_id)

    @app.route('/html/exec/<int:exec_id>/packs/<pack_id>/metrics')
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

    @app.route('/html/form/pivot')
    def html_format_pivot():
        with open("/home/newton/work/milabench_dev/milabench/milabench/web/template/pivot.html", "r") as fp:
            return render_template_string(fp.read())

    @app.route('/html/pivot')
    def html_pivot():
        from milabench.metrics.report import base_report_view
        from flask import request

        # If no parameters are provided, serve the pivot builder interface
        if not request.args:
            return render_template('pivot.html')

        # Get form parameters with defaults
        rows = request.args.get('rows', '').split(',') if request.args.get('rows') else ["run", "gpu", "pytorch", "bench"]
        cols = request.args.get('cols', '').split(',') if request.args.get('cols') else ["metric"]
        values = request.args.get('values', '').split(',') if request.args.get('values') else ["mean", "max"]
        filters = []

        if request.args.get('filters'):
            filters = json.loads(base64.b64decode(request.args.get('filters')))

        filter_fields = [f['field'] for f in filters]

        selected_keys = [
            make_selection_key(key) for key in [*rows, *cols, *values, *filter_fields]
        ]

        table = base_report_view(*selected_keys)

        if filters:
            table = table.where(*make_filters(filters))

        with sqlexec() as sess:
            cursor = sess.execute(table)
            results = cursor_to_json(cursor)

        pivot_spec = {
            "rows": rows,
            "cols": cols,
            "values": {
                v: ["mean"] for v in values
            },
            # We make the filter in SQL so we have less data to process
            "filters": []
        }

        df = pd.DataFrame(results)

        filtered = df
        for filter in pivot_spec.get("filters", []):
            filtered = filtered.query(filter)
    
        if filtered.empty:
            return pandas_to_html(filtered)

        overall = pd.pivot_table(
            filtered,
            values=pivot_spec["values"].keys(),
            index=pivot_spec["rows"],
            columns=pivot_spec["cols"],
            aggfunc=pivot_spec["values"],
            dropna=True,
        )


        # metrics = df["Metric:name"].unique()

        # columns = []
        # for col in pivot_spec["cols"]:
        #     columns.append(df[col].unique())

        # Dynamically build column order based on selected values
        column_order = []
        # FIXME I need to add pivot columns here

        df = overall
        if column_order:
            df = df[column_order]

        if "Pack:name" in df:
            priority_map = defaultdict(int)
            df = df.reset_index()

            for i, k in enumerate(sorted(list(set(df['Pack:name'])))):
                priority_map[k] = i

            priority_map.update({
                "fp16": -1,
                "bf16": -2,
                "tf32": -3,
                "fp32": -4,
            })

            df['_priority'] = df['Pack:name'].map(priority_map)
            df = df.sort_values('_priority').drop(columns=['_priority'])
            df = df.set_index(rows)

        return pandas_to_html(df)

    @app.route('/')
    def index():
        parts = []

        parts = "".join(parts)

        body = f"""
            <h1>U</h1>
            {parts}
        """

        with open("/home/newton/work/milabench_dev/milabench/milabench/web/template/main.html", "r") as fp:
            return render_template_string(fp.read())

        # return ()
        return page("Milabench", body)

    return app


def main():
    # flask --app milabench.web.view:main run
    app = view_server({})
    return app


if __name__ == "__main__":
    main()
