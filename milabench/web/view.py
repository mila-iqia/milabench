from io import StringIO
import os
import json
from contextlib import contextmanager
from collections import defaultdict
import base64

import pandas as pd
from flask import Flask, jsonify, render_template_string, render_template
from flask_caching import Cache
from flask import request
import sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy import select, func, cast, TEXT

from milabench.metrics.sqlalchemy import Exec, Metric, Pack, Weight, SavedQuery
from milabench.metrics.sqlalchemy import SQLAlchemy
from milabench.metrics.report import fetch_data, make_pivot_summary, fetch_data_by_id
from milabench.report import make_report
from .utils import database_uri, page, make_selection_key, make_filters, cursor_to_json, cursor_to_dataframe


class MultiIndexFormater:
    """Format a dataframe using the last element on a multi index"""
    def __init__(self, df, default_float="{:.2f}".format):
        self.df = df
        self.default = default_float
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

        if isinstance(item, str):
            return default

        return self.default


def gradient(x, mn, mx):
    import numpy as np

    c1 = np.array([255, 0, 0])
    c2 = np.array([255, 255, 255])
    c3 = np.array([0, 255, 0])

    pct = (x - mn) / (mx - mn)

    if pct < 0.5:
        t = pct / 0.5
        return (1 - t) * c1 + t * c2

    else:
        t = (pct - 0.5) / 0.5
        return (1 - t) * c2 + t * c3


def conditional_format(v, props=''):
    color = gradient(v, mn=0.5, mx=1.5)
    return f"background: rgb({color[0]}, {color[1]}, {color[2]})"


def pandas_to_html(df, default_float="{:.2f}".format):
    fmt = MultiIndexFormater(df, default_float=default_float)

    table = df.to_html(
        formatters=fmt,
        classes=["table", "table-striped", "table-hover", "table-sm"],
        na_rep="",
        justify="right"
    )

    return page("df", table, more_css="""
        .table {
            width: auto;
        }
    """)


def pandas_to_html_relative(df, default_float="{:.2f}".format):
    table = (df.style
        .map(conditional_format)
        .format(precision=2, thousands="'", decimal=".")
        .set_table_attributes("class='table table-striped table-hover table-sm'")
        .to_html()
    )

    return page("df", table, more_css="""
        .table {
            width: auto;
        }
    """)


def view_server(config):
    """Display milabench results"""

    DATABASE_URI = database_uri()

    app = Flask(__name__)
    app.config.update(config)
    app.config.update({
        "CACHE_TYPE": "SimpleCache",
        "CACHE_DEFAULT_TIMEOUT": 300
    })

    cache = Cache(app)

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
    @app.route('/api/exec/list/<int:limit>')
    def api_exec_list(limit=25):
        stmt = sqlalchemy.select(Exec).order_by(Exec._id.desc()).limit(limit)

        results = []
        with sqlexec() as sess:
            cursor = sess.execute(stmt)
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

    @app.route('/api/profile/list')
    def api_ls_profile():
        stmt = select(func.distinct(Weight.profile))
        with sqlexec() as sess:
            return jsonify(sess.execute(stmt).scalars().all())

    @app.route('/api/profile/show/<string:profile>')
    def api_show_profile(profile):
        stmt = select(Weight).where(Weight.profile == profile)

        results = []
        with sqlexec() as sess:
            cursor = sess.execute(stmt)
            for row in cursor:
                for col in row:
                    results.append(col.as_dict())


        results.sort(key=lambda x: x['priority'])

        return jsonify(results)

    @app.route('/api/profile/save/<string:profile>', methods=['POST'])
    def api_save_profile(profile):
        from flask import request
        weights = request.json

        with sqlexec() as sess:
            for weight in weights:
                stmt = (
                    sqlalchemy.update(Weight)
                    .where(Weight._id == weight['_id'])
                    .values(
                        weight=weight['weight'],
                        priority=weight['priority'],
                        enabled=weight['enabled'],
                        group1=weight.get('group1'),
                        group2=weight.get('group2'),
                        group3=weight.get('group3'),
                        group4=weight.get('group4'),
                    )
                )
                sess.execute(stmt)
            sess.commit()

        return jsonify({"status": "success"})

    @app.route('/api/profile/copy', methods=['POST'])
    def api_copy_profile():
        from flask import request
        data = request.json
        source_profile = data['sourceProfile']
        new_profile = data['newProfile']

        with sqlexec() as sess:
            # Get all weights from source profile
            stmt = select(Weight).where(Weight.profile == source_profile)
            source_weights = []
            cursor = sess.execute(stmt)
            for row in cursor:
                for col in row:
                    source_weights.append(col.as_dict())

            # Create new weights for the new profile
            for weight in source_weights:
                new_weight = Weight(
                    profile=new_profile,
                    pack=weight['pack'],
                    weight=weight['weight'],
                    priority=weight['priority'],
                    enabled=weight['enabled'],
                    group1=weight.get('group1'),
                    group2=weight.get('group2'),
                    group3=weight.get('group3'),
                    group4=weight.get('group4'),
                )
                sess.add(new_weight)
            sess.commit()

        return jsonify({"status": "success"})

    @app.route('/api/query/list')
    def api_ls_saved():
        stmt = select(func.distinct(SavedQuery.name))
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

    @app.route('/api/exec/explore')
    def api_explore():
        from flask import request
        fields = {}
        tables = []
        sql_filters = None

        if request.args.get('filters'):
            filters = json.loads(base64.b64decode(request.args.get('filters')))
            # extract the fields that are queried upon
            # we will add them to the query and display the values
            sql_filters = make_filters(filters, fields=fields, used_tables=tables)

        table = (
            sqlalchemy.select(
                Exec._id.label("id"),
                Exec.name.label("run"),
                # Pack.name.label("bench"),
                *fields.values()
            )
            #
            #
        ).distinct(Exec._id)

        if sql_filters:
            table = table.where(*sql_filters)

        if 'Pack' in tables:
            table = table.join(Pack, Exec._id == Pack._id)

        if "Metric" in tables:
            table = table.join(Metric, Exec._id == Metric.exec_id)

        with sqlexec() as sess:
            cursor = sess.execute(table)
            columns = list(cursor.keys())
            results = []

            for row in cursor:
                results.append({k: v for k, v in zip(columns, row)})

        return jsonify(results)


    #
    # html routes
    #

    def fetch_data_type(client, run_id, profile="default"):
        if isinstance(run_id, str):
            return fetch_data(client, run_id, profile=profile)
        else:
            return fetch_data_by_id(client, run_id, profile=profile)

    def report(run_id, profile="default"):
        with SQLAlchemy(DATABASE_URI) as logger:
            df_post = fetch_data_type(logger.client, run_id, profile=profile)

        names = list(df_post["run"].unique())

        if len(names) > 0:
            print("multiple run report")

        full_name = names[0]
        replicated = make_pivot_summary(full_name, df_post)

        stream = StringIO()
        with open(os.devnull, "w") as devnull:
            make_report(replicated, stream=devnull, html=stream, weights=replicated)

        return stream.getvalue()

    @app.route('/html/report/<string:runame>')
    def html_report_name(runame):
        profile = request.cookies.get('scoreProfile')
        
        return report(runame, profile=profile)

    @app.route('/html/report/<int:run_id>')
    def html_report(run_id):
        profile = request.cookies.get('scoreProfile')

        return report(run_id, profile=profile)

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

    @app.route('/api/report/fast')
    def api_report_fast():
        from .plot import sql_direct_report

        profile = request.cookies.get('scoreProfile')

        with sqlexec() as sess:
            stmt = sql_direct_report(profile=profile)

            cursor = sess.execute(stmt)

            results = cursor_to_json(cursor)

        return jsonify(results)

    @app.route('/api/scaling')
    def api_scaling():
        """Fetch scaling data from the scaling configuration files"""
        from milabench.analysis.scaling import read_config, folder_path

        gpus = request.args.getlist("gpus")

        if len(gpus) == 0:
            gpus = list(os.listdir(folder_path))
            gpus.remove("default.yaml")

            gpus = [gpu.split(".")[0] for gpu in gpus]

        output = []
        for gpu in gpus:
            read_config(f"{gpu}.yaml", output)
        
        return output

    @app.route('/html/scaling/x=<string:x>/y=<string:y>')
    def scaling_plot(x, y): 
        """Fetch scaling data from the scaling configuration files"""
        import altair as alt
        from .utils import plot

        print(x, y)

        chart = (
            alt.Chart(f"/api/scaling").mark_point().encode(
                    x=f"{x}:Q",
                    y=f"{y}:Q",
                    shape="gpu:N",
                    color="gpu:N",
                    size="perf:Q",
                )
                .facet(
                    facet=alt.Facet("bench:N", title="Benchmark"),
                    columns=4
                )
        ).resolve_scale(y='independent', x='independent', size='independent')

        return plot(chart.to_json())

    @app.route('/api/grouped/plot')
    def api_grouped_plot():
        from .plot import grouped_plot

        profile = request.cookies.get('scoreProfile')

        with sqlexec() as sess:
            stmt = grouped_plot(profile=profile)

            cursor = sess.execute(stmt)

            results = cursor_to_json(cursor)

        return jsonify(results)
    
    @app.route('/html/grouped/plot')
    def html_grouped_plot():
        import altair as alt
        from .utils import plot

        # TODO: make those arguments
        g1 = "group1"
        g2 = "group2"
        color = "pytorch"

        row_order = ["fp16", "tf32", "fp32"]
        column_order = ["FLOPS", "BERT", "CONVNEXT"]

        # ----

        chart = alt.Chart(f"/api/grouped/plot").mark_bar().encode(
            y=alt.Y(color, type="nominal", scale=alt.Scale(zero=False), title="Pytorch"),
            x=alt.X("perf", type="quantitative", scale=alt.Scale(zero=False)),
            
            color=alt.Color(color, type="nominal"),

            row=alt.Row(f"{g2}", type="nominal", title="Group1", sort=row_order),
            column=alt.Column(f"{g1}", type="nominal", title="Group1", sort=column_order),
        )

        return plot(chart.to_json())


    @cache.memoize(timeout=3600)
    def cached_query(rows, cols, values, filters, profile="default"):
        from milabench.metrics.report import base_report_view

        filter_fields = [f['field'] for f in filters]

        selected_keys = [
            make_selection_key(key) for key in [*rows, *cols, *values, *filter_fields]
        ]

        table = base_report_view(*selected_keys, profile=profile)

        if filters:
            table = table.where(*make_filters(filters))

        with sqlexec() as sess:
            cursor = sess.execute(table)
            results = cursor_to_json(cursor)

        return results

    def make_pivot(profile="default"):
        # If no parameters are provided, serve the pivot builder interface
        if not request.args:
            return render_template('pivot.html')

        args = request.args

        rows = args.get('rows', '').split(',') if args.get('rows') else ["run", "gpu", "pytorch", "bench"]
        cols = args.get('cols', '').split(',') if args.get('cols') else ["metric"]
        values = args.get('values', '').split(',') if args.get('values') else ["mean", "max"]
        filters = []

        if args.get('filters'):
            filters = json.loads(base64.b64decode(args.get('filters')))

        results = cached_query(rows, cols, values, filters, profile=profile)

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

        overall = pd.pivot_table(
            df,
            values=pivot_spec["values"].keys(),
            index=pivot_spec["rows"],
            columns=pivot_spec["cols"],
            aggfunc=pivot_spec["values"],
            dropna=True,
        )

        import numpy as np

        pack_name = "Pack:name"
        if pack_name in overall.index.names:
            priority_map = df.drop_duplicates(subset=pack_name, keep='first').set_index(pack_name)['priority']

            bench_vals = overall.index.get_level_values(pack_name)
            priorities = bench_vals.map(priority_map)

            overall = overall.iloc[priorities.argsort()]

            # Compute the score
            rate_cols_mask = overall.columns.get_level_values('Metric:name') == 'rate'
            rate_columns = overall.columns[rate_cols_mask]

            if len(rate_columns) > 0:
                weight_map = df.drop_duplicates(subset=pack_name, keep='first').set_index(pack_name)['weight']
                weights = bench_vals.map(weight_map)

                scores = {}
                for col in rate_columns:
                    x = overall[col]
                    weighted_log_sum = (np.log(x + 1) * weights).sum()
                    weight_sum = sum(weights.values)
                    scores[col] = np.exp(weighted_log_sum / weight_sum)

                # Add the score as a row to overall
                scores_series = pd.Series(scores)
                score_row = pd.DataFrame([scores_series], columns=overall.columns)

                existing_index = overall.index[0]
                pack_name_pos = overall.index.names.index("Pack:name")

                if isinstance(overall.index, pd.MultiIndex):
                    new_index_label = list(existing_index)
                    new_index_label[pack_name_pos] = "score"
                    new_index_label = tuple(new_index_label)
                    score_row.index = pd.MultiIndex.from_tuples([new_index_label], names=overall.index.names)
                else:
                    score_row.index = pd.Index(["score"])

                overall = pd.concat([overall, score_row])

        # We need to reorder the df by the same order
        return overall

    @app.route('/html/relative/pivot')
    def html_relative_pivot():
        # retrieve the cookie `scoreProfile` and use that as the profile
        profile = request.cookies.get('scoreProfile')
        df = make_pivot(profile=profile)

        first_col = df.iloc[:, 0]

        df = df.div(first_col, axis=0)

        return pandas_to_html_relative(df)

    @app.route('/html/pivot')
    def html_pivot():
        profile = request.cookies.get('scoreProfile')

        df = make_pivot(profile)

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
