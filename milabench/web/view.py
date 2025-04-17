from io import StringIO
import os
import json
from contextlib import contextmanager

from flask import Flask, jsonify
import sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy import select, func, cast, TEXT

from milabench.metrics.sqlalchemy import Exec, Metric, Pack
from milabench.metrics.sqlalchemy import SQLAlchemy
from milabench.metrics.report import fetch_data, make_pivot_summary
from milabench.report import make_report
from .utils import database_uri, page


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

    @app.route('/api/ls/gpu')
    def api_ls_gpu():
        """Fetch a list of gpus milabench ran on"""
        jsonb_each_table = func.json_each(
            Exec.meta['accelerators']['gpus']
        ).table_valued('key', 'value').alias('gpu_entry')

        # Reference the 'value' column
        product_expr = jsonb_each_table.c.value.op('->>')('product')

        # Build the full query
        stmt = select(
            func.distinct(product_expr).label('product')
        ).select_from(
            Exec, jsonb_each_table  # This ensures both are in FROM clause
        )

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

    @app.route('/api/compare(<exec_a_id>,<exec_b_id>)')
    def api_metrics_compare(exec_a_id, exec_b_id):
        # stmt = sqlalchemy.select(Metric).where(Metric.exec_id == exec_id, Metric.pack_id == pack_id)

        results = []

        return results

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
