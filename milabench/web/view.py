from io import StringIO
import os
import json


from flask import Flask, jsonify

from milabench.metrics.sqlalchemy import SQLAlchemy
from milabench.metrics.report import fetch_data, make_pivot_summary
from milabench.report import make_report
from .utils import database_uri, page


def view_server(config):
    """Display milabench results"""

    DATABASE_URI = database_uri()

    app = Flask(__name__)
    app.config.update(config)
    
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

        full_name = df_post["run"].unique()[0]

        replicated = make_pivot_summary(full_name, df_post)

        stream = StringIO()
        
        with open(os.devnull, "w") as devnull :
            make_report(replicated, stream=devnull, html=stream)

        return stream.getvalue()
    
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
