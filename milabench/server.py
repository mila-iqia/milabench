
from flask import Flask, request


app = Flask(__name__)


@app.route("/metric/summary/save", methods=['POST'])
def metric_summary_save():
    """Receive a summary report"""
    content = request.json


    