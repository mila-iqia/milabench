import pandas as pd
from flask import Flask, jsonify, render_template_string, render_template
from flask_caching import Cache
from flask import request
import sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy import select, func, cast, TEXT
from sqlalchemy import cast, Integer

from milabench.metrics.sqlalchemy import Exec, Metric, Pack, Weight, SavedQuery
from milabench.metrics.sqlalchemy import SQLAlchemy
from milabench.metrics.report import fetch_data, make_pivot_summary, fetch_data_by_id
from milabench.report import make_report
from .utils import database_uri, page, make_selection_key, make_filters, cursor_to_json, cursor_to_dataframe


def grouped_plot(group=None, profile="default"):
    # TODO: move this as arguments
    group1 = Weight.group3
    group2 = Weight.group4
    exec_ids = (48, 47, 46)
    metric = "rate"
    more = [
        cast(Exec.meta['accelerators']['gpus']['0']['product'], TEXT).label("product"),
        cast(Exec.meta['pytorch']['build_settings']['TORCH_VERSION'], TEXT).label("pytorch"),
    ]
    
    # ---
    average_perf_per_pack = (
        select(
            Metric.pack_id,
            Metric.exec_id,
            func.avg(Metric.value).label("avg_value")
        )
        .where(
            Metric.name == metric,
            Metric.exec_id.in_(exec_ids)
        )
        .group_by(Metric.pack_id, Metric.exec_id)
    )

    sub = average_perf_per_pack.subquery()

    perf_per_bench = (
        select(
            Pack.name.label("bench"),
            sub.c.exec_id,
            func.avg(sub.c.avg_value).label("perf")
        )
        .join(Pack, sub.c.pack_id == Pack._id)
        .group_by(Pack.name, sub.c.exec_id)
    )

    sub = perf_per_bench.subquery()
    
    perf_per_group = (
        select(
            sub.c.exec_id,
            group1.label("group1"),
            group2.label("group2"),
            func.avg(sub.c.perf).label("perf"),
            *more
        )
        .join(Weight, Weight.pack == sub.c.bench)
        .join(Exec, Exec._id == sub.c.exec_id)
        .where(
            Weight.profile == profile,
            group1 is not None,
            group2 is not None,
            group1 != "",
            group2 != "",
        )
        .group_by(sub.c.exec_id, group1, group2, *more)
    )

    return perf_per_group



