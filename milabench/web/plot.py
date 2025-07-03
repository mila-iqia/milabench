import pandas as pd
from flask import Flask, jsonify, render_template_string, render_template
from flask_caching import Cache
from flask import request
import sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy import select, func, cast, TEXT, Float
from sqlalchemy import cast, Integer

from milabench.metrics.sqlalchemy import Exec, Metric, Pack, Weight, SavedQuery
from milabench.metrics.sqlalchemy import SQLAlchemy
from milabench.metrics.report import fetch_data, make_pivot_summary, fetch_data_by_id
from milabench.report import make_report
from .utils import database_uri, page, make_selection_key, make_filters, cursor_to_json, cursor_to_dataframe


def grouped_plot(group1_col, group2_col, group1_name, group2_name, exec_ids, metric, more=None, weighted=False,profile="default"):
    # group1 = Weight.group3
    # group2 = Weight.group4
    # exec_ids = (48, 47, 46)
    # metric = "rate"
    # more = [
    #     cast(Exec.meta['accelerators']['gpus']['0']['product'], TEXT).label("product"),
    #     cast(Exec.meta['pytorch']['build_settings']['TORCH_VERSION'], TEXT).label("pytorch"),
    # ]

    if more is None:
        more = []
    
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

    perf_formula = (func.avg(sub.c.avg_value)).label('perf')
    if weighted:
        perf_formula = (func.avg(sub.c.avg_value) * Weight.weight).label('perf')

    perf_per_bench = (
        select(
            Pack.name.label("bench"),
            sub.c.exec_id,
            perf_formula
        )
        .join(Pack, sub.c.pack_id == Pack._id)
        .join(Weight, Weight.pack == Pack.name)
        .where(
            Weight.profile == profile,
        )
        .group_by(Pack.name, sub.c.exec_id, Weight.weight)
    )

    sub = perf_per_bench.subquery()
    
    args = []
    filters = []
    group_by = []

    if group1_col is not None:
        args.append(group1_col.label(group1_name))
        group_by.append(group1_col)
        filters.extend([
            group1_col is not None,
            group1_col != "",
        ])
    
    if group2_col is not None:
        args.append(group2_col.label(group2_name))
        group_by.append(group2_col)
        filters.extend([
            group2_col is not None,
            group2_col != "",
        ])

    perf_per_group = (
        select(
            sub.c.exec_id,
            *args,
            func.avg(sub.c.perf).label(metric),
            *more
        )
        .join(Exec, Exec._id == sub.c.exec_id)
        .join(Weight, Weight.pack == sub.c.bench)
        .where(
            Weight.profile == profile,
            *filters
        )
        .group_by(sub.c.exec_id, *group_by, *more)
    )

    return perf_per_group


def sql_direct_report(group=None, profile="default"):
    """Use SQL to directly compute the report from the metrics.
    
    But we lose a bit of flexibility when it comes to how things get computed.
    But it is much faster.
    """
    # TODO: move this as arguments
    exec_ids = (48, 47, 46)
    more = [
        cast(Exec.meta['accelerators']['gpus']['0']['product'], TEXT).label("product"),
        cast(Exec.meta['pytorch']['build_settings']['TORCH_VERSION'], TEXT).label("pytorch"),
    ]
    
    # ---
    average_perf_per_pack = (
        select(
            Metric.pack_id,
            Metric.exec_id,
            # FIXME: we actually don't do an exact average
            # we drop min and max and then do the average
            func.avg(Metric.value).label("perf"),
            func.stddev(Metric.value).label("std"),
        )
        .where(
            Metric.name == "rate",
            Metric.exec_id.in_(exec_ids)
        )
        .group_by(Metric.pack_id, Metric.exec_id)
    )

    sub = average_perf_per_pack.subquery()

    perf_per_bench = (
        select(
            Pack.name.label("bench"),
            sub.c.exec_id,
            func.avg(sub.c.std).label("std"),
            # Count the number of processes per bench
            # NOTE: this does not work for multi node (should be 2 but will return 1)
            func.count().label("n"),
            func.avg(Pack.ngpu).label("ngpu"),
            func.avg(sub.c.perf).label("avg"),

            # HERE: This is a sum of the average
            # so multiple runs (i.e mono-gpu runs) are summed up
            func.sum(sub.c.perf).label("score")
        )
        .join(Pack, sub.c.pack_id == Pack._id)
        .group_by(Pack.name, sub.c.exec_id)
    )

    # This gives the raw score per bench before weighting
    sub = perf_per_bench.subquery()

    weighted_perf_per_bench = (
        select(
            sub.c.exec_id,
            sub.c.bench,
            func.avg(sub.c.ngpu).label("ngpu"),
            func.avg(sub.c.n).label("n"),
            func.avg(sub.c.avg).label("perf"),
            func.avg(sub.c.score).label("score"),
            func.avg(sub.c.std).label("std"),
            func.avg(func.log(sub.c.score + 1) * Weight.weight).label("log_score")
        ) 
        .join(Weight, Weight.pack == sub.c.bench)
        .where(
            Weight.profile == profile,
        ).group_by(sub.c.bench, sub.c.exec_id)
    )

    sub = weighted_perf_per_bench.subquery()
    
    # Final query to consolidate all the data into the report table we know
    perf_per_group = (
        select(
            sub.c.exec_id,

            sub.c.bench,
            
            *more,
            # fail,

            # n,
            func.avg(sub.c.n).cast(Float).label("n"),

            # ngpu
            func.avg(sub.c.ngpu).cast(Float).label("ngpu"),
            # perf
            
            func.avg(sub.c.perf).label("perf"),

            # sem%
            func.avg(sub.c.std / sub.c.perf).label("sem"),

            # std%
            func.avg(sub.c.std).label("std"),
            
            # peak_memory

            # score
            func.avg(sub.c.score).label("score"),

            # weight
            func.avg(Weight.weight).cast(Float).label("weight"),

            # No included usually
            func.avg(sub.c.log_score).label("log_score"),

            func.avg(Weight.priority).cast(Float).label("order"),
        )
        .join(Weight, Weight.pack == sub.c.bench)
        .join(Exec, Exec._id == sub.c.exec_id)
        .where(
            Weight.profile == profile,
            Weight.pack == sub.c.bench
        )
        .group_by(sub.c.bench, sub.c.exec_id, *more)
        .order_by("order")
    )

    return perf_per_group
