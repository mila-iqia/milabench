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


def grouped_plot(group1_col, group2_col, group1_name, group2_name, exec_ids, metric, more=None, weighted=False, profile="default", visibility=0):
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
        .join(Exec, Exec._id == Metric.exec_id)
        .where(
            Metric.name == metric,
            Metric.exec_id.in_(exec_ids),
            Exec.visibility == visibility 
        )
        .group_by(Metric.pack_id, Metric.exec_id)
    )

    sub = average_perf_per_pack.subquery()

    perf_formula = (func.avg(func.coalesce(sub.c.avg_value, 0))).label('perf')
    if weighted:
        perf_formula = (func.avg(func.coalesce(sub.c.avg_value, 0)) * Weight.weight).label('perf')

    perf_per_bench = (
        select(
            Weight.pack.label("bench"),
            sub.c.exec_id,
            perf_formula
        )
        .join(Pack, Pack._id == sub.c.pack_id)
        .outerjoin(Weight, Weight.pack == Pack.name)
        .where(
            Weight.profile == profile,
        )
        .group_by(Weight.pack, sub.c.exec_id, Weight.weight)
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
        .outerjoin(Weight, Weight.pack == sub.c.bench)  # Use LEFT JOIN to include all weights
        .where(
            Weight.profile == profile,
            *filters
        )
        .group_by(sub.c.exec_id, *group_by, *more)
    )

    return perf_per_group



def regular_average(exec_ids, visibility=0):
    average_perf_per_pack = (
        select(
            Metric.pack_id,
            Metric.exec_id,
            func.avg(Metric.value).label("perf"),
            func.stddev(Metric.value).label("std"),
        )
        .join(Exec, Exec._id == Metric.exec_id)
        .where(
            Exec.visibility == visibility,
            Metric.name == "rate",
            Metric.exec_id.in_(exec_ids)
        )
        .group_by(Metric.pack_id, Metric.exec_id)
    )

    return average_perf_per_pack


def average_drop_min_max(exec_ids, visibility=0):
    # Step 1: Assign row numbers or ranks to values per group
    ranked_metrics = (
        select(
            Metric.pack_id,
            Metric.exec_id,
            Metric.value,
            func.row_number().over(
                partition_by=(Metric.pack_id,),
                order_by=Metric.value.asc()
            ).label("row_asc"),
            func.row_number().over(
                partition_by=(Metric.pack_id,),
                order_by=Metric.value.desc()
            ).label("row_desc"),
        )
        .join(Exec, Exec._id == Metric.exec_id)
        .where(
            Exec.visibility == visibility,
            Metric.name == "rate",
            Metric.exec_id.in_(exec_ids)
        )
    ).subquery()

    # Step 2: Filter out min and max rows (row_asc = 1 or row_desc = 1)
    filtered_metrics = (
        select(
            ranked_metrics.c.pack_id,
            ranked_metrics.c.exec_id,
            ranked_metrics.c.value,
        )
        .where(
            ranked_metrics.c.row_asc > 1,
            ranked_metrics.c.row_desc > 1,
        )
    ).subquery()

    # Step 3: Aggregate the remaining values
    return (
        select(
            filtered_metrics.c.pack_id,
            filtered_metrics.c.exec_id,
            func.avg(filtered_metrics.c.value).label("perf"),
            func.stddev(filtered_metrics.c.value).label("std"),
            # func.count().label("count"),
        )
        .group_by(filtered_metrics.c.pack_id, filtered_metrics.c.exec_id)
    )


def perf_per_bench_query(exec_ids, profile="default", drop_min_max=True, visibility=0):
    if drop_min_max:
        average_perf_per_pack = average_drop_min_max(exec_ids, visibility=visibility)
    else:
        average_perf_per_pack = regular_average(exec_ids, visibility=visibility)

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

            # func.sum(sub.c.count).label("count"),

            # HERE: This is a sum of the average
            # so multiple runs (i.e mono-gpu runs) are summed up
            func.sum(sub.c.perf).label("score")
        )
        .join(Pack, sub.c.pack_id == Pack._id)
        .group_by(Pack.name, sub.c.exec_id)
    )

    return perf_per_bench


def weighted_perf_per_bench_query(exec_ids, profile="default", drop_min_max=True):
    perf_per_bench = perf_per_bench_query(exec_ids, profile, drop_min_max)

    # This gives the raw score per bench before weighting
    sub = perf_per_bench.subquery()

    if len(exec_ids) == 1:
        exec_id = func.coalesce(sub.c.exec_id, exec_ids[0])
    else:
        exec_id = sub.c.exec_id

    weighted_perf_per_bench = (
        select(
            exec_id.label("exec_id"),
            Weight.pack.label("bench"),  # Use Weight.pack instead of sub.c.bench to ensure all weights are included
            func.avg(func.coalesce(sub.c.ngpu, 0)).label("ngpu"),
            func.avg(func.coalesce(sub.c.n, 0)).label("n"),
            func.avg(func.coalesce(sub.c.avg, 0)).label("perf"),
            func.avg(func.coalesce(sub.c.score, 0)).label("score"),
            func.avg(func.coalesce(sub.c.std, 0)).label("std"),
            # func.avg(func.coalesce(sub.c.count, 0)).label("count"),
            func.avg(func.ln(func.coalesce(sub.c.score, 0) + 1) * Weight.weight * Weight.enabled.cast(Integer)).label("log_score")
        )
        .outerjoin(sub, Weight.pack == sub.c.bench) 
        # .outerjoin(Weight, Weight.pack == sub.c.bench)  # Use LEFT JOIN (outerjoin) instead of INNER JOIN
        .where(
            Weight.profile == profile,
        ).group_by(Weight.pack, sub.c.exec_id)
    )

    return weighted_perf_per_bench


def sql_direct_report(exec_ids, profile="default", drop_min_max=True, more=None):
    """Use SQL to directly compute the report from the metrics.

    But we lose a bit of flexibility when it comes to how things get computed.
    But it is much faster.
    """
    if more is None:
        more = []

    weighted_perf_per_bench = weighted_perf_per_bench_query(exec_ids, profile, drop_min_max)

    sub = weighted_perf_per_bench.subquery()

    weight_total = select(func.sum(Weight.weight * Weight.enabled.cast(Integer))).where(Weight.profile == profile).scalar_subquery()

    # Final query to consolidate all the data into the report table we know
    perf_per_group = (
        select(
            sub.c.exec_id,

            Weight.pack.label("bench"),

            *more,
            # fail,

            # n,
            func.avg(sub.c.n).cast(Float).label("n"),

            # ngpu
            func.avg(sub.c.ngpu).cast(Float).label("ngpu"),
            # perf

            func.avg(sub.c.perf).label("perf"),

            # sem% - handle division by zero
            func.avg(
                sqlalchemy.case(
                    (sub.c.perf > 0, sub.c.std / sub.c.perf), 
                    else_=0
                )
            ).label("sem"),

            # std%
            func.avg(sub.c.std).label("std"),

            # peak_memory

            # score
            func.avg(sub.c.score).label("score"),

            # weight
            func.avg(Weight.weight).cast(Float).label("weight"),

            # enabled
            func.avg(Weight.enabled.cast(Integer)).cast(Float).label("enabled"),

            # No included usually
            func.avg(sub.c.log_score).label("log_score"),

            func.avg(Weight.priority).cast(Float).label("order"),

            weight_total.label("weight_total"),

            # func.avg(sub.c.count).label("count"),
        )
        .outerjoin(Weight, Weight.pack == sub.c.bench)
        .join(Exec, Exec._id == sub.c.exec_id)
        .where(
            Weight.profile == profile,
            Weight.pack == sub.c.bench
        )
        .group_by(Weight.pack, sub.c.exec_id, *more)
        .order_by("order")
    )

    return perf_per_group



def pivot_query(sesh, rows, cols, values, filters, profile="default", visibility=0):
    from milabench.metrics.report import base_report_view

    filter_fields = [f['field'] for f in filters]
    names = {}

    groub_by_rows = [
        make_selection_key(key, names=names) for key in [*rows]
    ]

    selected_keys = groub_by_rows + [
        make_selection_key(key, names=names) for key in [*cols, *list(values.keys()), *filter_fields]
    ]

    query = base_report_view(*selected_keys, profile=profile, visibility=visibility)

    if filters:
        query = query.where(*make_filters(filters))

    sub = query.subquery()

    # This only fetches the unique columns
    col_names = [names[col] for col in cols]
    query = select(*[getattr(sub.c, col_name) for col_name in col_names]).distinct()
    final_columns = [row for row in sesh.execute(query)]  

    # Generate the SQL query to make the pivot
    agg = []
    for value_col, functions in values.items():
        for product_value in final_columns:
            frags = []
            conds = []

            for col_name, v in zip(col_names, product_value):
                frags.append(f"{col_name}={v}")
                conds.append(getattr(sub.c, col_name) == v)

            for f in functions:
                k_name = names.get(value_col)
                
                label = "/".join(frags + [k_name, f])
                value = getattr(sub.c, k_name)

                switch = sqlalchemy.case((sqlalchemy.and_(*conds), value), else_=None).cast(Float)
                fun = getattr(func, f)

                agg.append(fun(switch).label(label))

    final_group_by = [
        getattr(sub.c, names[key]) for key in rows
    ]

    return select(*final_group_by, *agg).group_by(*final_group_by).order_by(*final_group_by)