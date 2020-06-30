from hrepr import hrepr, HTML
from collections import defaultdict
from functools import reduce
import json
import os
import sys
import numpy as np
import glob
import operator
import pandas as pd
from pandas import DataFrame, Series
import math
from itertools import chain


nan = math.nan

H = HTML()


def extract_reports2(report_folder, filter, group):
    results = defaultdict(list)
    for root, _, files in os.walk(report_folder):
        for filename in files:
            filename = os.path.join(root, filename)
            if filename.endswith(".json"):
                with open(filename) as f:
                    contents = json.load(f)
                    if filter(contents, filename):
                        contents["__path__"] = filename
                        gr = group(contents)
                        contents["__group__"] = gr
                        results[gr].append(contents)
    return results


def _metrics(samples):
    return {
        "n": len(samples),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
    }


def _successful(entries):
    return [
        entry for entry in entries
        if entry["success"] and all(
            not math.isnan(value)
            for value in entry["metrics"].values()
        )
    ]


def summarize_group(group):
    n = len(group)
    good = _successful(group)
    failures = n - len(good)
    group = good
    total = [entry["timings"]["program"]["time"] for entry in group]
    rates = []
    pergpu = defaultdict(list)
    peak_memory = 0
    for entry in group:
        for k, v in entry["timings"].items():
            if k.startswith("train"):
                mean = np.mean(v["rates"][-20:-2])
                rates.append(mean)
                device = entry["environ"].get("CUDA_VISIBLE_DEVICES", ",").split(",")
                if len(device) == 1:
                    device, = device
                    pergpu[int(device)].append(mean)
                    peak = entry["gpu_monitor"][device]["memory"]["max"]
                    peak_memory = max(peak_memory, peak)
                else:
                    for _, v in entry["gpu_monitor"].items():
                        peak_memory = max(peak_memory, v["memory"]["max"])
    assert n > 0

    return {
        "n": n,
        "failures": failures,
        "total": _metrics(total),
        "train": _metrics(rates),
        "pergpu": {
            device_id: _metrics(results)
            for device_id, results in pergpu.items()
        },
        "peak_memory": peak_memory
    }


def summarize(report_folder, filter, group):
    reports = extract_reports2(report_folder, filter, group)
    return {
        group_name: summarize_group(group)
        for group_name, group in sorted(reports.items())
    }


def _make_row(summary, compare, weights, penalize_variance=False):
    row = {}
    if weights is not None:
        row["weight"] = weights["weight"] if weights else nan
    row["n"] = summary["n"] if summary else nan
    row["fail"] = summary["failures"] if summary else nan
    row["perf"] = summary["train"]["mean"] if summary else nan
    if compare is not None:
        row["perf_base"] = compare["train"]["mean"] if compare else nan
        row["perf_ratio"] = row["perf"] / row["perf_base"]
    # row["std"] = summary["train"]["std"] if summary else nan
    row["std%"] = summary["train"]["std"] / summary["train"]["mean"] if summary else nan
    if penalize_variance:
        penalty = min(summary["train"]["std"], row["perf"])
    else:
        penalty = 0
    row["perf_adj"] = (1 - row["fail"] / row["n"]) * (
        row["perf"] - penalty
    ) if summary else nan
    if compare is not None:
        if penalize_variance:
            penalty = min(compare["train"]["std"], row["perf_base"]) if compare else nan
        else:
            penalty = 0
        row["perf_base_adj"] = (
            row["perf_base"] - penalty if compare else nan
        )
        row["perf_ratio_adj"] = row["perf_adj"] / row["perf_base_adj"]
    return row


class WithClass:
    def __init__(self, value, klass):
        self.value = value
        self.klass = klass

    def __str__(self):
        return str(self.value)

    def __hrepr__(self, H, hrepr):
        return H.span[self.klass](str(self.value))


class PassFail(WithClass):
    def __init__(self, value, passfail):
        self.passfail = "PASS" if passfail else "FAIL"
        super().__init__(
            value=f"{value:10.2%} ({self.passfail})",
            klass=self.passfail
        )


class Table:
    def __init__(self, fields):
        self.fields = fields

    def __hrepr__(self, H, hrepr):
        tb = H.table["table"]()
        for k, v in self.fields.items():
            if isinstance(v, float):
                entry = H.td["float"](f"{v:.2f}")
            else:
                entry = H.td(hrepr(v))
            tb = tb(H.tr(
                H.th(k), entry
            ))
        return tb

    def __str__(self):
        l = max(map(len, self.fields.keys())) + 2
        lines = []
        for k, v in self.fields.items():
            v = f"{v:10.2f}" if isinstance(v, float) else v
            lines.append(f"{k + ':':{l}} {v}")
        return "\n".join(lines)


class Outputter:
    def __init__(self, stdout, html):
        self.stdout = stdout
        self.html_file = html and open(html, 'w')
        self._html(_table_style)

    def _html(self, contents):
        if self.html_file:
            print(contents, file=self.html_file)

    def _text(self, text):
        if self.stdout:
            print(text, file=self.stdout)

    def html(self, x):
        if not self.html_file:
            return
        if isinstance(x, DataFrame):
            sty = _style(x)
            self._html(sty._repr_html_())
        else:
            self._html(hrepr(x))

    def text(self, x):
        if not self.stdout:
            return
        if isinstance(x, DataFrame):
            self._text(x.to_string(formatters=_formatters))
        else:
            self._text(str(x))

    def print(self, x):
        self.text(x)
        self.html(x)

    def section(self, title):
        self.text("")
        self.text(title)
        self.text(f"-" * len(title))
        self.html(H.h2(title))

    def title(self, title):
        self._html(f'<html><head><title>{title}</title></head><body>')
        self.html(H.h2(title))
        self._text("=" * len(title))
        self._text(title)
        self._text("=" * len(title))

    def finalize(self):
        self.html("</body>")


def _report_pergpu(entries, measure='mean'):
    ngpus = max(len(v["pergpu"]) for v in entries.values())

    df = DataFrame({
        k: {i: v["pergpu"][i][measure] for i in range(ngpus)}
        for k, v in entries.items()
        if set(v["pergpu"].keys()) == set(range(ngpus))
    }).transpose()

    maxes = df.loc[:, list(range(ngpus))].max(axis=1).transpose()
    df = (df.transpose() / maxes).transpose()

    return df


def make_report(
    summary,
    compare=None,
    weights=None,
    html=None,
    compare_gpus=False,
    price=None,
    title=None,
    penalize_variance=False,
):
    all_keys = list(sorted({
        *(summary.keys() if summary else []),
        *(compare.keys() if compare else []),
        *(weights.keys() if weights else []),
    }))

    df = DataFrame(
        {
            key: _make_row(
                summary.get(key, {}),
                compare and compare.get(key, {}),
                weights and weights.get(key, {}),
                penalize_variance=penalize_variance,
            )
            for key in all_keys
        }
    ).transpose()

    out = Outputter(
        stdout=sys.stdout,
        html=html
    )

    out.title(title or "Benchmark results")
    out.print(df)

    if weights:
        out.section("Scores")

        def _score(column):
            # This computes a weighted geometric mean
            perf = df[column]
            weights = df["weight"]
            logscore = np.sum(np.log(perf) * weights) / np.sum(weights)
            return np.exp(logscore)

        score = _score('perf_adj')
        failure_rate = df["fail"].sum() / df["n"].sum()
        scores = {
            "Failure rate": PassFail(failure_rate, failure_rate <= 0.01),
            "Score": WithClass(f"{score:10.2f}", "score"),
        }
        if compare:
            score_base = _score('perf_base_adj')
            scores.update({
                "Score (baseline)": score_base,
                "Ratio": score / score_base
            })
        if price:
            rpp = price / score
            scores["Price ($)"] = f"{price:10.2f}"
            scores["RPP (Price / Score)"] = WithClass(f"{rpp:10.2f}", "rpp")
        out.print(Table(scores))

    if compare_gpus:
        for measure in ['mean', 'min', 'max']:
            df = _report_pergpu(summary, measure=measure)
            out.section(f"GPU comparison ({measure})")
            out.print(df)


_formatters = {
    'n': '{:.0f}'.format,
    'fail': '{:.0f}'.format,
    'std': '{:10.2f}'.format,
    'perf': '{:10.2f}'.format,
    'perf_base': '{:10.2f}'.format,
    'perf_adj': '{:10.2f}'.format,
    'perf_ratio': '{:10.2f}'.format,
    'perf_base_adj': '{:10.2f}'.format,
    'perf_ratio_adj': '{:10.2f}'.format,
    'std%': '{:6.1%}'.format,
    'weight': '{:4.2f}'.format,
    0: '{:.0%}'.format,
    1: '{:.0%}'.format,
    2: '{:.0%}'.format,
    3: '{:.0%}'.format,
    4: '{:.0%}'.format,
    5: '{:.0%}'.format,
    6: '{:.0%}'.format,
    7: '{:.0%}'.format,
    8: '{:.0%}'.format,
    9: '{:.0%}'.format,
    10: '{:.0%}'.format,
    11: '{:.0%}'.format,
    12: '{:.0%}'.format,
    13: '{:.0%}'.format,
    14: '{:.0%}'.format,
    15: '{:.0%}'.format,
}


_table_style = H.style("""
body {
    font-family: monospace;
}
td, th {
    text-align: right;
    min-width: 75px;
}
.table th {
    text-align: left;
}
.PASS {
    color: green;
    font-weight: bold;
}
.FAIL {
    color: red;
    font-weight: bold;
}
.score, .rpp {
    color: blue;
    font-weight: bold;
}
""")


def _style(df):

    def _redgreen(value):
        return 'color: green' if value else 'color: red'

    def _redblack(value):
        return 'color: red; font-weight: bold' if value else 'color: black'

    def _greyblack(value):
        return 'color: #888' if value else 'color: black'

    def _gpu_pct(value):
        if value >= 0.9:
            color = '#080'
        elif value >= 0.8:
            color = '#880'
        elif value >= 0.7:
            color = '#F80'
        else:
            color = '#F00'
        return f'color: {color}'

    def _ratio(value):
        if value >= 1.1:
            color = '#080'
        elif value >= 0.9:
            color = '#880'
        elif value >= 0.75:
            color = '#F80'
        else:
            color = '#F00'
        return f'color: {color}'

    def _fail(values):
        return (values > 0).map(_redblack)

    def _weight(values):
        return (values < 1).map(_greyblack)

    # Text formatting
    sty = df.style
    sty = sty.format(_formatters)

    # Format GPU efficiency map columns
    gpu_columns = set(range(16)) & set(df.columns)
    sty = sty.applymap(_gpu_pct, subset=list(gpu_columns))

    # sty.apply(_row, axis=1)

    # Format fail column
    if 'fail' in df.columns:
        sty = sty.apply(_fail, subset=['fail'])

    # Format weight column
    if 'weight' in df.columns:
        sty = sty.apply(_weight, subset=['weight'])

    # Format performance ratios
    for col in ['perf_ratio', 'perf_ratio_adj']:
        if col in df.columns:
            sty = sty.applymap(_ratio, subset=[col])

    return sty
