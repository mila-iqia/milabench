import math
import sys

import numpy as np
from hrepr import HTML, hrepr
from pandas import DataFrame

from milabench.utils import error_guard

nan = math.nan

H = HTML()


@error_guard({})
def _make_row(summary, compare, weights):
    mkey = "train_rate"
    metric = "mean"
    row = {}

    row["n"] = summary["n"] if summary else nan
    row["fail"] = summary["failures"] if summary else nan
    row["perf"] = summary[mkey][metric] if summary else nan

    if compare:
        row["perf_base"] = compare[mkey][metric]
        row["perf_ratio"] = row["perf_adj"] / row["perf_base"]

    row["std%"] = summary[mkey]["std"] / summary[mkey][metric] if summary else nan
    row["sem%"] = summary[mkey]["sem"] / summary[mkey][metric] if summary else nan
    # row["iqr%"] = (summary[mkey]["q3"] - summary[mkey]["q1"]) / summary[mkey]["median"] if summary else nan
    row["peak_memory"] = (
        max(
            (data["memory"]["max"] for data in summary["gpu_load"].values())
            if summary["gpu_load"]
            else [-1]
        )
        if summary
        else nan
    )

    # Sum of all the GPU performance
    # to get the overall perf of the whole machine
    acc = 0
    for _, metrics in summary["per_gpu"].items():
        acc += metrics[metric]

    success_ratio = 1 - row["fail"] / row["n"]
    score = (acc if acc > 0 else row["perf"]) * success_ratio

    row["score"] = score
    print(score)
    row["weight"] = weights.get("weight", summary["weight"])
    # ----

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
        super().__init__(value=f"{value:10.2%} ({self.passfail})", klass=self.passfail)


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
            tb = tb(H.tr(H.th(k), entry))
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
        self.html_file = html and open(html, "w")
        self._html(_table_style)
        self._html(_error_style)

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
        self._html(f"<html><head><title>{title}</title></head><body>")
        self.html(H.h2(title))
        self._text("=" * len(title))
        self._text(title)
        self._text("=" * len(title))

    def finalize(self):
        self.html(H.raw("</body>"))


def _report_pergpu(entries, measure="50"):
    ngpus = max(len(v["per_gpu"]) for v in entries.values())

    # {"bench": {"0": <value>, "1": <value>} ... }
    data = {}
    for k, v in entries.items():
        values = dict()
        data[k] = values

        for i in range(ngpus):
            gpu = v["per_gpu"].get(i, dict())
            values[i] = gpu.get(measure, float("nan"))

    df = DataFrame(data).transpose()
    maxes = df.loc[:, list(range(ngpus))].max(axis=1).transpose()
    df = (df.transpose() / maxes).transpose()

    return df


columns_order = {
    "fail": 1,
    "n": 2,
    "perf": 3,
    "perf_adj": 4,
    "sem%": 5,
    "std%": 6,
    "peak_memory": 7,
    "score": 8,
    "weight": 9,
}


def make_dataframe(summary, compare=None, weights=None):
    if weights is None:
        weights = dict()

    all_keys = list(
        sorted(
            {
                *(summary.keys() if summary else []),
                *(compare.keys() if compare else []),
                *(weights.keys() if weights else []),
            }
        )
    )

    return DataFrame(
        {
            key: _make_row(
                summary.get(key, {}),
                compare and compare.get(key, {}),
                weights and weights.get(key, {}),
            )
            for key in all_keys
        }
    ).transpose()







@error_guard({})
def make_report(
    summary,
    compare=None,
    html=None,
    compare_gpus=False,
    price=None,
    title=None,
    sources=None,
    errdata=None,
    weights=None,
    stream=sys.stdout
):
    if weights is None:
        weights = dict()

    df = make_dataframe(summary, compare, weights)
    
    # Reorder columns
    df = df[sorted(df.columns, key=lambda k: columns_order.get(k, 0))]
    
    out = Outputter(stdout=stream, html=html)

    if sources:
        if isinstance(sources, str):
            sources = [sources]
        for source in sources:
            out.print(f"Source: {source}")
    out.title(title or "Benchmark results")
    out.print(df)

    out.section("Scores")

    def _score(column):
        # This computes a weighted geometric mean
        perf = df[column]
        weights = df["weight"]
        logscore = np.sum(np.log(perf) * weights) / np.sum(weights)
        return np.exp(logscore)

    score = _score("score")
    failure_rate = df["fail"].sum() / df["n"].sum()
    scores = {
        "Failure rate": PassFail(failure_rate, failure_rate <= 0.01),
        "Score": WithClass(f"{score:10.2f}", "score"),
    }
    if compare:
        score_base = _score("perf_base")
        scores.update({"Score (baseline)": score_base, "Ratio": score / score_base})

    if price:
        rpp = price / score
        scores["Price ($)"] = f"{price:10.2f}"
        scores["RPP (Price / Score)"] = WithClass(f"{rpp:10.2f}", "rpp")

    out.print(Table(scores))

    if compare_gpus:
        for measure in ["mean", "min", "max"]:
            df = _report_pergpu(summary, measure=measure)
            out.section(f"GPU comparison ({measure})")
            out.print(df)

    if errdata:
        out.section(f"Errors")
        out._text(f"{len(errdata)} errors, details in HTML report.")
        boxid = 0
        for filename, err in sorted(list(errdata.items())):
            boxid += 1
            lines = []
            for x in err:
                if "#stdout" in x:
                    lines.append(x["#stdout"])
                if "#stderr" in x:
                    lines.append(H.span["err"](x["#stderr"]))
            if not lines:
                lines = [H.i("no output")]
            out._html(
                H.div(
                    H.input["toggle"](type="checkbox", id=f"box{boxid}"),
                    H.label["toggle"](filename, **{"for": f"box{boxid}"}),
                    H.div["collapsible"](lines),
                )
            )

    out.finalize()


_formatters = {
    "n": "{:.0f}".format,
    "fail": "{:.0f}".format,
    "std": "{:10.2f}".format,
    "iqr": "{:10.2f}".format,
    "perf": "{:10.2f}".format,
    "perf_base": "{:10.2f}".format,
    "perf_adj": "{:10.2f}".format,
    "perf_ratio": "{:10.2f}".format,
    "perf_base_adj": "{:10.2f}".format,
    "perf_ratio_adj": "{:10.2f}".format,
    "std%": "{:6.1%}".format,
    "sem%": "{:6.1%}".format,
    "iqr%": "{:6.1%}".format,
    "weight": "{:4.2f}".format,
    "peak_memory": "{:.0f}".format,
    0: "{:.0%}".format,
    1: "{:.0%}".format,
    2: "{:.0%}".format,
    3: "{:.0%}".format,
    4: "{:.0%}".format,
    5: "{:.0%}".format,
    6: "{:.0%}".format,
    7: "{:.0%}".format,
    8: "{:.0%}".format,
    9: "{:.0%}".format,
    10: "{:.0%}".format,
    11: "{:.0%}".format,
    12: "{:.0%}".format,
    13: "{:.0%}".format,
    14: "{:.0%}".format,
    15: "{:.0%}".format,
}


_table_style = H.style(
    """
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
"""
)


_error_style = H.style(
    """
input.toggle[type='checkbox'] {
    display: none;
}

label.toggle {
  display: block;
  font-weight: bold;
  cursor: pointer;
}

label.toggle:hover {
  background: #ddd;
}

.collapsible {
  max-height: 0px;
  overflow: hidden;
  transition: max-height .10s ease-in-out;
  white-space: pre;
}

input.toggle:checked + label.toggle + .collapsible {
  max-height: 500px;
  overflow: scroll;
}

.err {
    color: red;
}
"""
)


def _style(df):
    def _redgreen(value):
        return "color: green" if value else "color: red"

    def _redblack(value):
        return "color: red; font-weight: bold" if value else "color: black"

    def _greyblack(value):
        return "color: #888" if value else "color: black"

    def _gpu_pct(value):
        if value >= 0.9:
            color = "#080"
        elif value >= 0.8:
            color = "#880"
        elif value >= 0.7:
            color = "#F80"
        else:
            color = "#F00"
        return f"color: {color}"

    def _ratio(value):
        if value >= 1.1:
            color = "#080"
        elif value >= 0.9:
            color = "#880"
        elif value >= 0.75:
            color = "#F80"
        else:
            color = "#F00"
        return f"color: {color}"

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
    if "fail" in df.columns:
        sty = sty.apply(_fail, subset=["fail"])

    # Format weight column
    if "weight" in df.columns:
        sty = sty.apply(_weight, subset=["weight"])

    # Format performance ratios
    for col in ["perf_ratio", "perf_ratio_adj"]:
        if col in df.columns:
            sty = sty.applymap(_ratio, subset=[col])

    return sty
