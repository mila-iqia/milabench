import math
import sys
import io

import numpy as np
from hrepr import HTML, hrepr
from pandas import DataFrame

from milabench.utils import error_guard
from milabench.summary import Summary
from milabench.config import get_config_global
from milabench.system import option

nan = math.nan

H = HTML()


@error_guard({})
def _make_row(summary, compare, config, query=None):
    mkey = "train_rate"
    metric = "mean"

    weight = config.get("weight", summary.get("weight", 0))
    is_enabled = config.get("enabled", summary.get("enabled", 0))

    row = {
        "n": nan,
        "fail": nan,
        "ngpu": summary.get("ngpu", 0),
        "perf": nan,
        "std%": nan,
        "sem%": nan,
        "peak_memory": nan,
        "score": nan,
        "weight": weight,
        "enabled": is_enabled,
        "energy (Kj)": summary.get("energy", 0),
        "perf_watt": summary.get("perf_watt", 0)
    }

    if not summary or summary.get("empty", False):
        return row

    # Count not running an enabled benchmark as a failure
    failures = summary["failures"]
    if is_enabled and row["n"] <= 0:
        failures += 1

    row["n"] = summary["n"]
    row["fail"] = failures
    row["perf"] = summary[mkey][metric]

    if compare:
        row["perf_base"] = compare[mkey][metric]
        row["perf_ratio"] = row["perf_adj"] / row["perf_base"]

    row["std%"] = summary[mkey]["std"] / summary[mkey][metric]
    row["sem%"] = summary[mkey]["sem"] / summary[mkey][metric]
    # row["iqr%"] = (summary[mkey]["q3"] - summary[mkey]["q1"]) / summary[mkey]["median"] if summary else nan

    if summary["gpu_load"]:
        memory = [data["memory"]["max"] for data in summary["gpu_load"].values()]
        row["peak_memory"] = max(memory)

        power = [data["power"]["median"] for data in summary["gpu_load"].values()]
        row["median_watt"] = sum(power) / len(power)

    # Sum of all the GPU performance
    # to get the overall perf of the whole machine
    if "per_gpu" in summary:
        acc = 0
        for _, metrics in summary["per_gpu"].items():
            acc += metrics[metric]
    else:
        acc = row["perf"]

    success_ratio = 1 - row["fail"] / max(row["n"], 1)
    score = (acc if acc > 0 else row["perf"]) * success_ratio
    row["score"] = score

    if query is not None:
        extra = summary.get("extra", dict())
        for q in query:
            if v := extra.get(q):
                row[q] = v

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
        length = max(map(len, self.fields.keys())) + 2
        lines = []
        for k, v in self.fields.items():
            v = f"{v:10.2f}" if isinstance(v, float) else v
            lines.append(f"{k + ':':{length}} {v}")
        return "\n".join(lines)


class Outputter:
    def __init__(self, stdout, html):
        self.stdout = stdout
        if isinstance(html, io.StringIO):
            self.html_file = html
        else:
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
            self._text(pandas_to_string(x, formatters=_formatters))
        else:
            self._text(str(x))

    def print(self, x):
        self.text(x)
        self.html(x)

    def section(self, title):
        self.text("")
        self.text(title)
        self.text("-" * len(title))
        self.html(H.h2(title))

    def subsection(self, title):
        self.text("")
        self.text(title)
        self.text("^" * len(title))
        self.html(H.h3(title))

    def title(self, title):
        self._html(f"<html><head><title>{title}</title></head><body>")
        self.html(H.h1(title))
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
    n: i for i, n in enumerate([
        "fail",
        "n",
        "ngpu",
        "perf",
        "perf_adj",
        "sem%",
        "std%",
        "peak_memory",
        "score",
        "weight",
        "energy (Kj)",
        "perf_watt",
        "median_watt",
    ])
}


def make_dataframe(summary, compare=None, weights=None, query=None):
    if weights is not None:
        # We've overriden the config
        required = weights.keys()

        for key in required:
            if key not in summary:
                summary[key] = {
                    "name": key,
                    "n": 0,
                    "successes": 0,
                    "failures": 0,
                    "enabled": weights.get(key, {}).get("enabled", 1),
                    "empty": True,
                    "weight": weights.get(key, {}).get("weight", 0)
                }

    if weights is None:
        weights = dict()

    all_keys = list(
        {
            *(summary.keys() if summary else []),
            *(compare.keys() if compare else []),
            *(weights.keys() if weights else []),
        }
    )

    def sort_by(key):
        """Group similar runs together"""
        if summary:
            priority = summary.get(key, {}).get("priority", None)
            if priority:
                return priority

        if weights:
            return weights.get(key, {}).get("group", key)

        if summary:
            return summary.get(key, {}).get("group", key)

        return key

    # Sort by name first so bench with similar names are together
    #   we want bench in the same group with similar names to be close
    all_keys = sorted(all_keys)

    # Sort by group so bench are grouped together
    #   we want flops bench to be close together no matter what their names are
    all_keys = sorted(all_keys, key=sort_by)

    df = DataFrame(
        {
            key: _make_row(
                summary.get(key, {}),
                compare and compare.get(key, {}),
                weights and weights.get(key, {}),
                query=query,
            )
            for key in all_keys
        }
    ).transpose()

    if option("report.lean", int, 0) != 0:
        df = df.dropna(subset=["n"])

    # Reorder columns
    df = df[sorted(df.columns, key=lambda k: columns_order.get(k, 2000))]

    return df


def normalize_dataframe(df):
    columns = filter(lambda k: k in columns_order, df.columns)
    columns = sorted(columns, key=lambda k: columns_order.get(k, 0))

    for col in columns:
        df[col] = df[col].astype(float)

    return df[columns]


def get_meta(summary):
    try:
        for summary in summary.values():
            if meta := summary.get("meta"):
                return meta
    except:
        return {}


def print_meta(out, meta):
    out.section("System")

    for k, v in meta.items():

        if k == "accelerators":
            gpus = v["gpus"]
            n = len(gpus)
            gpu = {}
            if n > 0:
                _, gpu = gpus.popitem()
            stats = {
                "n": n,
                "product": gpu.get("product", "NA"),
                "memory": gpu.get("memory", {}).get("total", 0)
            }
            out.subsection(k)
            out.print(Table(stats))

        elif isinstance(v, dict):
            v.pop("build_settings", None)
            out.subsection(k)
            out.print(Table(v))


def short_meta(out, meta):
    stats = {}
    for k, v in meta.items():
        if k == "accelerators":
            gpus = v["gpus"]
            n = len(gpus)
            gpu = {}
            if n > 0:
                _, gpu = gpus.popitem()
            stats["product"] = gpu.get("product", "NA")
            stats["n_gpu"] = n
            stats["memory"] = str(gpu.get("memory", {}).get("total", 0))

        if k == "cpu":
            stats["cpu"] = v["brand"]
            stats["n_cpu"] = v ["count"]

    out.section("System")
    out.print(Table(stats))


def to_latex(df):
    from dataclasses import dataclass
    from .system import option

    default_columns = [
        "ngpu",
        "perf",
        "sem%",
        "std%"
    ]

    @dataclass
    class LatexTable:
        output: str = option("latex.output", str, None)
        columns: str = option("latex.columns", str, ",".join(default_columns))

    options = LatexTable()

    columns = options.columns.split(",")

    df = df[columns]

    if options.output is not None:
        with open(options.output, "w") as fp:
            txt = df.to_latex(formatters=_formatters, escape=False)
            txt = txt.replace("%", "\\%").replace("_", "\\_")
            fp.write(txt)



def get_weight_total(config):
    total = 0
    for _, bench in config.items():
        total += bench["weight"] * int(bench["enabled"] is True)
    return total


@error_guard({})
def make_report(
    summary: dict[str, Summary],
    compare=None,
    html=None,
    compare_gpus=False,
    price=None,
    title=None,
    sources=None,
    errdata=None,
    weights=None,
    stream=sys.stdout,
):
    # We want the score to be consistent with the loaded config
    # that means select/exclude and unsupported bench weight still counts
    if weights is None:
        weights = get_config_global()

    meta = get_meta(summary)

    df = make_dataframe(summary, compare, weights)
    out = Outputter(stdout=stream, html=html)

    if sources:
        if isinstance(sources, str):
            sources = [sources]
        for source in sources:
            out.print(f"Source: {source}")

    out.title(title or "Benchmark results")

    if meta:
        short_meta(out, meta)

    out.section("Breakdown")

    # Reorder columns
    normalized = normalize_dataframe(df)
    out.print(normalized)

    # to_latex(normalized)

    out.section("Scores")

    def _score(column):
        try:
            # This computes a weighted geometric mean

            # perf can be object np.float64 !?
            # success_ratio = 1 - row["fail"] / max(row["n"], 1)

            # score = (acc if acc > 0 else row["perf"]) * success_ratio
            score = df[column].astype(float)
            score = score.fillna(0)  # Replace nan by 0

            weights = df["weight"] * df["enabled"].astype(int)

            # if total weight is 0 ?
            weight_total = np.sum(weights)

            # score cannot be 0
            logscore = np.sum(np.log(score + 1) * weights) / weight_total

            return np.exp(logscore)
        except ZeroDivisionError:
            return -1

    score = _score("score")
    failure_rate = df["fail"].sum() / max(df["n"].sum(), 1)
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
        out.section("Errors")
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
    return normalized


_formatters = {
    "fail": "{:4.0f}".format,
    "n": "{:3.0f}".format,
    "ngpu": "{:4.0f}".format,
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
    "score": "{:10.2f}".format,
    "weight": "{:6.2f}".format,
    "peak_memory": "{:11.0f}".format,
    "elapsed": "{:5.0f}".format,
    "batch_size": "{:3.0f}".format,
    "energy (Kj)": "{:3.3f}".format,
    "perf_watt": "{:3.3f}".format,
    "median_watt": "{:3.3f}".format,
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


def pandas_to_string(df, formatters=_formatters):
    """Default stdout printer does not insert a column sep which makes it hard to retranscribe results elsewhere.
    to_csv does not align the output.
    """
    from collections import defaultdict

    columns = df.columns.tolist()

    # Compute column size
    col_size = defaultdict(int)
    for index, row in df.iterrows():
        col_size["bench"] = max(col_size["bench"], len(index), len("bench"))
        for col, val in zip(columns, row):
            fmt = formatters.get(col)
            if fmt is not None:
                val = fmt(val)
                col_size[col] = max(col_size[col], len(val), len(col))

    # Generate report
    sep = " | "
    lines = []
    for index, row in df.iterrows():
        size = col_size["bench"]
        line = [f"{index:<{size}}"]

        for col, val in zip(columns, row):
            fmt = formatters.get(col)
            if fmt is not None:
                val = fmt(val)
            else:
                val = str(val)

            size = col_size[col]
            val = f"{val:>{size}}"
            line.append(val)

        lines.append(sep.join(line))

    def fmtcol(col):
        size = col_size[col]
        return f"{col:>{size}}"

    size = col_size["bench"]
    header = sep.join([f"{'bench':<{size}}"] + [fmtcol(col) for col in columns])

    return "\n".join([header] + lines)


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
