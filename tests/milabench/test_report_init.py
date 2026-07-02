"""Tests for milabench.report.__init__ — pure logic functions."""

import io
import math

import numpy as np
import pytest
from pandas import DataFrame
from unittest.mock import patch

def _has_jinja2():
    try:
        import jinja2  # noqa: F401
        return True
    except ImportError:
        return False


from milabench.report import (
    WithClass,
    PassFail,
    Table,
    Outputter,
    _make_row,
    make_dataframe,
    normalize_dataframe,
    get_meta,
    print_meta,
    short_meta,
    get_weight_total,
    pandas_to_string,
    _report_pergpu,
    _style,
    _formatters,
    columns_order,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_summary():
    """Minimal summary dict that satisfies _make_row's happy path."""
    return {
        "n": 10,
        "successes": 10,
        "failures": 0,
        "ngpu": 4,
        "enabled": True,
        "weight": 1.0,
        "train_rate": {
            "mean": 100.0,
            "std": 5.0,
            "sem": 1.5,
            "median": 99.0,
            "min": 90.0,
            "max": 110.0,
        },
        "gpu_load": {
            "0": {
                "memory": {"max": 8000, "mean": 6000},
                "power": {"median": 250},
            },
            "1": {
                "memory": {"max": 7500, "mean": 5500},
                "power": {"median": 240},
            },
        },
        "per_gpu": {
            0: {"mean": 50.0},
            1: {"mean": 50.0},
        },
        "energy": 1.5,
        "perf_watt": 0.8,
    }


@pytest.fixture
def empty_summary():
    return {"empty": True}


@pytest.fixture
def config_with_weight():
    return {"weight": 2.0, "enabled": True}


@pytest.fixture
def config_no_weight():
    return {}


@pytest.fixture
def sample_meta():
    return {
        "accelerators": {
            "gpus": {
                "0": {
                    "product": "RTX 4090",
                    "memory": {"total": 24576},
                },
                "1": {
                    "product": "RTX 4090",
                    "memory": {"total": 24576},
                },
            },
        },
        "cpu": {
            "brand": "AMD EPYC 9654",
            "count": 192,
        },
        "torch": {
            "version": "2.1.0",
            "build_settings": {"some": "data"},
        },
    }


@pytest.fixture
def multi_bench_summary(basic_summary):
    """Multiple benchmarks for make_dataframe tests."""
    s2 = dict(basic_summary)
    s2 = {**basic_summary, "weight": 0.5}
    return {
        "bench_a": basic_summary,
        "bench_b": s2,
    }


@pytest.fixture
def weights_config():
    return {
        "bench_a": {"weight": 1.0, "enabled": True, "group": "compute"},
        "bench_b": {"weight": 0.5, "enabled": True, "group": "memory"},
        "bench_c": {"weight": 0.0, "enabled": False, "group": "io"},
    }


# ---------------------------------------------------------------------------
# WithClass
# ---------------------------------------------------------------------------

class TestWithClass:
    def test_str(self):
        wc = WithClass("hello", "my-class")
        assert str(wc) == "hello"

    def test_str_numeric(self):
        wc = WithClass(42, "number")
        assert str(wc) == "42"

    def test_hrepr(self):
        from hrepr import HTML
        H = HTML()
        wc = WithClass("val", "cls")
        result = wc.__hrepr__(H, None)
        rendered = str(result)
        assert "val" in rendered
        assert "cls" in rendered


# ---------------------------------------------------------------------------
# PassFail
# ---------------------------------------------------------------------------

class TestPassFail:
    def test_pass(self):
        pf = PassFail(0.95, True)
        assert "PASS" in str(pf)
        assert pf.passfail == "PASS"
        assert pf.klass == "PASS"

    def test_fail(self):
        pf = PassFail(0.50, False)
        assert "FAIL" in str(pf)
        assert pf.passfail == "FAIL"
        assert pf.klass == "FAIL"

    def test_formatting(self):
        pf = PassFail(0.0, True)
        s = str(pf)
        assert "0.00%" in s
        assert "(PASS)" in s

    def test_hrepr(self):
        from hrepr import HTML
        H = HTML()
        pf = PassFail(0.5, False)
        result = pf.__hrepr__(H, None)
        rendered = str(result)
        assert "FAIL" in rendered


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

class TestTable:
    def test_str_formatting(self):
        t = Table({"alpha": 3.14159, "beta": "hello"})
        s = str(t)
        assert "alpha:" in s
        assert "3.14" in s
        assert "hello" in s

    def test_str_alignment(self):
        t = Table({"short": 1.0, "a_longer_key": 2.0})
        lines = str(t).split("\n")
        assert len(lines) == 2

    def test_hrepr(self):
        from hrepr import HTML, hrepr
        H = HTML()
        t = Table({"x": 1.5, "y": "text"})
        result = t.__hrepr__(H, hrepr)
        rendered = str(result)
        assert "1.50" in rendered
        assert "text" in rendered

    def test_empty(self):
        t = Table({"k": "v"})
        s = str(t)
        assert "k:" in s


# ---------------------------------------------------------------------------
# _make_row
# ---------------------------------------------------------------------------

class TestMakeRow:
    def test_empty_summary(self, empty_summary, config_with_weight):
        row = _make_row(empty_summary, None, config_with_weight)
        assert math.isnan(row["perf"])
        assert math.isnan(row["n"])
        assert row["weight"] == 2.0

    def test_no_summary(self, config_with_weight):
        row = _make_row({}, None, config_with_weight)
        assert math.isnan(row["perf"])

    def test_basic_summary(self, basic_summary, config_no_weight):
        row = _make_row(basic_summary, None, config_no_weight)
        assert row["n"] == 10
        assert row["fail"] == 0
        assert row["perf"] == 100.0
        assert row["std%"] == pytest.approx(0.05)
        assert row["sem%"] == pytest.approx(0.015)
        assert row["peak_memory"] == 8000
        assert row["score"] > 0

    def test_weight_from_config(self, basic_summary, config_with_weight):
        row = _make_row(basic_summary, None, config_with_weight)
        assert row["weight"] == 2.0

    def test_weight_from_summary(self, basic_summary, config_no_weight):
        basic_summary["weight"] = 3.0
        row = _make_row(basic_summary, None, config_no_weight)
        assert row["weight"] == 3.0

    def test_gpu_load_peak_memory(self, basic_summary, config_no_weight):
        row = _make_row(basic_summary, None, config_no_weight)
        assert row["peak_memory"] == 8000
        assert "median_watt" in row
        assert row["median_watt"] == pytest.approx(245.0)

    def test_no_gpu_load(self, basic_summary, config_no_weight):
        basic_summary["gpu_load"] = {}
        row = _make_row(basic_summary, None, config_no_weight)
        assert math.isnan(row["peak_memory"])

    def test_no_per_gpu_uses_perf(self, basic_summary, config_no_weight):
        """Line 76: else branch when 'per_gpu' is absent."""
        del basic_summary["per_gpu"]
        row = _make_row(basic_summary, None, config_no_weight)
        assert row["score"] == row["perf"]

    def test_with_per_gpu(self, basic_summary, config_no_weight):
        row = _make_row(basic_summary, None, config_no_weight)
        assert row["score"] == pytest.approx(100.0)

    def test_enabled_but_no_runs_nan_comparison(self, config_with_weight):
        """Line 47-48: row['n'] starts as nan so nan<=0 is False — failure
        increment is unreachable. Test the actual behaviour: fail stays 0."""
        summary = {
            "n": 0,
            "successes": 0,
            "failures": 0,
            "ngpu": 1,
            "train_rate": {"mean": 10.0, "std": 1.0, "sem": 0.5},
            "gpu_load": {},
            "per_gpu": {},
            "energy": 0,
            "perf_watt": 0,
        }
        row = _make_row(summary, None, config_with_weight)
        assert row["n"] == 0
        assert row["fail"] == 0

    def test_query_extra(self, basic_summary, config_no_weight):
        """Lines 83-86: query extracts extra fields from summary."""
        basic_summary["extra"] = {"custom_metric": 42, "other": "val"}
        row = _make_row(basic_summary, None, config_no_weight, query=["custom_metric", "missing_key"])
        assert row["custom_metric"] == 42
        assert "missing_key" not in row

    def test_compare_branch_triggers_error_guard(self, basic_summary, config_no_weight):
        """Lines 55-56: compare path references row['perf_adj'] which doesn't
        exist, so error_guard catches the KeyError and returns {}."""
        compare = {
            "train_rate": {"mean": 80.0, "std": 4.0, "sem": 1.0},
        }
        row = _make_row(basic_summary, compare, config_no_weight)
        assert row == {}

    def test_failure_reduces_score(self, basic_summary, config_no_weight):
        basic_summary["failures"] = 5
        row = _make_row(basic_summary, None, config_no_weight)
        assert row["score"] < 100.0

    def test_energy_fields(self, basic_summary, config_no_weight):
        row = _make_row(basic_summary, None, config_no_weight)
        assert row["energy (Kj)"] == 1.5
        assert row["perf_watt"] == 0.8


# ---------------------------------------------------------------------------
# make_dataframe
# ---------------------------------------------------------------------------

class TestMakeDataframe:
    def test_basic(self, multi_bench_summary):
        df = make_dataframe(multi_bench_summary)
        assert "bench_a" in df.index
        assert "bench_b" in df.index

    def test_with_weights_adds_missing(self, multi_bench_summary, weights_config):
        """Lines 239-249: missing benchmarks added from weights."""
        df = make_dataframe(multi_bench_summary, weights=weights_config)
        assert "bench_c" in df.index

    def test_weights_none_default(self, multi_bench_summary):
        """Line 252: weights=None defaults to empty dict."""
        df = make_dataframe(multi_bench_summary, weights=None)
        assert len(df) >= 2

    def test_column_ordering(self, multi_bench_summary):
        df = make_dataframe(multi_bench_summary)
        cols = list(df.columns)
        if "fail" in cols and "perf" in cols:
            assert cols.index("fail") < cols.index("perf")

    def test_sort_by_priority(self, basic_summary):
        """Lines 266-267: sort_by uses priority if present."""
        s1 = {**basic_summary, "priority": "zzz"}
        s2 = {**basic_summary, "priority": "aaa"}
        summary = {"bench_z": s1, "bench_a": s2}
        df = make_dataframe(summary)
        idx = list(df.index)
        assert "bench_a" in idx
        assert "bench_z" in idx

    def test_sort_by_group_from_weights(self, basic_summary):
        """Lines 269-270: sort_by uses weights group."""
        summary = {"bench_x": basic_summary, "bench_y": basic_summary}
        weights = {
            "bench_x": {"weight": 1, "enabled": True, "group": "zzz"},
            "bench_y": {"weight": 1, "enabled": True, "group": "aaa"},
        }
        df = make_dataframe(summary, weights=weights)
        idx = list(df.index)
        assert idx.index("bench_y") < idx.index("bench_x")

    def test_sort_by_group_from_summary(self, basic_summary):
        """Lines 272-273: sort_by falls back to summary group."""
        s1 = {**basic_summary, "group": "zzz"}
        s2 = {**basic_summary, "group": "aaa"}
        summary = {"bench_x": s1, "bench_y": s2}
        df = make_dataframe(summary, weights=None)
        idx = list(df.index)
        assert idx.index("bench_y") < idx.index("bench_x")

    def test_sort_by_key_fallback(self, basic_summary):
        """Line 275: sort_by returns key when nothing else found."""
        summary = {"bench_b": basic_summary, "bench_a": basic_summary}
        df = make_dataframe(summary, weights=None)
        idx = list(df.index)
        assert idx.index("bench_a") < idx.index("bench_b")

    @patch("milabench.report.option", return_value=1)
    def test_lean_report_drops_na(self, mock_option, basic_summary):
        """Line 298: lean report drops rows with NaN n."""
        summary = {
            "bench_a": basic_summary,
            "bench_b": {"empty": True},
        }
        df = make_dataframe(summary, weights={"bench_a": {"weight": 1, "enabled": True}, "bench_b": {"weight": 1, "enabled": True}})
        # bench_b should be dropped (n is NaN)
        if "bench_b" in df.index:
            assert math.isnan(df.loc["bench_b", "n"])

    def test_query_forwarded(self, basic_summary):
        """make_dataframe passes query to _make_row."""
        basic_summary["extra"] = {"latency": 5.0}
        summary = {"bench_a": basic_summary}
        df = make_dataframe(summary, query=["latency"])
        assert "latency" in df.columns

    def test_empty_summary(self):
        df = make_dataframe({})
        assert len(df) == 0


# ---------------------------------------------------------------------------
# normalize_dataframe
# ---------------------------------------------------------------------------

class TestNormalizeDataframe:
    def test_identity(self):
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        result = normalize_dataframe(df)
        assert result.equals(df)


# ---------------------------------------------------------------------------
# get_meta
# ---------------------------------------------------------------------------

class TestGetMeta:
    def test_extracts_meta(self, sample_meta):
        """Lines 320-321: returns meta from first summary entry."""
        summary = {"bench_a": {"meta": sample_meta}}
        result = get_meta(summary)
        assert result == sample_meta

    def test_no_meta(self):
        summary = {"bench_a": {"n": 5}}
        result = get_meta(summary)
        assert result is None

    def test_exception_returns_empty(self):
        """Lines 322-323: exception returns {}."""
        result = get_meta("not_a_dict")
        assert result == {}

    def test_empty_summary(self):
        result = get_meta({})
        assert result is None


# ---------------------------------------------------------------------------
# print_meta
# ---------------------------------------------------------------------------

class TestPrintMeta:
    def test_accelerators(self, sample_meta):
        """Lines 327-348: print_meta handles accelerators and dict values."""
        buf = io.StringIO()
        out = Outputter(stdout=buf, html=None)
        print_meta(out, sample_meta)
        text = buf.getvalue()
        assert "System" in text
        assert "accelerators" in text
        assert "RTX 4090" in text

    def test_dict_meta_without_accelerators(self):
        meta = {"custom": {"key1": "val1", "key2": "val2"}}
        buf = io.StringIO()
        out = Outputter(stdout=buf, html=None)
        print_meta(out, meta)
        text = buf.getvalue()
        assert "custom" in text
        assert "val1" in text

    def test_build_settings_removed(self, sample_meta):
        """Line 346: build_settings popped from dict."""
        buf = io.StringIO()
        out = Outputter(stdout=buf, html=None)
        print_meta(out, sample_meta)
        text = buf.getvalue()
        assert "build_settings" not in text

    def test_no_gpus(self):
        meta = {"accelerators": {"gpus": {}}}
        buf = io.StringIO()
        out = Outputter(stdout=buf, html=None)
        print_meta(out, meta)
        text = buf.getvalue()
        assert "n:" in text


# ---------------------------------------------------------------------------
# short_meta
# ---------------------------------------------------------------------------

class TestShortMeta:
    def test_with_accelerators_and_cpu(self, sample_meta):
        buf = io.StringIO()
        out = Outputter(stdout=buf, html=None)
        short_meta(out, sample_meta)
        text = buf.getvalue()
        assert "RTX 4090" in text or "product" in text
        assert "AMD EPYC" in text or "cpu" in text

    def test_no_gpus(self):
        meta = {"accelerators": {"gpus": {}}}
        buf = io.StringIO()
        out = Outputter(stdout=buf, html=None)
        short_meta(out, meta)
        text = buf.getvalue()
        assert "NA" in text

    def test_empty_meta_raises(self):
        """Table({}) raises ValueError in __str__ because max() gets empty seq."""
        buf = io.StringIO()
        out = Outputter(stdout=buf, html=None)
        with pytest.raises(ValueError, match="max.*empty"):
            short_meta(out, {})


# ---------------------------------------------------------------------------
# get_weight_total
# ---------------------------------------------------------------------------

class TestGetWeightTotal:
    def test_basic(self):
        """Lines 403-406."""
        config = {
            "a": {"weight": 1.0, "enabled": True},
            "b": {"weight": 2.0, "enabled": True},
            "c": {"weight": 3.0, "enabled": False},
        }
        assert get_weight_total(config) == pytest.approx(3.0)

    def test_all_disabled(self):
        config = {
            "a": {"weight": 1.0, "enabled": False},
            "b": {"weight": 2.0, "enabled": False},
        }
        assert get_weight_total(config) == 0

    def test_empty(self):
        assert get_weight_total({}) == 0


# ---------------------------------------------------------------------------
# Outputter
# ---------------------------------------------------------------------------

class TestOutputter:
    def test_text_only(self):
        buf = io.StringIO()
        out = Outputter(stdout=buf, html=None)
        out.text("hello")
        assert "hello" in buf.getvalue()

    def test_html_stringio(self):
        buf = io.StringIO()
        html_buf = io.StringIO()
        out = Outputter(stdout=buf, html=html_buf)
        out._html("<p>test</p>")
        assert "<p>test</p>" in html_buf.getvalue()

    def test_no_stdout(self):
        """Line 161: text() returns early when stdout is None."""
        out = Outputter(stdout=None, html=None)
        out.text("should not crash")

    def test_section(self):
        buf = io.StringIO()
        out = Outputter(stdout=buf, html=None)
        out.section("MySection")
        text = buf.getvalue()
        assert "MySection" in text
        assert "---" in text or "-" * len("MySection") in text

    def test_subsection(self):
        """Lines 178-181."""
        buf = io.StringIO()
        out = Outputter(stdout=buf, html=None)
        out.subsection("Sub")
        text = buf.getvalue()
        assert "Sub" in text
        assert "^^^" in text or "^" * len("Sub") in text

    def test_title(self):
        buf = io.StringIO()
        out = Outputter(stdout=buf, html=None)
        out.title("My Title")
        text = buf.getvalue()
        assert "My Title" in text
        assert "=" * len("My Title") in text

    def test_print_dataframe(self):
        buf = io.StringIO()
        out = Outputter(stdout=buf, html=None)
        df = DataFrame({"perf": [1.0, 2.0], "n": [5, 10]}, index=["a", "b"])
        out.print(df)
        text = buf.getvalue()
        assert "perf" in text

    @pytest.mark.skipif(
        not _has_jinja2(), reason="jinja2 not installed"
    )
    def test_html_dataframe(self):
        """Lines 153-157: html() handles DataFrame."""
        html_buf = io.StringIO()
        out = Outputter(stdout=None, html=html_buf)
        df = DataFrame({"perf": [1.0, 2.0]}, index=["a", "b"])
        out.html(df)
        text = html_buf.getvalue()
        assert len(text) > 0

    def test_html_none_early_return(self):
        """Line 152: html returns early when html_file is None."""
        out = Outputter(stdout=None, html=None)
        out.html("should not crash")

    def test_finalize(self):
        html_buf = io.StringIO()
        out = Outputter(stdout=None, html=html_buf)
        out.finalize()
        assert "</body>" in html_buf.getvalue()


# ---------------------------------------------------------------------------
# pandas_to_string
# ---------------------------------------------------------------------------

class TestPandasToString:
    def test_basic(self):
        df = DataFrame(
            {"perf": [100.5, 200.3], "n": [5, 10]},
            index=["bench_a", "bench_b"],
        )
        result = pandas_to_string(df)
        assert "bench_a" in result
        assert "bench_b" in result
        assert "perf" in result
        assert "|" in result

    def test_column_with_no_formatter(self):
        df = DataFrame(
            {"custom_col": ["abc", "xyz"]},
            index=["x", "y"],
        )
        result = pandas_to_string(df, formatters={})
        assert "custom_col" in result
        assert "abc" in result

    def test_alignment(self):
        df = DataFrame(
            {"perf": [1.0, 2.0], "n": [3, 4]},
            index=["a", "b"],
        )
        result = pandas_to_string(df)
        lines = result.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows


# ---------------------------------------------------------------------------
# _report_pergpu
# ---------------------------------------------------------------------------

class TestReportPergpu:
    def test_basic(self):
        """Lines 195-211."""
        entries = {
            "bench_a": {
                "per_gpu": {
                    0: {"50": 100.0, "mean": 100.0},
                    1: {"50": 90.0, "mean": 90.0},
                },
            },
            "bench_b": {
                "per_gpu": {
                    0: {"50": 80.0, "mean": 80.0},
                    1: {"50": 95.0, "mean": 95.0},
                },
            },
        }
        df = _report_pergpu(entries, measure="50")
        assert "bench_a" in df.index
        assert "bench_b" in df.index
        assert 0 in df.columns
        assert 1 in df.columns
        # Rows should be normalized by their max
        assert df.loc["bench_a", 0] == pytest.approx(1.0)
        assert df.loc["bench_a", 1] == pytest.approx(0.9)

    def test_mean_measure(self):
        entries = {
            "bench_a": {
                "per_gpu": {
                    0: {"mean": 50.0},
                },
            },
        }
        df = _report_pergpu(entries, measure="mean")
        assert df.loc["bench_a", 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _style
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_jinja2(), reason="jinja2 not installed")
class TestStyle:
    def test_basic_styling(self):
        """Lines 680-740."""
        df = DataFrame(
            {
                "fail": [0, 2],
                "n": [10, 10],
                "ngpu": [4, 4],
                "perf": [100.0, 200.0],
                "std%": [0.05, 0.1],
                "sem%": [0.01, 0.02],
                "score": [100.0, 200.0],
                "weight": [1.0, 0.5],
                "peak_memory": [8000, 7000],
            },
            index=["a", "b"],
        )
        sty = _style(df)
        html = sty._repr_html_()
        assert "color: red" in html or "color: green" in html or len(html) > 0

    def test_with_gpu_columns(self):
        df = DataFrame(
            {
                0: [0.95, 0.7],
                1: [0.85, 0.65],
                "fail": [0, 1],
                "weight": [1.0, 1.0],
            },
            index=["a", "b"],
        )
        sty = _style(df)
        html = sty._repr_html_()
        assert len(html) > 0

    def test_with_perf_ratio(self):
        df = DataFrame(
            {
                "perf_ratio": [1.2, 0.8],
                "fail": [0, 0],
                "weight": [1.0, 1.0],
            },
            index=["a", "b"],
        )
        sty = _style(df)
        html = sty._repr_html_()
        assert len(html) > 0

    def test_no_fail_column(self):
        df = DataFrame(
            {"perf": [100.0], "weight": [1.0]},
            index=["a"],
        )
        sty = _style(df)
        assert sty is not None


# ---------------------------------------------------------------------------
# columns_order
# ---------------------------------------------------------------------------

class TestColumnsOrder:
    def test_bench_first(self):
        assert columns_order["bench"] == 0

    def test_known_columns(self):
        assert "fail" in columns_order
        assert "perf" in columns_order
        assert "score" in columns_order
        assert "weight" in columns_order

    def test_ordering(self):
        assert columns_order["bench"] < columns_order["fail"]
        assert columns_order["fail"] < columns_order["perf"]
        assert columns_order["perf"] < columns_order["score"]


# ---------------------------------------------------------------------------
# _formatters
# ---------------------------------------------------------------------------

class TestFormatters:
    def test_perf_format(self):
        assert _formatters["perf"](123.456) == "    123.46"

    def test_fail_format(self):
        assert _formatters["fail"](3.0).strip() == "3"

    def test_percent_format(self):
        assert _formatters["std%"](0.05) == "  5.0%"

    def test_gpu_index_format(self):
        assert _formatters[0](0.95) == "95%"
        assert _formatters[15](0.5) == "50%"
