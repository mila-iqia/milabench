"""Tests for milabench.dashboard.live module.

Covers: MetricAcc, helper functions, ReportMachinePerf (rendering, data
update, edge cases, summary, backward compat, dump_metric_table),
ReportGPUPerf, and LivePrinter.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from milabench.dashboard.live import (
    MetricAcc,
    ReportGPUPerf,
    ReportMachinePerf,
    LivePrinter,
    drop_min_max,
    get_benchname,
    get_per_gpu_key,
    _get,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakePack:
    config: dict
    tag: str = "run.0"


def _make_entry(event, data, config=None, tag="run.0"):
    """Build a lightweight object that mimics BenchLogEntry."""
    pack = _FakePack(config=config or {"name": "bench1", "job-number": 0, "device": 0}, tag=tag)
    entry = MagicMock()
    entry.event = event
    entry.data = data
    entry.pack = pack
    entry.tag = tag
    return entry


def _build_report(**kwargs):
    """Instantiate ReportMachinePerf with print suppressed."""
    with patch("builtins.print"):
        return ReportMachinePerf(**kwargs)


# ---------------------------------------------------------------------------
# Tests for standalone helpers
# ---------------------------------------------------------------------------

class TestGetBenchname:
    def test_returns_name(self):
        assert get_benchname({"name": "foo"}) == "foo"

    def test_missing_name(self):
        assert get_benchname({}) is None


class TestGetPerGpuKey:
    def test_defaults(self):
        assert get_per_gpu_key({}) == "NX-DY"

    def test_with_values(self):
        assert get_per_gpu_key({"job-number": 3, "device": 1}) == "N3-D1"


class TestDropMinMax:
    def test_fewer_than_five(self):
        assert drop_min_max([3, 1, 2]) == [1, 2, 3]

    def test_five_or_more(self):
        assert drop_min_max([5, 1, 3, 2, 4]) == [2, 3, 4]

    def test_none_values_filtered(self):
        assert drop_min_max([None, 3, 1, None, 2]) == [1, 2, 3]

    def test_empty(self):
        assert drop_min_max([]) == []


class TestGet:
    def test_nested(self):
        d = {"a": {"b": 42}}
        assert _get(d, "a", "b") == 42

    def test_missing_returns_default(self):
        assert np.isnan(_get({}, "x", "y"))

    def test_custom_default(self):
        assert _get({}, "x", "y", default=-1) == -1

    def test_partial_path(self):
        assert np.isnan(_get({"a": {}}, "a", "missing"))


# ---------------------------------------------------------------------------
# MetricAcc dataclass
# ---------------------------------------------------------------------------

class TestMetricAcc:
    def test_defaults(self):
        acc = MetricAcc()
        assert acc.name is None
        assert acc.started == 0
        assert acc.finished == 0
        assert acc.shown is False
        assert acc.successes == 0
        assert acc.failures == 0
        assert acc.early_stop is False

    def test_metrics_nested_defaultdict(self):
        acc = MetricAcc()
        acc.metrics["rate"]["gpu0"].append(100.0)
        assert acc.metrics["rate"]["gpu0"] == [100.0]


# ---------------------------------------------------------------------------
# ReportMachinePerf – construction and header
# ---------------------------------------------------------------------------

class TestReportMachinePerfInit:
    def test_show_header_format(self):
        r = _build_report()
        header = r.show_header()
        assert "name" in header
        assert "perf" in header
        assert "peak_memory" in header

    def test_context_manager_exit(self):
        """__exit__ is a no-op and should not raise."""
        r = _build_report()
        r.__exit__()

    def test_show_pending_noop(self):
        """show_pending (line 93-94) is a no-op."""
        r = _build_report()
        assert r.show_pending() is None


# ---------------------------------------------------------------------------
# on_config – line 111
# ---------------------------------------------------------------------------

class TestOnConfig:
    def test_stores_config(self):
        r = _build_report()
        entry = _make_entry("config", {"name": "bench_x", "extra": 1})
        r.on_config(entry)
        assert r.config == {"name": "bench_x", "extra": 1}


# ---------------------------------------------------------------------------
# benchname – line 103
# ---------------------------------------------------------------------------

class TestBenchname:
    def test_returns_config_name(self):
        r = _build_report()
        r.config = {"name": "mybench"}
        assert r.benchname() == "mybench"


# ---------------------------------------------------------------------------
# add_metric / reduce / bench_line
# ---------------------------------------------------------------------------

class TestAddMetricAndReduce:
    @pytest.fixture()
    def report(self):
        return _build_report()

    def test_add_metric(self, report):
        report.add_metric("bench1", "gpu0", "rate", 100.0)
        acc = report.accumulator["bench1"]
        assert acc.metrics["rate"]["gpu0"] == [100.0]

    def test_reduce_single_group(self, report):
        for v in [10, 20, 30]:
            report.add_metric("b", "g0", "rate", float(v))
        acc = report.accumulator["b"]
        reduced = report.reduce(acc)
        assert "rate" in reduced
        assert "mean" in reduced["rate"]

    def test_reduce_multiple_groups_sum(self, report):
        """Non-temperature/memory metrics are summed across groups."""
        report.add_metric("b", "g0", "rate", 100.0)
        report.add_metric("b", "g1", "rate", 200.0)
        acc = report.accumulator["b"]
        reduced = report.reduce(acc)
        assert reduced["rate"]["mean"] == pytest.approx(300.0)

    def test_reduce_temperature_averaged(self, report):
        report.add_metric("b", "g0", "temperature", 60.0)
        report.add_metric("b", "g1", "temperature", 80.0)
        acc = report.accumulator["b"]
        reduced = report.reduce(acc)
        assert reduced["temperature"]["mean"] == pytest.approx(70.0)

    def test_reduce_memory_averaged(self, report):
        report.add_metric("b", "g0", "memory", 4.0)
        report.add_metric("b", "g1", "memory", 8.0)
        acc = report.accumulator["b"]
        reduced = report.reduce(acc)
        assert reduced["memory"]["mean"] == pytest.approx(6.0)

    def test_reduce_loss_averaged(self, report):
        report.add_metric("b", "g0", "loss", 0.5)
        report.add_metric("b", "g1", "loss", 1.5)
        acc = report.accumulator["b"]
        reduced = report.reduce(acc)
        assert reduced["loss"]["mean"] == pytest.approx(1.0)

    def test_reduce_walltime_averaged(self, report):
        report.add_metric("b", "g0", "walltime", 10.0)
        report.add_metric("b", "g1", "walltime", 20.0)
        acc = report.accumulator["b"]
        reduced = report.reduce(acc)
        assert reduced["walltime"]["mean"] == pytest.approx(15.0)

    def test_reduce_std_quadrature(self, report):
        """std/sem stats are combined in quadrature across groups."""
        report.add_metric("b", "g0", "rate", 100.0)
        report.add_metric("b", "g1", "rate", 100.0)
        acc = report.accumulator["b"]
        reduced = report.reduce(acc)
        assert reduced["rate"]["std"] == pytest.approx(0.0, abs=1e-10)

    def test_bench_line_format(self, report):
        report.add_metric("b", "g0", "rate", 100.0)
        acc = report.accumulator["b"]
        acc.name = "b"
        acc.started = 1
        acc.failures = 0
        line = report.bench_line(acc)
        assert "b" in line
        assert "|" in line


# ---------------------------------------------------------------------------
# on_start / on_end / on_stop
# ---------------------------------------------------------------------------

class TestStartEndStop:
    @pytest.fixture()
    def report(self):
        return _build_report()

    def test_on_start_increments(self, report):
        config = {"name": "b1", "job-number": 0, "device": 0}
        entry = _make_entry("start", {"time": 100.0}, config=config, tag="run.0")
        acc = report.accumulator["b1"]
        report.current_acc = acc
        report.on_start(entry)
        assert acc.started == 1
        assert acc.times["run.0"] == 100.0

    def test_on_end_success(self, report):
        config = {"name": "b1", "job-number": 0, "device": 0}
        acc = report.accumulator["b1"]
        acc.started = 1
        acc.times["run.0"] = 100.0

        entry = _make_entry("end", {"time": 200.0, "return_code": 0}, config=config, tag="run.0")
        with patch("builtins.print"):
            report.on_end(entry)
        assert acc.finished == 1
        assert acc.successes == 1
        assert acc.failures == 0

    def test_on_end_failure(self, report):
        config = {"name": "b1", "job-number": 0, "device": 0}
        acc = report.accumulator["b1"]
        acc.started = 1
        acc.times["run.0"] = 100.0

        entry = _make_entry("end", {"time": 200.0, "return_code": 1}, config=config, tag="run.0")
        with patch("builtins.print"):
            report.on_end(entry)
        assert acc.failures == 1
        assert acc.successes == 0

    def test_on_end_early_stop_counts_success(self, report):
        config = {"name": "b1", "job-number": 0, "device": 0}
        acc = report.accumulator["b1"]
        acc.started = 1
        acc.early_stop = True
        acc.times["run.0"] = 100.0

        entry = _make_entry("end", {"time": 200.0, "return_code": 1}, config=config, tag="run.0")
        with patch("builtins.print"):
            report.on_end(entry)
        assert acc.successes == 1

    def test_bench_finished_removes_from_accumulator(self, report):
        config = {"name": "b1", "job-number": 0, "device": 0}
        acc = report.accumulator["b1"]
        acc.name = "b1"
        acc.started = 1
        acc.times["run.0"] = 10.0
        report.add_metric("b1", "g0", "rate", 100.0)

        entry = _make_entry("end", {"time": 20.0, "return_code": 0}, config=config, tag="run.0")
        with patch("builtins.print"):
            report.on_end(entry)
        assert "b1" not in report.accumulator

    def test_on_stop_sets_early_stop(self, report):
        config = {"name": "b1", "job-number": 0, "device": 0}
        entry = _make_entry("stop", {}, config=config)
        report.on_stop(entry)
        assert report.accumulator["b1"].early_stop is True


# ---------------------------------------------------------------------------
# on_data – gpudata and plain metrics
# ---------------------------------------------------------------------------

class TestOnData:
    @pytest.fixture()
    def report(self):
        r = _build_report()
        r.current_bench = "b1"
        r.current_group = "g0"
        return r

    def test_plain_metric(self, report):
        entry = _make_entry("data", {"rate": 42.0})
        report.on_data(entry)
        acc = report.accumulator["b1"]
        assert 42.0 in acc.metrics["rate"]["g0"]

    def test_ignored_metrics_skipped(self, report):
        entry = _make_entry("data", {"task": "encode", "progress": 50, "units": "it/s"})
        report.on_data(entry)
        acc = report.accumulator["b1"]
        assert "task" not in acc.metrics
        assert "progress" not in acc.metrics
        assert "units" not in acc.metrics

    def test_gpudata_memory_uses_first_element(self, report):
        gpu_payload = {
            "gpu0": {"memory": [0.75], "load": 0.9, "temperature": 65},
        }
        entry = _make_entry("data", {"gpudata": gpu_payload})
        report.on_data(entry)
        acc = report.accumulator["b1"]
        assert 0.75 in acc.metrics["memory"]["g0"]
        assert 0.9 in acc.metrics["load"]["g0"]
        assert 65 in acc.metrics["temperature"]["g0"]

    def test_gpudata_non_memory_stored_directly(self, report):
        gpu_payload = {
            "gpu0": {"load": 0.5},
        }
        entry = _make_entry("data", {"gpudata": gpu_payload})
        report.on_data(entry)
        acc = report.accumulator["b1"]
        assert 0.5 in acc.metrics["load"]["g0"]


# ---------------------------------------------------------------------------
# summary / _backward_compat – lines 183-200
# ---------------------------------------------------------------------------

class TestSummaryAndBackwardCompat:
    @pytest.fixture()
    def report_with_data(self):
        r = _build_report()
        r.add_metric("b1", "g0", "rate", 100.0)
        r.add_metric("b1", "g0", "rate", 110.0)
        acc = r.accumulator["b1"]
        acc.successes = 2
        acc.failures = 1
        return r

    def test_summary_keys(self, report_with_data):
        s = report_with_data.summary()
        assert "b1" in s
        assert s["b1"]["successes"] == 2
        assert s["b1"]["failures"] == 1
        assert s["b1"]["n"] == 3

    def test_backward_compat_renames_rate(self, report_with_data):
        s = report_with_data.summary()
        assert s["b1"]["name"] == "b1"
        assert "train_rate" in s["b1"]
        assert "rate" not in s["b1"]

    def test_backward_compat_no_rate_key(self):
        r = _build_report()
        r.add_metric("b2", "g0", "memory", 4.0)
        acc = r.accumulator["b2"]
        acc.successes = 1
        acc.failures = 0
        s = r.summary()
        assert s["b2"]["train_rate"] == {}

    def test_summary_empty(self):
        r = _build_report()
        s = r.summary()
        assert s == {}

    def test_summary_multiple_benchmarks(self):
        r = _build_report()
        r.add_metric("a", "g0", "rate", 10.0)
        r.accumulator["a"].successes = 1
        r.add_metric("b", "g0", "rate", 20.0)
        r.accumulator["b"].successes = 1
        s = r.summary()
        assert "a" in s and "b" in s


# ---------------------------------------------------------------------------
# dump_metric_table – lines 222-248
# ---------------------------------------------------------------------------

class TestDumpMetricTable:
    @pytest.fixture()
    def report_with_metrics(self):
        r = _build_report()
        r.add_metric("b1", "g0", "rate", 100.0)
        r.add_metric("b1", "g0", "rate", 200.0)
        r.add_metric("b1", "g0", "memory", 4.0)
        acc = r.accumulator["b1"]
        acc.name = "b1"
        return r, acc

    def test_prints_table(self, report_with_metrics):
        r, acc = report_with_metrics
        with patch("builtins.print") as mock_print:
            r.dump_metric_table(acc)
        assert mock_print.call_count >= 1

    def test_shown_flag_prevents_repeat(self, report_with_metrics):
        r, acc = report_with_metrics
        with patch("builtins.print") as mock_print:
            r.dump_metric_table(acc)
            first_count = mock_print.call_count
            r.dump_metric_table(acc)
            assert mock_print.call_count == first_count

    def test_header_shown_once(self, report_with_metrics):
        r, acc = report_with_metrics
        with patch("builtins.print") as mock_print:
            r.dump_metric_table(acc, show_header=True)
        header_calls = [
            c for c in mock_print.call_args_list if "name" in str(c)
        ]
        assert len(header_calls) == 1

    def test_show_header_false(self):
        r = _build_report()
        r.add_metric("b1", "g0", "rate", 50.0)
        acc = r.accumulator["b1"]
        acc.name = "b1"
        with patch("builtins.print") as mock_print:
            r.dump_metric_table(acc, show_header=False)
        for call in mock_print.call_args_list:
            if "name" in str(call) and "rate" not in str(call):
                pytest.fail("Header should not have been printed")

    def test_header_not_shown_twice(self):
        r = _build_report()
        r.add_metric("b1", "g0", "rate", 50.0)
        acc1 = r.accumulator["b1"]
        acc1.name = "b1"

        r.add_metric("b2", "g0", "rate", 60.0)
        acc2 = r.accumulator["b2"]
        acc2.name = "b2"

        with patch("builtins.print"):
            r.dump_metric_table(acc1, show_header=True)
            assert r.header_shown is True

        acc2.shown = False
        with patch("builtins.print") as mock_print:
            r.dump_metric_table(acc2, show_header=True)
        header_calls = [
            c for c in mock_print.call_args_list if "name" in str(c)
        ]
        assert len(header_calls) == 0


# ---------------------------------------------------------------------------
# on_event / show_progress_line
# ---------------------------------------------------------------------------

class TestOnEvent:
    def test_on_event_sets_current_state(self):
        r = _build_report()
        config = {"name": "bench_x", "job-number": 2, "device": 3}
        entry = _make_entry("data", {"rate": 1.0}, config=config, tag="run.1")
        r.current_bench = "bench_x"
        r.current_group = "N2-D3"

        with patch.object(r, "show_progress_line"):
            r.on_event(entry)
        assert r.current_bench == "bench_x"
        assert r.current_group == "N2-D3"
        assert r.current_acc is r.accumulator["bench_x"]

    def test_show_progress_line_throttled(self):
        r = _build_report()
        r.add_metric("b1", "g0", "rate", 100.0)
        r.current_acc = r.accumulator["b1"]
        r.current_acc.name = "b1"
        r.time = time.time()

        with patch("builtins.print") as mock_print:
            r.show_progress_line()
        assert mock_print.call_count == 0

    def test_show_progress_line_after_interval(self):
        r = _build_report()
        r.add_metric("b1", "g0", "rate", 100.0)
        r.current_acc = r.accumulator["b1"]
        r.current_acc.name = "b1"
        r.time = time.time() - 1.0

        with patch("builtins.print") as mock_print:
            r.show_progress_line()
        assert mock_print.call_count == 1


# ---------------------------------------------------------------------------
# group_reduce edge cases
# ---------------------------------------------------------------------------

class TestGroupReduce:
    @pytest.fixture()
    def report(self):
        return _build_report()

    def test_load_averaged(self, report):
        result = report.group_reduce("load", "mean", [10.0, 20.0])
        assert result == pytest.approx(15.0)

    def test_std_quadrature(self, report):
        result = report.group_reduce("rate", "std", np.array([3.0, 4.0]))
        assert result == pytest.approx(5.0)

    def test_sem_quadrature(self, report):
        result = report.group_reduce("rate", "sem", np.array([3.0, 4.0]))
        assert result == pytest.approx(5.0)

    def test_default_sum(self, report):
        result = report.group_reduce("rate", "mean", [10, 20, 30])
        assert result == 60


# ---------------------------------------------------------------------------
# report() method
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_returns_zero(self):
        r = _build_report()
        assert r.report({}) == 0
        assert r.report({"a": 1}, extra="x") == 0


# ---------------------------------------------------------------------------
# ReportGPUPerf – line 296
# ---------------------------------------------------------------------------

class TestReportGPUPerf:
    def test_groupkey_always_all(self):
        with patch("builtins.print"):
            r = ReportGPUPerf()
        assert r.groupkey({"job-number": 5, "device": 2}) == "all"
        assert r.groupkey({}) == "all"


# ---------------------------------------------------------------------------
# LivePrinter – empty class
# ---------------------------------------------------------------------------

class TestLivePrinter:
    def test_instantiation(self):
        lp = LivePrinter()
        assert lp is not None


# ---------------------------------------------------------------------------
# Edge cases: empty data, reduce with no values
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_reduce_empty_metrics(self):
        r = _build_report()
        acc = MetricAcc(name="empty")
        reduced = r.reduce(acc)
        assert reduced == {}

    def test_bench_line_with_nan(self):
        """When metrics have no data the line should still render without error."""
        r = _build_report()
        acc = r.accumulator["b1"]
        acc.name = "b1"
        acc.started = 1
        acc.failures = 0
        r.add_metric("b1", "g0", "rate", 100.0)
        line = r.bench_line(acc)
        assert isinstance(line, str)

    def test_on_data_empty_payload(self):
        r = _build_report()
        r.current_bench = "b1"
        r.current_group = "g0"
        entry = _make_entry("data", {})
        r.on_data(entry)
        acc = r.accumulator["b1"]
        assert len(acc.metrics) == 0

    def test_multiple_gpu_devices_in_single_data_event(self):
        r = _build_report()
        r.current_bench = "b1"
        r.current_group = "g0"
        gpu_payload = {
            "gpu0": {"memory": [0.5], "load": 0.8},
            "gpu1": {"memory": [0.6], "load": 0.9},
        }
        entry = _make_entry("data", {"gpudata": gpu_payload})
        r.on_data(entry)
        acc = r.accumulator["b1"]
        assert len(acc.metrics["memory"]["g0"]) == 2
        assert len(acc.metrics["load"]["g0"]) == 2
