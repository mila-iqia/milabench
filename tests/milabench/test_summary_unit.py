import pytest
from collections import defaultdict
from math import nan, isnan
from unittest.mock import patch

from milabench.summary import (
    aggregate,
    augment,
    _classify,
    _filter_failures,
    _keep_latest,
    _merge,
    _metrics,
    _summarize,
    make_summary,
    node_count,
    local_gpu,
    ngpu,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic run data
# ---------------------------------------------------------------------------

def _config_entry(name="bench1", tag=None, device=None, num_machines=1, plan=None, **extra):
    cfg = {"name": name, "tag": tag or [], "group": "grp", **extra}
    if device is not None:
        cfg["device"] = device
    if num_machines != 1:
        cfg["num_machines"] = num_machines
    if plan is not None:
        cfg["plan"] = plan
    return {"event": "config", "data": cfg}


def _meta_entry(**kw):
    return {"event": "meta", "data": kw}


def _start_entry(time=100.0):
    return {"event": "start", "data": {"time": time}}


def _end_entry(time=200.0, return_code=0):
    return {"event": "end", "data": {"time": time, "return_code": return_code}}


def _data_entry(task=None, time=None, **kw):
    d = dict(kw)
    if task is not None:
        d["task"] = task
    if time is not None:
        d["time"] = time
    return {"event": "data", "data": d}


def _gpu_entry(time=None, **devices):
    d = dict(devices)
    if time is not None:
        d["time"] = time
    return _data_entry(gpudata=d)


def _line_entry(pipe="stdout", data="hello"):
    return {"event": "line", "pipe": pipe, "data": data}


def _stop_entry():
    return {"event": "stop"}


def _make_gpu_device(memory_pct=0.5, load=0.8, power=250.0):
    return {"memory": [memory_pct], "load": load, "power": power}


def _minimal_run(name="bench1", tag=None, device=None, return_code=0,
                 train_rates=None, gpu_times=None, loss_vals=None,
                 num_machines=1, plan=None, extra_data=None, early_stop=False,
                 **config_kw):
    """Build a complete run (list of events) for aggregate()."""
    if train_rates is None:
        train_rates = [100.0, 110.0, 105.0]

    entries = [
        _config_entry(name=name, tag=tag, device=device,
                      num_machines=num_machines, plan=plan, **config_kw),
        _meta_entry(title="test"),
        _start_entry(time=10.0),
    ]

    for r in train_rates:
        entries.append(_data_entry(train_rate=r))

    if loss_vals:
        for l in loss_vals:
            entries.append(_data_entry(loss=l))

    if gpu_times:
        for t, devs in gpu_times:
            entries.append(_gpu_entry(time=t, **devs))

    if extra_data:
        for d in extra_data:
            entries.append(d)

    if early_stop:
        entries.append(_stop_entry())

    entries.append(_end_entry(time=20.0, return_code=return_code))
    return entries


# ---------------------------------------------------------------------------
# Tests for node_count / local_gpu / ngpu
# ---------------------------------------------------------------------------

class TestNodeCountGpu:
    def test_node_count_default(self):
        assert node_count({}) == 1

    def test_node_count_specified(self):
        assert node_count({"num_machines": 4}) == 4

    def test_local_gpu_default_per_gpu(self):
        assert local_gpu({}) == 1

    def test_local_gpu_njobs(self):
        assert local_gpu({"plan": {"method": "njobs"}, "devices": [0, 1, 2]}) == 3

    def test_local_gpu_njobs_empty_devices(self):
        assert local_gpu({"plan": {"method": "njobs"}, "devices": []}) == 0

    def test_ngpu_multi(self):
        cfg = {"num_machines": 2, "plan": {"method": "njobs"}, "devices": [0, 1]}
        assert ngpu(cfg) == 4


# ---------------------------------------------------------------------------
# Tests for aggregate – targeting uncovered lines
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_no_config_returns_none(self):
        """Non-run data without a config event returns None."""
        result = aggregate([_data_entry(train_rate=1.0)])
        assert result is None

    def test_basic_success(self):
        run = _minimal_run()
        result = aggregate(run)
        assert result is not None
        assert result["data"]["success"] == [True]

    def test_line_60_time_in_dict_value(self):
        """Line 60: when data event has time and a dict value, time is injected."""
        run = _minimal_run(extra_data=[
            _data_entry(time=42.0, some_dict={"key": "val"}),
        ])
        result = aggregate(run)
        dicts = result["data"]["some_dict"]
        assert any(d.get("time") == 42.0 for d in dicts)

    def test_line_101_device_filters_missing_key(self):
        """Line 101: gpudata entry without the device key is skipped."""
        gpu_data = [
            (1.0, {"0": _make_gpu_device(), "1": _make_gpu_device()}),
            (2.0, {"1": _make_gpu_device()}),  # device "0" missing
        ]
        run = _minimal_run(device=0, gpu_times=gpu_data)
        result = aggregate(run)
        assert len(result["data"]["gpudata"]) == 1

    def test_lines_117_118_energy_calculation(self):
        """Lines 117-118: energy accumulates power * elapsed between timestamps."""
        dev = _make_gpu_device(power=100.0)
        gpu_data = [
            (10.0, {"0": dev}),
            (12.0, {"0": dev}),
            (15.0, {"0": dev}),
        ]
        run = _minimal_run(gpu_times=gpu_data)
        result = aggregate(run)
        expected_energy = (100.0 * 2.0 + 100.0 * 3.0) / 1000
        assert abs(result["data"]["energy"][0] - expected_energy) < 1e-9

    def test_line_126_elapsed_with_gpu_times(self):
        """Line 126: elapsed = last_time - first_time when gpudata has timestamps."""
        dev = _make_gpu_device()
        gpu_data = [
            (5.0, {"0": dev}),
            (15.0, {"0": dev}),
        ]
        run = _minimal_run(gpu_times=gpu_data)
        result = aggregate(run)
        assert result["data"]["elapsed"] == [10.0]

    def test_elapsed_zero_without_gpu_times(self):
        run = _minimal_run()
        result = aggregate(run)
        assert result["data"]["elapsed"] == [0]

    def test_line_146_nolog_forces_success(self):
        """Line 146: tag containing 'nolog' forces success=True despite failure."""
        run = _minimal_run(tag=["nolog"], return_code=1, train_rates=[])
        result = aggregate(run)
        assert result["data"]["success"] == [True]

    def test_early_stop_success(self):
        run = _minimal_run(return_code=1, early_stop=True)
        result = aggregate(run)
        assert result["data"]["success"] == [True]

    def test_nonzero_return_code_fail(self):
        run = _minimal_run(return_code=1)
        result = aggregate(run)
        assert result["data"]["success"] == [False]

    def test_nan_loss_fail(self):
        run = _minimal_run(loss_vals=[1.0, float("nan")])
        result = aggregate(run)
        assert result["data"]["success"] == [False]

    def test_no_train_rate_fail(self):
        run = _minimal_run(train_rates=[])
        result = aggregate(run)
        assert result["data"]["success"] == [False]

    def test_loss_gain(self):
        run = _minimal_run(loss_vals=[3.0, 2.0, 1.0])
        result = aggregate(run)
        assert result["data"]["loss_gain"] == [-2.0]

    def test_task_rate_renamed(self):
        run = _minimal_run(extra_data=[_data_entry(task="encode", rate=50.0)])
        result = aggregate(run)
        assert "encode_rate" in result["data"]

    def test_per_gpu_populated_when_device_set(self):
        dev = _make_gpu_device()
        gpu_data = [(1.0, {"0": dev})]
        run = _minimal_run(device=0, gpu_times=gpu_data)
        result = aggregate(run)
        per_gpu = result["data"]["per_gpu"]
        assert all(d == 0 for d, _ in per_gpu)

    def test_line_event(self):
        run = _minimal_run(extra_data=[_line_entry(pipe="stderr", data="err")])
        result = aggregate(run)
        assert "err" in result["data"]["stderr"]

    def test_walltime(self):
        run = _minimal_run()
        result = aggregate(run)
        assert result["data"]["walltime"] == [10.0]

    def test_meta_stored(self):
        run = _minimal_run()
        result = aggregate(run)
        assert result["meta"] == {"title": "test"}


# ---------------------------------------------------------------------------
# Tests for _classify / _filter_failures / _keep_latest / _merge
# ---------------------------------------------------------------------------

def _make_agg(name="bench1", success=True, start_time=1.0):
    return {
        "config": {"name": name, "group": "g", "weight": 1, "tag": []},
        "start": {"time": start_time},
        "end": {"time": start_time + 10},
        "data": {
            "success": [success],
            "train_rate": [100.0],
            "walltime": [10.0],
            "gpudata": [],
            "per_gpu": [],
            "ngpu": [1],
            "energy": [0],
            "elapsed": [10],
        },
        "meta": {},
    }


class TestClassifyFilterMerge:
    def test_classify_groups_by_name(self):
        aggs = [_make_agg("a"), _make_agg("b"), _make_agg("a")]
        result = _classify(aggs)
        assert len(result["a"]) == 2
        assert len(result["b"]) == 1

    def test_filter_failures_lines_170_175(self):
        """Lines 170-175: only successful aggregates remain; empty groups dropped."""
        aggs = [_make_agg("a", success=True), _make_agg("a", success=False)]
        classified = {"a": aggs, "b": [_make_agg("b", success=False)]}
        result = _filter_failures(classified)
        assert "a" in result
        assert "b" not in result
        assert len(result["a"]) == 1

    def test_keep_latest_lines_180_184(self):
        """Lines 180-184: only the aggregate with the latest start time survives."""
        old = _make_agg("a", start_time=1.0)
        new = _make_agg("a", start_time=5.0)
        classified = {"a": [old, new]}
        result = _keep_latest(classified)
        assert len(result["a"]) == 1
        assert result["a"][0]["start"]["time"] == 5.0

    def test_keep_latest_empty_list(self):
        result = _keep_latest({"a": []})
        assert "a" not in result

    def test_merge(self):
        a1 = _make_agg("a")
        a2 = _make_agg("a")
        merged = _merge([a1, a2])
        assert len(merged["data"]["train_rate"]) == 2


# ---------------------------------------------------------------------------
# Tests for _metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_basic_metrics(self):
        m = _metrics([10, 20, 30, 40, 50])
        assert m["mean"] == pytest.approx(30.0)
        assert m["min"] <= m["q1"] <= m["median"] <= m["q3"] <= m["max"]

    def test_trims_when_five_or_more(self):
        m = _metrics([1, 2, 3, 4, 5])
        assert m["min"] == pytest.approx(2.0)
        assert m["max"] == pytest.approx(4.0)

    def test_empty_returns_nans(self):
        m = _metrics([])
        assert all(isnan(v) for v in m.values())

    def test_none_values_filtered(self):
        m = _metrics([None, 5, None, 10])
        assert m["mean"] == pytest.approx(7.5)

    def test_single_value(self):
        m = _metrics([42])
        assert m["mean"] == pytest.approx(42.0)
        assert m["std"] == pytest.approx(0.0)

    def test_all_none_returns_nans(self):
        m = _metrics([None, None])
        assert all(isnan(v) for v in m.values())


# ---------------------------------------------------------------------------
# Tests for augment – lines 239-241, 244-246
# ---------------------------------------------------------------------------

class TestAugment:
    def test_elapsed_query(self):
        """Lines 244-246: elapsed computed from start/end times."""
        group = {"start": {"time": 10.0}, "end": {"time": 25.0}, "config": {}}
        result = augment(group, query=("elapsed",))
        assert result["elapsed"] == 15.0

    def test_batch_size_query(self):
        """Lines 239-241: batch_size extracted from config/start."""
        group = {
            "start": {"time": 10.0},
            "end": {"time": 20.0},
            "config": {},
        }
        with patch("milabench.summary.get_batch_size", create=True):
            from milabench.sizer import get_batch_size
            with patch("milabench.sizer.get_batch_size", return_value=32):
                result = augment(group, query=("batch_size",))
                assert result["batch_size"] == 32

    def test_empty_query(self):
        group = {"start": {"time": 1}, "end": {"time": 2}, "config": {}}
        result = augment(group)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests for _summarize – line 291
# ---------------------------------------------------------------------------

def _make_group(gpu_entries=None, train_rates=None, per_gpu_pairs=None,
                success=None, name="bench1"):
    if train_rates is None:
        train_rates = [100, 110, 105]
    if gpu_entries is None:
        gpu_entries = []
    if per_gpu_pairs is None:
        per_gpu_pairs = [(0, r) for r in train_rates]
    if success is None:
        success = [True]

    return {
        "config": {"name": name, "group": "g", "weight": 1, "enabled": True, "tag": []},
        "start": {"time": 10.0},
        "end": {"time": 20.0},
        "data": {
            "success": success,
            "train_rate": train_rates,
            "walltime": [10.0],
            "gpudata": gpu_entries,
            "per_gpu": per_gpu_pairs,
            "ngpu": [1],
            "energy": [5.0],
            "elapsed": [10.0],
        },
        "meta": {"title": "test"},
    }


class TestSummarize:
    def test_basic_summarize(self):
        group = _make_group()
        s = _summarize(group)
        assert s["name"] == "bench1"
        assert s["n"] == 1
        assert s["train_rate"]["mean"] is not None

    def test_line_291_power_in_gpu_load(self):
        """Line 291: power appended to gpu_load stats when present."""
        gpu_entries = [
            {"0": {"memory": [0.5], "load": 0.8, "power": 200}},
            {"0": {"memory": [0.6], "load": 0.9, "power": 250}},
        ]
        group = _make_group(gpu_entries=gpu_entries)
        s = _summarize(group)
        assert "0" in s["gpu_load"]
        assert "power" in s["gpu_load"]["0"]
        assert s["gpu_load"]["0"]["power"]["mean"] == pytest.approx(225.0)

    def test_gpu_load_skips_memory_1_or_load_0(self):
        """GPU entries with memory[0]==1 or load==0 are skipped."""
        gpu_entries = [
            {"0": {"memory": [1], "load": 0.5, "power": 100}},
            {"1": {"memory": [0.5], "load": 0, "power": 100}},
            {"2": {"memory": [0.5], "load": 0.5, "power": 100}},
        ]
        group = _make_group(gpu_entries=gpu_entries)
        s = _summarize(group)
        assert "0" not in s["gpu_load"]
        assert "1" not in s["gpu_load"]
        assert "2" in s["gpu_load"]

    def test_energy_included(self):
        group = _make_group()
        s = _summarize(group)
        assert "energy" in s
        assert s["energy"] == 5.0

    def test_weight_and_enabled(self):
        group = _make_group()
        s = _summarize(group)
        assert s["weight"] == 1
        assert s["enabled"] is True


# ---------------------------------------------------------------------------
# Tests for make_summary – lines 349-354, 359, 362
# ---------------------------------------------------------------------------

class TestMakeSummary:
    def _full_run_dict(self, name="bench1", return_code=0, train_rates=None):
        run = _minimal_run(name=name, return_code=return_code, train_rates=train_rates)
        dev = _make_gpu_device()
        run.insert(-1, _gpu_entry(time=1.0, **{"0": dev}))
        run.insert(-1, _gpu_entry(time=2.0, **{"0": dev}))
        return {f"{name}_run": run}

    def test_basic_make_summary(self):
        runs = self._full_run_dict()
        result = make_summary(runs)
        assert "bench1" in result

    def test_lines_349_354_assertion_error_ignored(self):
        """Lines 349-354: runs that raise AssertionError or Exception are skipped."""
        bad_run = [
            _config_entry(name="bad"),
            _meta_entry(),
            _start_entry(),
            _start_entry(),  # duplicate start triggers AssertionError
            _end_entry(),
        ]
        runs = {"bad_run": bad_run}
        runs.update(self._full_run_dict("good"))
        result = make_summary(runs)
        assert "bad_run" not in result
        assert "good" in result

    def test_lines_349_354_generic_exception_ignored(self):
        """Lines 351-354: generic exceptions during aggregate are caught."""
        runs = {"broken": "not_iterable"}
        runs.update(self._full_run_dict("ok"))
        result = make_summary(runs)
        assert "ok" in result

    def test_line_359_filter_failures(self):
        """Line 359: filter_failures=True drops failed benchmarks."""
        runs = self._full_run_dict("fail_bench", return_code=1, train_rates=[])
        runs.update(self._full_run_dict("pass_bench"))
        result = make_summary(runs, filter_failures=True)
        assert "fail_bench" not in result
        assert "pass_bench" in result

    def test_line_362_latest_only(self):
        """Line 362: latest_only=True keeps only the latest aggregate."""
        runs = self._full_run_dict("bench")
        result = make_summary(runs, latest_only=True)
        assert "bench" in result

    def test_filter_and_latest_combined(self):
        runs = self._full_run_dict("bench")
        result = make_summary(runs, filter_failures=True, latest_only=True)
        assert "bench" in result

    def test_empty_runs(self):
        result = make_summary({})
        assert result == {}


# ---------------------------------------------------------------------------
# Edge cases for aggregate
# ---------------------------------------------------------------------------

class TestAggregateEdgeCases:
    def test_device_specific_gpu_filtering(self):
        """Device-specific filtering keeps only the requested device column."""
        dev0 = _make_gpu_device(power=100)
        dev1 = _make_gpu_device(power=200)
        gpu_data = [(1.0, {"0": dev0, "1": dev1})]
        run = _minimal_run(device=0, gpu_times=gpu_data)
        result = aggregate(run)
        for entry in result["data"]["gpudata"]:
            assert "0" in entry
            assert "1" not in entry

    def test_gpu_no_time_no_energy(self):
        """gpudata without time fields: no energy accumulated."""
        run = _minimal_run(extra_data=[
            _data_entry(gpudata={"0": _make_gpu_device()}),
            _data_entry(gpudata={"0": _make_gpu_device()}),
        ])
        result = aggregate(run)
        assert result["data"]["energy"] == [0.0]

    def test_multiple_devices_energy(self):
        """Energy accumulates across multiple GPUs."""
        dev0 = _make_gpu_device(power=100)
        dev1 = _make_gpu_device(power=200)
        gpu_data = [
            (10.0, {"0": dev0, "1": dev1}),
            (12.0, {"0": dev0, "1": dev1}),
        ]
        run = _minimal_run(gpu_times=gpu_data)
        result = aggregate(run)
        expected = (100 * 2 + 200 * 2) / 1000
        assert abs(result["data"]["energy"][0] - expected) < 1e-9
