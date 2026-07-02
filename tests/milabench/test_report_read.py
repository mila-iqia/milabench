"""Comprehensive tests for milabench.report.read — pure-logic functions.

Focuses on testable functions that do not require GPU or remote infrastructure:
tag parsing, value flattening, metadata extraction, event dispatching,
energy estimation, aggregation, and score computation.
"""

import json
import queue
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from milabench.report.read import (
    EventProcessor,
    EventTracking,
    MetricExtractor,
    BenchmarkStatusExtractor,
    MetaExtractor,
    ConfigExtractor,
    LogExtractor,
    _Value,
    Threading,
    accumulate_per_bench,
    accumulate_per_device,
    aggregate,
    augment_energy_estimator,
    extract_meta_from_run_folder,
    extract_tags,
    flatten_values,
    insert_path,
    make_tags,
    nice_cpu_count,
    workitem_readfile,
    workitem_readfolder,
    DEFAULT_IGNORED,
    bench_tags,
    run_tags,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_worker_pack():
    """Create a minimal Threading-based worker pack for unit-testing Workers."""
    work_q = queue.Queue()
    result_q = queue.Queue()
    error_q = queue.Queue()
    active = threading.Event()
    active.set()
    pending = _Value("i", 0)
    done = _Value("i", 0)
    results = _Value("i", 0)
    return work_q, result_q, error_q, active, pending, done, results


def _drain_results(result_q):
    items = []
    while not result_q.empty():
        items.append(result_q.get_nowait())
    return items


def _write_data_file(path, events):
    """Write a list of dicts as JSON-lines to a .data file."""
    with open(path, "w") as fp:
        for ev in events:
            fp.write(json.dumps(ev) + "\n")


# ===========================================================================
# make_tags / extract_tags
# ===========================================================================

class TestMakeTags:
    def test_creates_compiled_patterns(self):
        tags = make_tags(["foo=bar([0-9]+)", "baz=qux([a-z]+)"])
        assert "foo" in tags
        assert "baz" in tags
        assert tags["foo"].search("bar42").group(1) == "42"

    def test_no_match(self):
        tags = make_tags(["x=prefix([0-9]+)"])
        assert tags["x"].search("no_numbers_here") is None


class TestExtractTags:
    def test_extracts_known_bench_tags(self):
        name = "conc8.mxctx1024.w4.D0"
        result = dict(extract_tags(name, bench_tags))
        assert result["concurrency"] == "8"
        assert result["max_context"] == "1024"
        assert result["worker"] == "4"
        assert result["device"] == "0"

    def test_extracts_run_tags(self):
        name = "g1440.p600.o350.2025-12-26_09-59-43"
        result = dict(extract_tags(name, run_tags))
        assert result["clock"] == "1440"
        assert result["power"] == "600"
        assert result["observation"] == "350"

    def test_no_matching_tags(self):
        result = list(extract_tags("nothing_here", bench_tags))
        assert result == []

    def test_partial_match(self):
        result = dict(extract_tags("D7.something", bench_tags))
        assert result == {"device": "7"}

    def test_capacity_with_Go_suffix(self):
        result = dict(extract_tags("c64Go", bench_tags))
        assert result["capacity"] == "64Go"

    def test_capacity_without_suffix(self):
        result = dict(extract_tags("c128", bench_tags))
        assert result["capacity"] == "128"


# ===========================================================================
# workitem helpers
# ===========================================================================

class TestWorkItemHelpers:
    def test_readfolder(self):
        item = workitem_readfolder(Path("/a/b"), {"k": 1})
        assert item == {"action": "folder", "value": "/a/b", "meta": {"k": 1}}

    def test_readfile(self):
        item = workitem_readfile(Path("/x/y.data"), {"k": 2})
        assert item == {"action": "file", "value": "/x/y.data", "meta": {"k": 2}}


# ===========================================================================
# flatten_values
# ===========================================================================

class TestFlattenValues:
    def test_flat_dict(self):
        result = dict(flatten_values({"a": 1, "b": 2}))
        assert result == {"a": 1, "b": 2}

    def test_nested_dict(self):
        result = dict(flatten_values({"a": {"b": 3}}))
        assert result == {"a.b": 3}

    def test_list_values(self):
        result = dict(flatten_values({"x": [10, 20]}))
        assert result == {"x.0": 10, "x.1": 20}

    def test_deeply_nested(self):
        result = dict(flatten_values({"a": {"b": {"c": 42}}}))
        assert result == {"a.b.c": 42}

    def test_empty_dict(self):
        result = list(flatten_values({}))
        assert result == []

    def test_empty_list(self):
        result = list(flatten_values([]))
        assert result == []

    def test_scalar_at_root(self):
        result = list(flatten_values(99))
        assert result == [("", 99)]

    def test_mixed_nesting(self):
        payload = {"a": [{"b": 1}, {"b": 2}], "c": 3}
        result = dict(flatten_values(payload))
        assert result["a.0.b"] == 1
        assert result["a.1.b"] == 2
        assert result["c"] == 3

    def test_with_namespace(self):
        result = dict(flatten_values({"k": 5}, namespace=("prefix",)))
        assert result == {"prefix.k": 5}


# ===========================================================================
# insert_path
# ===========================================================================

class TestInsertPath:
    def test_first_insert(self):
        meta = {}
        entry = MagicMock()
        entry.name = "folder_a"
        insert_path(meta, "p", entry)
        assert meta["p0"] == "folder_a"

    def test_duplicate_is_skipped(self):
        meta = {"p0": "folder_a"}
        entry = MagicMock()
        entry.name = "folder_a"
        insert_path(meta, "p", entry)
        assert "p1" not in meta

    def test_sequential_inserts(self):
        meta = {}
        for i, name in enumerate(["a", "b", "c"]):
            entry = MagicMock()
            entry.name = name
            insert_path(meta, "p", entry)
        assert meta == {"p0": "a", "p1": "b", "p2": "c"}


# ===========================================================================
# extract_meta_from_run_folder
# ===========================================================================

class TestExtractMetaFromRunFolder:
    def test_with_date_and_run_tags(self, tmp_path):
        folder = tmp_path / "g1440.p600.o350.2025-12-26_09-59-43"
        folder.mkdir()
        result = extract_meta_from_run_folder(folder, {})
        assert result["clock"] == "1440"
        assert result["power"] == "600"
        assert result["observation"] == "350"
        assert result["date"] == datetime(2025, 12, 26, 9, 59, 43)
        assert result["p0"] == folder.name

    def test_no_tags(self, tmp_path):
        folder = tmp_path / "plain_folder"
        folder.mkdir()
        result = extract_meta_from_run_folder(folder, {})
        assert "clock" not in result
        assert "date" not in result
        assert result["p0"] == "plain_folder"

    def test_preserves_existing_meta(self, tmp_path):
        folder = tmp_path / "p600.2025-01-01_00-00-00"
        folder.mkdir()
        result = extract_meta_from_run_folder(folder, {"existing": "value"})
        assert result["existing"] == "value"
        assert result["power"] == "600"


# ===========================================================================
# EventTracking
# ===========================================================================

class TestEventTracking:
    def test_default_rc_none_is_not_success(self):
        t = EventTracking()
        # rc_code=None means "not finished yet", None != 0 → failure
        assert t.success() is False

    def test_rc_zero_is_success(self):
        t = EventTracking(rc_code=0)
        assert t.success() is True

    def test_rc_nonzero_is_failure(self):
        t = EventTracking(rc_code=1)
        assert t.success() is False

    def test_stop_overrides_failure(self):
        t = EventTracking(rc_code=1, stop=True)
        assert t.success() is True

    def test_stop_with_zero_rc(self):
        t = EventTracking(rc_code=0, stop=True)
        assert t.success() is True

    def test_default_values(self):
        t = EventTracking()
        assert t.start_time is None
        assert t.rc_code is None
        assert t.stop is False
        assert t.error is False
        assert t.gpu_count == 1


# ===========================================================================
# _Value (thread-safe value wrapper)
# ===========================================================================

class TestValue:
    def test_stores_value(self):
        v = _Value("i", 42)
        assert v.value == 42

    def test_get_lock_is_reentrant(self):
        v = _Value("i", 0)
        lock = v.get_lock()
        with lock:
            with lock:
                v.value = 10
        assert v.value == 10


# ===========================================================================
# nice_cpu_count
# ===========================================================================

class TestNiceCpuCount:
    def test_returns_at_least_one(self):
        assert nice_cpu_count() >= 1


# ===========================================================================
# EventProcessor — processline dispatching
# ===========================================================================

class TestEventProcessorDispatch:
    def _make_processor(self):
        pack = _make_worker_pack()
        return EventProcessor(pack)

    def test_dispatch_config(self):
        proc = self._make_processor()
        proc.config = MagicMock(return_value=None)
        proc.processline({"event": "config"}, {})
        proc.config.assert_called_once()

    def test_dispatch_meta(self):
        proc = self._make_processor()
        proc.meta = MagicMock(return_value=None)
        proc.processline({"event": "meta"}, {})
        proc.meta.assert_called_once()

    def test_dispatch_start(self):
        proc = self._make_processor()
        proc.start = MagicMock(return_value=None)
        proc.processline({"event": "start"}, {})
        proc.start.assert_called_once()

    def test_dispatch_data(self):
        proc = self._make_processor()
        proc.data = MagicMock(return_value=None)
        proc.processline({"event": "data", "data": {"rate": 100}}, {})
        proc.data.assert_called_once_with({"rate": 100}, {})

    def test_dispatch_line(self):
        proc = self._make_processor()
        proc.line = MagicMock(return_value=None)
        proc.processline({"event": "line", "data": "hello", "pipe": "stdout"}, {})
        proc.line.assert_called_once_with("hello", "stdout", {})

    def test_dispatch_end(self):
        proc = self._make_processor()
        proc.end = MagicMock(return_value=None)
        proc.processline({"event": "end"}, {})
        proc.end.assert_called_once()

    def test_dispatch_error(self):
        proc = self._make_processor()
        proc.error = MagicMock(return_value=None)
        proc.processline({"event": "error"}, {})
        proc.error.assert_called_once()

    def test_dispatch_stop(self):
        proc = self._make_processor()
        proc.stop = MagicMock(return_value=None)
        proc.processline({"event": "stop"}, {})
        proc.stop.assert_called_once()

    def test_dispatch_message(self):
        proc = self._make_processor()
        proc.message = MagicMock(return_value=None)
        proc.processline({"event": "message"}, {})
        proc.message.assert_called_once()

    def test_dispatch_format_error(self):
        proc = self._make_processor()
        proc.format_error = MagicMock(return_value=None)
        proc.processline({"event": "format_error"}, {})
        proc.format_error.assert_called_once()

    def test_dispatch_phase(self):
        proc = self._make_processor()
        proc.phase = MagicMock(return_value=None)
        proc.processline({"event": "phase"}, {})
        proc.phase.assert_called_once()

    def test_dispatch_overseer_error(self):
        proc = self._make_processor()
        proc.overseer_error = MagicMock(return_value=None)
        proc.processline({"event": "overseer_error"}, {})
        proc.overseer_error.assert_called_once()

    def test_unknown_event_does_not_raise(self, capsys):
        proc = self._make_processor()
        proc.processline({"event": "totally_unknown"}, {})
        captured = capsys.readouterr()
        assert "Unhandled event" in captured.out


# ===========================================================================
# EventProcessor.readfile — file reading with .data files
# ===========================================================================

class TestEventProcessorReadfile:
    def test_reads_data_file(self, tmp_path):
        data_file = tmp_path / "bench.data"
        events = [
            {"event": "start", "data": {"time": 100.0}},
            {"event": "data", "data": {"rate": 50, "unit": "item/s", "time": 1.0}},
            {"event": "end", "data": {"time": 200.0, "return_code": 0}},
        ]
        _write_data_file(data_file, events)

        pack = _make_worker_pack()
        proc = EventProcessor(pack)
        proc.start = MagicMock()
        proc.data = MagicMock()
        proc.end = MagicMock()

        proc.readfile(str(data_file), {"bench": "test"})

        proc.start.assert_called_once()
        proc.data.assert_called_once()
        proc.end.assert_called_once()

    def test_skips_non_data_files(self, tmp_path):
        txt_file = tmp_path / "bench.txt"
        txt_file.write_text("not a data file")

        pack = _make_worker_pack()
        proc = EventProcessor(pack)
        proc.start = MagicMock()
        proc.readfile(str(txt_file), {})
        proc.start.assert_not_called()

    def test_accept_file_filter(self, tmp_path):
        data_file = tmp_path / "bench.data"
        _write_data_file(data_file, [{"event": "start", "data": {"time": 1.0}}])

        pack = _make_worker_pack()
        proc = EventProcessor(pack, accept_file=lambda f, m: False)
        proc.start = MagicMock()
        proc.readfile(str(data_file), {})
        proc.start.assert_not_called()

    def test_early_stop_via_processline(self, tmp_path):
        data_file = tmp_path / "bench.data"
        events = [
            {"event": "config", "data": {}},
            {"event": "data", "data": {"rate": 50, "time": 1.0}},
            {"event": "end", "data": {"time": 2.0, "return_code": 0}},
        ]
        _write_data_file(data_file, events)

        pack = _make_worker_pack()
        proc = EventProcessor(pack)
        proc.config = MagicMock(return_value=True)
        proc.data = MagicMock()
        proc.end = MagicMock()

        proc.readfile(str(data_file), {})

        proc.config.assert_called_once()
        proc.data.assert_not_called()
        proc.end.assert_not_called()

    def test_empty_data_file(self, tmp_path):
        data_file = tmp_path / "bench.data"
        data_file.write_text("")

        pack = _make_worker_pack()
        proc = EventProcessor(pack)
        proc.readfile(str(data_file), {})

    def test_run_id_is_deterministic(self, tmp_path):
        data_file = tmp_path / "bench.data"
        _write_data_file(data_file, [{"event": "start", "data": {"time": 1.0}}])

        meta_captures = []

        pack = _make_worker_pack()
        proc = EventProcessor(pack)
        original_start = EventProcessor.start

        def capture_start(self, event, meta):
            meta_captures.append(dict(meta))

        proc.start = lambda ev, m: capture_start(proc, ev, m)
        proc.readfile(str(data_file), {})

        proc2 = EventProcessor(_make_worker_pack())
        proc2.start = lambda ev, m: capture_start(proc2, ev, m)
        proc2.readfile(str(data_file), {})

        assert meta_captures[0]["run_id"] == meta_captures[1]["run_id"]


# ===========================================================================
# MetaExtractor / ConfigExtractor / LogExtractor
# ===========================================================================

class TestMetaExtractor:
    def test_extracts_meta_and_stops(self, tmp_path):
        data_file = tmp_path / "bench.data"
        events = [
            {"event": "meta", "data": {"arch": "x86", "gpu": "A100"}},
            {"event": "data", "data": {"rate": 50, "time": 1.0}},
        ]
        _write_data_file(data_file, events)

        pack = _make_worker_pack()
        proc = MetaExtractor(pack)
        proc.readfile(str(data_file), {})

        results = _drain_results(pack[1])
        assert len(results) == 1
        assert results[0]["arch"] == "x86"


class TestConfigExtractor:
    def test_extracts_config_and_stops(self, tmp_path):
        data_file = tmp_path / "bench.data"
        events = [
            {"event": "config", "data": {"batch_size": 32}},
            {"event": "data", "data": {"rate": 50, "time": 1.0}},
        ]
        _write_data_file(data_file, events)

        pack = _make_worker_pack()
        proc = ConfigExtractor(pack)
        proc.readfile(str(data_file), {})

        results = _drain_results(pack[1])
        assert len(results) == 1
        assert results[0]["batch_size"] == 32


class TestLogExtractor:
    def test_extracts_errors(self, tmp_path):
        data_file = tmp_path / "bench.data"
        events = [
            {"event": "error", "data": {"msg": "OOM"}},
        ]
        _write_data_file(data_file, events)

        pack = _make_worker_pack()
        proc = LogExtractor(pack)
        proc.readfile(str(data_file), {"bench": "test"})

        results = _drain_results(pack[1])
        assert len(results) == 1
        assert results[0]["bench"] == "test"

    def test_extracts_lines(self, tmp_path):
        data_file = tmp_path / "bench.data"
        events = [
            {"event": "line", "data": "some output", "pipe": "stdout"},
        ]
        _write_data_file(data_file, events)

        pack = _make_worker_pack()
        proc = LogExtractor(pack)
        proc.readfile(str(data_file), {"bench": "test"})

        results = _drain_results(pack[1])
        assert len(results) == 1
        assert results[0]["text"] == "some output"
        assert results[0]["pipe"] == "stdout"

    def test_extracts_messages(self, tmp_path):
        data_file = tmp_path / "bench.data"
        events = [
            {"event": "message", "data": {"level": "info", "msg": "starting"}},
        ]
        _write_data_file(data_file, events)

        pack = _make_worker_pack()
        proc = LogExtractor(pack)
        proc.readfile(str(data_file), {"bench": "test"})

        results = _drain_results(pack[1])
        assert len(results) == 1
        assert results[0]["msg"] == "starting"


# ===========================================================================
# MetricExtractor
# ===========================================================================

class TestMetricExtractor:
    def _make_extractor(self):
        pack = _make_worker_pack()
        ext = MetricExtractor(pack)
        return ext, pack

    def test_data_produces_flat_metrics(self):
        ext, pack = self._make_extractor()
        data = {"rate": 100.5, "loss": 0.3, "unit": "item/s", "time": 1000.0, "task": "train"}
        ext.data(data, {"bench": "resnet"})

        results = _drain_results(pack[1])
        metrics = {r["metric"]: r for r in results}
        assert "rate" in metrics
        assert metrics["rate"]["value"] == 100.5
        assert metrics["rate"]["unit"] == "item/s"
        assert "loss" in metrics

    def test_data_ignores_default_ignored(self):
        ext, pack = self._make_extractor()
        data = {"$queued": 5, "progress": 50, "rate": 10, "time": 1.0}
        ext.data(data, {"bench": "test"})

        results = _drain_results(pack[1])
        metric_names = {r["metric"] for r in results}
        assert "$queued" not in metric_names
        assert "progress" not in metric_names
        assert "rate" in metric_names

    def test_data_with_batch_id(self):
        ext, pack = self._make_extractor()
        data = {"rate": 10, "time": 1.0, "batch_id": 42}
        ext.data(data, {"bench": "test"})

        results = _drain_results(pack[1])
        assert results[0]["batch_id"] == 42

    def test_start_end_produces_elapsed_and_success(self):
        ext, pack = self._make_extractor()
        meta = {"bench": "test", "run_id": "abc"}

        ext.start({"event": "start", "data": {"time": 100.0}}, meta)
        ext.end({"event": "end", "data": {"time": 150.0, "return_code": 0}}, meta)

        results = _drain_results(pack[1])
        metrics = {r["metric"]: r for r in results}
        assert "elapsed" in metrics
        assert metrics["elapsed"]["value"] == pytest.approx(50.0)
        assert "success" in metrics
        assert metrics["success"]["value"] == 1
        assert "ngpu" in metrics

    def test_start_end_failure(self):
        ext, pack = self._make_extractor()
        meta = {"bench": "test", "run_id": "abc"}

        ext.start({"event": "start", "data": {"time": 100.0}}, meta)
        ext.end({"event": "end", "data": {"time": 110.0, "return_code": 1}}, meta)

        results = _drain_results(pack[1])
        metrics = {r["metric"]: r for r in results}
        assert metrics["success"]["value"] == 0

    def test_stop_before_end_marks_success(self):
        ext, pack = self._make_extractor()
        meta = {"bench": "test", "run_id": "abc"}

        ext.start({"event": "start", "data": {"time": 100.0}}, meta)
        ext.stop({"event": "stop"}, meta)
        ext.end({"event": "end", "data": {"time": 110.0, "return_code": 1}}, meta)

        results = _drain_results(pack[1])
        metrics = {r["metric"]: r for r in results}
        assert metrics["success"]["value"] == 1

    def test_nested_data_is_flattened(self):
        ext, pack = self._make_extractor()
        data = {"loss": {"train": 0.5, "val": 0.8}, "time": 1.0}
        ext.data(data, {"bench": "test"})

        results = _drain_results(pack[1])
        metric_names = {r["metric"] for r in results}
        assert "loss.train" in metric_names
        assert "loss.val" in metric_names

    def test_gpudata_single_device(self):
        ext, pack = self._make_extractor()
        data = {
            "gpudata": {"0": {"temperature": 65.0, "power": 300.0, "memory": [1000, 2000]}},
            "time": 1.0,
            "task": "train",
        }
        ext.data(data, {"bench": "test", "device": "0"})

        results = _drain_results(pack[1])
        metrics = {r["metric"]: r for r in results}
        assert "gpudata.temperature" in metrics
        assert "gpudata.power" in metrics
        assert "gpudata.memory.0" in metrics
        assert "gpudata.memory.1" in metrics
        assert metrics["gpudata.memory.0"]["value"] == 1000
        assert metrics["gpudata.memory.1"]["value"] == 2000

    def test_gpudata_multi_device_averages(self):
        ext, pack = self._make_extractor()
        data = {
            "gpudata": {
                "0": {"power": 300.0},
                "1": {"power": 400.0},
            },
            "time": 1.0,
            "task": "train",
        }
        ext.data(data, {"bench": "test"})

        results = _drain_results(pack[1])
        metrics = {r["metric"]: r for r in results}
        assert "gpudata.power" in metrics
        assert metrics["gpudata.power"]["value"] == pytest.approx(350.0)
        assert metrics["gpudata.power"]["count"] == 2

    def test_units_field_takes_priority(self):
        ext, pack = self._make_extractor()
        data = {"rate": 10, "unit": "a", "units": "b", "time": 1.0}
        ext.data(data, {"bench": "test"})

        results = _drain_results(pack[1])
        assert results[0]["unit"] == "b"


# ===========================================================================
# BenchmarkStatusExtractor
# ===========================================================================

class TestBenchmarkStatusExtractor:
    def test_success_status(self, tmp_path):
        data_file = tmp_path / "bench.data"
        events = [
            {"event": "start", "data": {"time": 100.0}},
            {"event": "end", "data": {"time": 200.0, "return_code": 0}},
        ]
        _write_data_file(data_file, events)

        pack = _make_worker_pack()
        proc = BenchmarkStatusExtractor(pack)
        proc.readfile(str(data_file), {"bench": "test", "run_id": "x"})

        results = _drain_results(pack[1])
        assert len(results) == 1
        assert results[0]["status"] == "success"

    def test_failed_status(self, tmp_path):
        data_file = tmp_path / "bench.data"
        events = [
            {"event": "start", "data": {"time": 100.0}},
            {"event": "end", "data": {"time": 200.0, "return_code": 1}},
        ]
        _write_data_file(data_file, events)

        pack = _make_worker_pack()
        proc = BenchmarkStatusExtractor(pack)
        proc.readfile(str(data_file), {"bench": "test", "run_id": "x"})

        results = _drain_results(pack[1])
        assert len(results) == 1
        assert results[0]["status"] == "failed"

    def test_stop_makes_success(self, tmp_path):
        data_file = tmp_path / "bench.data"
        events = [
            {"event": "start", "data": {"time": 100.0}},
            {"event": "stop", "data": {}},
            {"event": "end", "data": {"time": 200.0, "return_code": 1}},
        ]
        _write_data_file(data_file, events)

        pack = _make_worker_pack()
        proc = BenchmarkStatusExtractor(pack)
        proc.readfile(str(data_file), {"bench": "test", "run_id": "x"})

        results = _drain_results(pack[1])
        assert results[0]["status"] == "success"


# ===========================================================================
# augment_energy_estimator
# ===========================================================================

class TestAugmentEnergyEstimator:
    def test_no_power_metrics_passes_through(self):
        metrics = [
            {"metric": "rate", "value": 100, "time": 1, "bench": "a", "p0": "r"},
        ]
        result = list(augment_energy_estimator(metrics))
        assert len(result) == 1
        assert result[0]["metric"] == "rate"

    def test_single_power_metric_no_energy(self):
        metrics = [
            {"metric": "gpudata.power", "value": 300, "time": 1.0, "bench": "a", "p0": "r", "count": 1},
        ]
        result = list(augment_energy_estimator(metrics))
        assert len(result) == 1

    def test_two_power_metrics_produce_energy(self):
        metrics = [
            {"metric": "gpudata.power", "value": 300, "time": 0.0, "bench": "a", "p0": "r", "count": 1},
            {"metric": "gpudata.power", "value": 400, "time": 1.0, "bench": "a", "p0": "r", "count": 1},
        ]
        result = list(augment_energy_estimator(metrics))
        energy_metrics = [m for m in result if m["metric"] == "energy"]
        assert len(energy_metrics) == 1
        assert energy_metrics[0]["value"] == pytest.approx(350.0)

    def test_multi_gpu_energy(self):
        metrics = [
            {"metric": "gpudata.power", "value": 200, "time": 0.0, "bench": "a", "p0": "r", "count": 4},
            {"metric": "gpudata.power", "value": 200, "time": 2.0, "bench": "a", "p0": "r", "count": 4},
        ]
        result = list(augment_energy_estimator(metrics))
        energy_metrics = [m for m in result if m["metric"] == "energy"]
        # energy = elapsed * count * (p1+p2)/2 = 2 * 4 * (200+200)/2 = 1600
        assert energy_metrics[0]["value"] == pytest.approx(1600.0)

    def test_different_devices_tracked_separately(self):
        metrics = [
            {"metric": "gpudata.power", "value": 300, "time": 0.0, "bench": "a", "p0": "r", "device": 0, "count": 1},
            {"metric": "gpudata.power", "value": 400, "time": 0.0, "bench": "a", "p0": "r", "device": 1, "count": 1},
            {"metric": "gpudata.power", "value": 300, "time": 1.0, "bench": "a", "p0": "r", "device": 0, "count": 1},
            {"metric": "gpudata.power", "value": 400, "time": 1.0, "bench": "a", "p0": "r", "device": 1, "count": 1},
        ]
        result = list(augment_energy_estimator(metrics))
        energy_metrics = [m for m in result if m["metric"] == "energy"]
        assert len(energy_metrics) == 2

    def test_sorts_by_time_when_forced(self):
        metrics = [
            {"metric": "gpudata.power", "value": 400, "time": 1.0, "bench": "a", "p0": "r", "count": 1},
            {"metric": "gpudata.power", "value": 300, "time": 0.0, "bench": "a", "p0": "r", "count": 1},
        ]
        result = list(augment_energy_estimator(metrics, force_sort=True))
        energy_metrics = [m for m in result if m["metric"] == "energy"]
        assert len(energy_metrics) == 1
        assert energy_metrics[0]["value"] == pytest.approx(350.0)

    def test_none_time_sorted_to_front(self):
        metrics = [
            {"metric": "rate", "value": 10, "time": None, "bench": "a", "p0": "r"},
            {"metric": "gpudata.power", "value": 300, "time": 1.0, "bench": "a", "p0": "r", "count": 1},
        ]
        result = list(augment_energy_estimator(metrics, force_sort=True))
        assert len(result) == 2


# ===========================================================================
# aggregate
# ===========================================================================

class TestAggregate:
    def test_basic_aggregation(self):
        metrics = [
            {"bench": "resnet", "device": 0, "p0": "run1", "metric": "rate", "value": 100},
            {"bench": "resnet", "device": 0, "p0": "run1", "metric": "rate", "value": 200},
            {"bench": "resnet", "device": 0, "p0": "run1", "metric": "loss", "value": 0.5},
        ]
        result = aggregate(metrics)
        key = ("resnet", 0, "run1")
        assert result[key]["rate"] == [100, 200]
        assert result[key]["loss"] == [0.5]

    def test_different_devices_separate_keys(self):
        metrics = [
            {"bench": "resnet", "device": 0, "p0": "r1", "metric": "rate", "value": 100},
            {"bench": "resnet", "device": 1, "p0": "r1", "metric": "rate", "value": 200},
        ]
        result = aggregate(metrics)
        assert ("resnet", 0, "r1") in result
        assert ("resnet", 1, "r1") in result

    def test_missing_device_defaults_minus_one(self):
        metrics = [
            {"bench": "bert", "p0": "r1", "metric": "rate", "value": 50},
        ]
        result = aggregate(metrics)
        assert ("bert", -1, "r1") in result

    def test_empty_metrics(self):
        result = aggregate([])
        assert result == {}


# ===========================================================================
# accumulate_per_device
# ===========================================================================

class TestAccumulatePerDevice:
    def test_applies_accumulator_function(self):
        agg = {
            ("resnet", 0, "r1"): {"rate": [100, 200, 300]},
            ("resnet", 1, "r1"): {"rate": [400, 500, 600]},
        }
        acc_fun = {"rate": lambda v: sum(v) / len(v)}
        result = accumulate_per_device(agg, acc_fun)

        key = ("resnet", "r1")
        assert result[key]["rate"][0] == pytest.approx(200.0)
        assert result[key]["rate"][1] == pytest.approx(500.0)

    def test_unknown_metric_skipped(self):
        agg = {
            ("resnet", 0, "r1"): {"unknown_metric": [1, 2, 3]},
        }
        acc_fun = {"rate": sum}
        result = accumulate_per_device(agg, acc_fun)
        key = ("resnet", "r1")
        assert "unknown_metric" not in result.get(key, {})


# ===========================================================================
# accumulate_per_bench
# ===========================================================================

class TestAccumulatePerBench:
    def test_scalar_accumulator(self):
        accumulated = {
            ("resnet", "r1"): {"rate": {0: 200, 1: 400}},
        }
        acc_fun = {"rate": sum}
        result = list(accumulate_per_bench(accumulated, acc_fun))
        assert len(result) == 1
        assert result[0]["bench"] == "resnet"
        assert result[0]["metric"] == "rate"
        assert result[0]["value"] == 600

    def test_dict_accumulator_produces_multiple_metrics(self):
        accumulated = {
            ("bert", "r1"): {"rate": {0: 100, 1: 200}},
        }
        acc_fun = {"rate": {"rate_sum": sum, "rate_max": max}}
        result = list(accumulate_per_bench(accumulated, acc_fun))
        metrics = {r["metric"]: r["value"] for r in result}
        assert metrics["rate_sum"] == 300
        assert metrics["rate_max"] == 200

    def test_empty_input(self):
        result = list(accumulate_per_bench({}, {"rate": sum}))
        assert result == []


# ===========================================================================
# compute_global_score
# ===========================================================================

class TestComputeGlobalScore:
    def test_basic_score(self):
        import numpy as np

        metrics = [
            {"bench": "resnet", "metric": "score", "value": 100},
            {"bench": "bert", "metric": "score", "value": 200},
        ]
        weights = {
            "resnet": {"weight": 1},
            "bert": {"weight": 1},
        }
        from milabench.report.read import compute_global_score

        score = compute_global_score(metrics, weights)
        expected = np.exp(
            (np.log(101) * 1 + np.log(201) * 1) / 2
        )
        assert score == pytest.approx(expected)

    def test_ignores_non_score_metrics(self):
        metrics = [
            {"bench": "resnet", "metric": "rate", "value": 100},
            {"bench": "resnet", "metric": "score", "value": 50},
        ]
        weights = {"resnet": {"weight": 1}}
        from milabench.report.read import compute_global_score

        import numpy as np

        score = compute_global_score(metrics, weights)
        expected = np.exp(np.log(51) * 1 / 1)
        assert score == pytest.approx(expected)

    def test_default_weight_used(self):
        metrics = [{"bench": "a", "metric": "score", "value": 10}]
        weights = {"a": {}}
        from milabench.report.read import compute_global_score

        import numpy as np

        score = compute_global_score(metrics, weights, default_weight=2)
        expected = np.exp(np.log(11) * 2 / 2)
        assert score == pytest.approx(expected)


# ===========================================================================
# DEFAULT_IGNORED constant
# ===========================================================================

class TestDefaultIgnored:
    def test_contains_expected_keys(self):
        assert "$queued" in DEFAULT_IGNORED
        assert "progress" in DEFAULT_IGNORED
        assert "progress.0" in DEFAULT_IGNORED
        assert "progress.1" in DEFAULT_IGNORED


# ===========================================================================
# EventProcessor.__call__ task dispatching
# ===========================================================================

class TestEventProcessorCall:
    def test_folder_action(self, tmp_path):
        sub = tmp_path / "run1"
        sub.mkdir()
        data = sub / "bench.data"
        _write_data_file(data, [{"event": "start", "data": {"time": 1.0}}])

        pack = _make_worker_pack()
        proc = EventProcessor(pack)
        task = {"action": "folder", "value": str(sub), "meta": {}}
        proc(task)

        work_items = _drain_results(pack[0])
        assert len(work_items) >= 1

    def test_file_action(self, tmp_path):
        data = tmp_path / "bench.data"
        _write_data_file(data, [{"event": "start", "data": {"time": 1.0}}])

        pack = _make_worker_pack()
        proc = EventProcessor(pack)
        proc.start = MagicMock()
        task = {"action": "file", "value": str(data), "meta": {}}
        proc(task)

        proc.start.assert_called_once()

    def test_unknown_action_raises(self):
        pack = _make_worker_pack()
        proc = EventProcessor(pack)
        with pytest.raises(RuntimeError, match="Unknown action"):
            proc({"action": "invalid", "value": "", "meta": {}})


# ===========================================================================
# End-to-end: full .data file through MetricExtractor
# ===========================================================================

class TestMetricExtractorEndToEnd:
    def test_full_pipeline(self, tmp_path):
        data_file = tmp_path / "resnet.data"
        events = [
            {"event": "start", "data": {"time": 1000.0}},
            {"event": "data", "data": {"rate": 50.0, "unit": "img/s", "time": 1001.0, "task": "train"}},
            {"event": "data", "data": {"rate": 55.0, "unit": "img/s", "time": 1002.0, "task": "train"}},
            {"event": "end", "data": {"time": 1010.0, "return_code": 0}},
        ]
        _write_data_file(data_file, events)

        pack = _make_worker_pack()
        ext = MetricExtractor(pack)
        ext.readfile(str(data_file), {"bench": "resnet", "run_id": "test123"})

        results = _drain_results(pack[1])
        metric_names = {r["metric"] for r in results}
        assert "rate" in metric_names
        assert "elapsed" in metric_names
        assert "success" in metric_names
        assert "ngpu" in metric_names

        elapsed = [r for r in results if r["metric"] == "elapsed"][0]
        assert elapsed["value"] == pytest.approx(10.0)

    def test_malformed_json_raises(self, tmp_path):
        data_file = tmp_path / "bad.data"
        data_file.write_text('{"event": "start", "data": {"time": 1.0}}\nNOT_JSON\n')

        pack = _make_worker_pack()
        ext = MetricExtractor(pack)
        with pytest.raises(json.JSONDecodeError):
            ext.readfile(str(data_file), {"bench": "test"})
