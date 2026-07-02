"""Comprehensive tests for milabench.log module."""

import json
import os
import time
from io import StringIO
from unittest.mock import MagicMock, mock_open, patch, PropertyMock

import pytest

from milabench.structs import BenchLogEntry
from milabench.log import (
    BaseLogger,
    BaseReporter,
    DashFormatter,
    DataReporter,
    LongDashFormatter,
    ShortDashFormatter,
    TagConsole,
    TerminalFormatter,
    TextReporter,
    find_byte_exponent,
    formatbyte,
    milabench_pprint,
    new_progress_bar,
    octet_units,
    pretty_print,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakePack:
    """Minimal stand-in for a pack object."""

    def __init__(self, tag="bench.0"):
        self.tag = tag
        self._logfile_path = "/tmp/fake_log"

    def logfile(self, pipe):
        return f"{self._logfile_path}/{pipe}.log"


def _entry(event, data=None, pipe=None, tag="bench.0", pack=None):
    """Create a BenchLogEntry for testing."""
    if pack is None:
        pack = FakePack(tag)
    return BenchLogEntry(pack=pack, event=event, data=data, pipe=pipe)


# ---------------------------------------------------------------------------
# pretty_print
# ---------------------------------------------------------------------------


class TestPrettyPrint:
    def test_returns_callable(self):
        fn = pretty_print()
        assert callable(fn)

    def test_returns_str_representation(self):
        fn = pretty_print()
        assert fn(42) == "42"
        assert fn({"a": 1}) == "{'a': 1}"
        assert fn([1, 2, 3]) == "[1, 2, 3]"

    def test_milabench_pprint_uses_str(self):
        assert milabench_pprint(42) == "42"
        assert milabench_pprint("hello") == "hello"


# ---------------------------------------------------------------------------
# BaseLogger
# ---------------------------------------------------------------------------


class TestBaseLogger:
    def test_start_and_end_are_noop(self):
        logger = BaseLogger()
        logger.start()
        logger.end()

    def test_context_manager_calls_start_and_end(self):
        logger = BaseLogger()
        logger.start = MagicMock()
        logger.end = MagicMock()

        with logger as ctx:
            assert ctx is logger
            logger.start.assert_called_once()

        logger.end.assert_called_once()

    def test_exit_receives_exc_info(self):
        logger = BaseLogger()
        logger.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# TagConsole
# ---------------------------------------------------------------------------


class TestTagConsole:
    def test_ensure_line_adds_newline(self):
        tc = TagConsole("test", 0)
        assert tc._ensure_line("hello") == "hello\n"

    def test_ensure_line_preserves_existing_newline(self):
        tc = TagConsole("test", 0)
        assert tc._ensure_line("hello\n") == "hello\n"

    def test_sprint_includes_header(self):
        tc = TagConsole("mytag", 0)
        result = tc.sprint("part1", "part2")
        assert "part1" in result
        assert "part2" in result
        assert result.endswith("\n")

    def test_spretty_with_string_object(self):
        tc = TagConsole("tag", 1)
        result = tc.spretty("prefix", "string_data")
        assert "prefix" in result
        assert "string_data" in result

    def test_spretty_with_non_string_object(self):
        tc = TagConsole("tag", 2, pretty_print=repr)
        result = tc.spretty("prefix", {"key": "val"})
        assert "prefix" in result
        assert "key" in result

    def test_print_outputs_to_stdout(self, capsys):
        tc = TagConsole("tag", 0)
        tc.print("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_pretty_outputs_to_stdout(self, capsys):
        tc = TagConsole("tag", 0)
        tc.pretty("label", "data")
        captured = capsys.readouterr()
        assert "label" in captured.out

    def test_close_is_noop(self):
        tc = TagConsole("tag", 0)
        tc.close()

    def test_color_cycling(self):
        tc0 = TagConsole("a", 0)
        tc6 = TagConsole("b", 6)
        assert tc0.header is not None
        assert tc6.header is not None


# ---------------------------------------------------------------------------
# TerminalFormatter
# ---------------------------------------------------------------------------


class TestTerminalFormatter:
    def test_console_creates_and_caches(self):
        fmt = TerminalFormatter()
        c1 = fmt.console("tag1")
        c2 = fmt.console("tag1")
        assert c1 is c2
        assert isinstance(c1, TagConsole)

    def test_console_unique_per_tag(self):
        fmt = TerminalFormatter()
        c1 = fmt.console("a")
        c2 = fmt.console("b")
        assert c1 is not c2

    def test_line_stdout_event(self, capsys):
        fmt = TerminalFormatter()
        entry = _entry("line", data="some output\n", pipe="stdout")
        fmt(entry)
        captured = capsys.readouterr()
        assert "some output" in captured.out

    def test_line_stderr_event(self, capsys):
        fmt = TerminalFormatter()
        entry = _entry("line", data="err msg\n", pipe="stderr")
        fmt(entry)
        captured = capsys.readouterr()
        assert "err msg" in captured.out

    def test_line_none_data(self, capsys):
        fmt = TerminalFormatter()
        entry = _entry("line", data=None, pipe="stdout")
        fmt(entry)

    def test_data_event_without_progress(self, capsys):
        fmt = TerminalFormatter()
        entry = _entry("data", data={"rate": 1.5, "loss": 0.3})
        fmt(entry)
        captured = capsys.readouterr()
        assert "data" in captured.out.lower() or "rate" in captured.out

    def test_data_event_with_progress_returns_early(self, capsys):
        fmt = TerminalFormatter()
        entry = _entry("data", data={"progress": [50, 100]})
        result = fmt(entry)
        assert result is None

    def test_start_event(self, capsys):
        fmt = TerminalFormatter()
        entry = _entry("start", data={"command": ["python", "train.py"], "time": time.time()})
        fmt(entry)
        captured = capsys.readouterr()
        assert "start" in captured.out.lower() or "python" in captured.out

    def test_stop_event_sets_early_stop(self):
        fmt = TerminalFormatter()
        assert fmt.early_stop is False
        entry = _entry("stop", data={})
        fmt(entry)
        assert fmt.early_stop is True

    def test_error_event_with_message(self, capsys):
        fmt = TerminalFormatter()
        entry = _entry("error", data={"type": "RuntimeError", "message": "boom"})
        fmt(entry)
        assert entry.tag in fmt.error_happened
        captured = capsys.readouterr()
        assert "boom" in captured.out

    def test_error_event_without_message(self, capsys):
        fmt = TerminalFormatter()
        entry = _entry("error", data={"type": "RuntimeError", "message": ""})
        fmt(entry)
        captured = capsys.readouterr()
        assert "RuntimeError" in captured.out

    def test_end_event_with_error(self, capsys):
        fmt = TerminalFormatter()
        fmt.error_happened.add("bench.0")
        entry = _entry(
            "end",
            data={"command": ["python", "train.py"], "time": time.time(), "return_code": 1},
        )
        fmt(entry)
        captured = capsys.readouterr()
        assert "1" in captured.out or "ERROR" in captured.out

    def test_end_event_success(self, capsys):
        fmt = TerminalFormatter()
        entry = _entry(
            "end",
            data={"command": ["python", "train.py"], "time": time.time(), "return_code": 0},
        )
        fmt(entry)
        captured = capsys.readouterr()
        assert "end" in captured.out.lower() or "python" in captured.out

    def test_end_event_with_early_stop_suppresses_error(self, capsys):
        fmt = TerminalFormatter()
        fmt.early_stop = True
        fmt.error_happened.add("bench.0")
        entry = _entry(
            "end",
            data={"command": ["python", "x.py"], "time": time.time(), "return_code": 1},
        )
        fmt(entry)

    def test_end_event_nonzero_rc_no_error_happened(self, capsys):
        fmt = TerminalFormatter()
        entry = _entry(
            "end",
            data={"command": ["cmd"], "time": time.time(), "return_code": 2},
        )
        fmt(entry)
        captured = capsys.readouterr()
        assert "2" in captured.out

    def test_end_event_rc_zero_but_error_happened(self, capsys):
        fmt = TerminalFormatter()
        fmt.error_happened.add("bench.0")
        entry = _entry(
            "end",
            data={"command": ["cmd"], "time": time.time(), "return_code": 0},
        )
        fmt(entry)
        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    def test_phase_event_is_noop(self, capsys):
        fmt = TerminalFormatter()
        entry = _entry("phase", data={})
        fmt(entry)

    def test_config_event_with_dump(self, capsys):
        fmt = TerminalFormatter(dump_config=True)
        entry = _entry("config", data={"lr": 0.01, "batch_size": 32})
        fmt(entry)
        captured = capsys.readouterr()
        assert "lr" in captured.out or "batch_size" in captured.out

    def test_config_event_skip_system(self, capsys):
        fmt = TerminalFormatter(dump_config=True)
        entry = _entry("config", data={"system": {"gpu": 8}})
        fmt(entry)

    def test_config_event_nested_dict(self, capsys):
        fmt = TerminalFormatter(dump_config=True)
        entry = _entry("config", data={"model": {"hidden_size": 768}})
        fmt(entry)
        captured = capsys.readouterr()
        assert "768" in captured.out or "hidden_size" in captured.out

    def test_config_event_no_dump(self, capsys):
        fmt = TerminalFormatter(dump_config=False)
        entry = _entry("config", data={"lr": 0.01})
        fmt(entry)
        captured = capsys.readouterr()
        assert "lr" not in captured.out

    def test_meta_event_is_noop(self, capsys):
        fmt = TerminalFormatter()
        entry = _entry("meta", data={})
        fmt(entry)

    def test_message_event(self, capsys):
        fmt = TerminalFormatter()
        entry = _entry("message", data={"message": "hello world"})
        fmt(entry)
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_unknown_event_pretty_prints(self, capsys):
        fmt = TerminalFormatter()
        entry = _entry("custom_event", data={"key": "value"})
        fmt(entry)
        captured = capsys.readouterr()
        assert "custom_event" in captured.out


# ---------------------------------------------------------------------------
# BaseReporter
# ---------------------------------------------------------------------------


class TestBaseReporter:
    def test_log_is_noop(self):
        r = BaseReporter(pipe="stdout")
        entry = _entry("data", data={})
        r.log(entry)

    def test_cleanup_closes_file_on_end(self):
        r = BaseReporter(pipe="stdout")
        mock_file = MagicMock()
        r.files["bench.0"] = mock_file

        entry = _entry("end", data={})
        r.cleanup(entry)

        mock_file.__exit__.assert_called_once_with(None, None, None)
        assert "bench.0" not in r.files

    def test_cleanup_no_file_for_tag(self):
        r = BaseReporter(pipe="stdout")
        entry = _entry("end", data={}, tag="other")
        r.cleanup(entry)

    def test_cleanup_ignores_non_end_events(self):
        r = BaseReporter(pipe="stdout")
        mock_file = MagicMock()
        r.files["bench.0"] = mock_file

        entry = _entry("data", data={})
        r.cleanup(entry)

        mock_file.__exit__.assert_not_called()

    def test_call_delegates_to_log_and_cleanup(self):
        r = BaseReporter(pipe="stdout")
        r.log = MagicMock()
        r.cleanup = MagicMock()
        entry = _entry("data", data={})

        r(entry)

        r.log.assert_called_once_with(entry)
        r.cleanup.assert_called_once_with(entry)

    def test_file_creates_directory_and_opens(self, tmp_path):
        pack = FakePack("bench.0")
        logdir = tmp_path / "logs"
        pack._logfile_path = str(logdir)
        pack.logfile = lambda pipe: str(logdir / f"{pipe}.log")

        r = BaseReporter(pipe="stdout")
        entry = _entry("line", data="test", pipe="stdout", pack=pack)

        f = r.file(entry)
        assert f is not None
        assert "bench.0" in r.files
        f.__exit__(None, None, None)

    def test_file_caches(self, tmp_path):
        pack = FakePack("bench.0")
        logdir = tmp_path / "logs"
        pack.logfile = lambda pipe: str(logdir / f"{pipe}.log")

        r = BaseReporter(pipe="stdout")
        entry = _entry("line", data="test", pipe="stdout", pack=pack)

        f1 = r.file(entry)
        f2 = r.file(entry)
        assert f1 is f2
        f1.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# TextReporter
# ---------------------------------------------------------------------------


class TestTextReporter:
    def test_log_writes_line_matching_pipe(self, tmp_path):
        pack = FakePack("bench.0")
        pack.logfile = lambda pipe: str(tmp_path / f"{pipe}.log")

        r = TextReporter(pipe="stdout")
        entry = _entry("line", data="hello world\n", pipe="stdout", pack=pack)
        r(entry)

        end_entry = _entry("end", data={}, pack=pack)
        r(end_entry)

        content = (tmp_path / "stdout.log").read_text()
        assert "hello world" in content

    def test_log_ignores_non_matching_pipe(self, tmp_path):
        pack = FakePack("bench.0")
        pack.logfile = lambda pipe: str(tmp_path / f"{pipe}.log")

        r = TextReporter(pipe="stdout")
        entry = _entry("line", data="err\n", pipe="stderr", pack=pack)
        r(entry)

        assert not (tmp_path / "stdout.log").exists()

    def test_log_ignores_non_line_events(self, tmp_path):
        pack = FakePack("bench.0")
        pack.logfile = lambda pipe: str(tmp_path / f"{pipe}.log")

        r = TextReporter(pipe="stdout")
        entry = _entry("data", data={"rate": 1.0}, pipe="stdout", pack=pack)
        r(entry)

        assert not (tmp_path / "stdout.log").exists()

    def test_close_with_no_open_files(self):
        r = TextReporter(pipe="stdout")
        r.close()


# ---------------------------------------------------------------------------
# DataReporter
# ---------------------------------------------------------------------------


class TestDataReporter:
    def test_init_sets_pipe_to_data(self):
        r = DataReporter()
        assert r.pipe == "data"

    def test_log_writes_json(self, tmp_path):
        pack = FakePack("bench.0")
        pack.logfile = lambda pipe: str(tmp_path / f"{pipe}.log")

        r = DataReporter()
        entry = _entry("data", data={"rate": 1.5}, pack=pack)
        r(entry)

        end_entry = _entry("end", data={}, pack=pack)
        r(end_entry)

        content = (tmp_path / "data.log").read_text()
        lines = content.strip().split("\n")
        assert len(lines) >= 1
        parsed = json.loads(lines[0])
        assert parsed["event"] == "data"

    def test_log_handles_unserializable_data(self, tmp_path):
        pack = FakePack("bench.0")
        pack.logfile = lambda pipe: str(tmp_path / f"{pipe}.log")

        r = DataReporter()

        class Unserializable:
            pass

        entry = _entry("data", data={"obj": Unserializable()}, pack=pack)
        r(entry)

        end_entry = _entry("end", data={}, pack=pack)
        r(end_entry)

        content = (tmp_path / "data.log").read_text()
        assert "#unrepresentable" in content or "Unserializable" in content


# ---------------------------------------------------------------------------
# new_progress_bar
# ---------------------------------------------------------------------------


class TestNewProgressBar:
    def test_creates_progress_with_task(self):
        p = new_progress_bar()
        assert hasattr(p, "_task")


# ---------------------------------------------------------------------------
# find_byte_exponent / formatbyte
# ---------------------------------------------------------------------------


class TestByteFormatting:
    def test_find_byte_exponent_bytes(self):
        name, exp = find_byte_exponent(500)
        assert name == " o"
        assert exp == 1

    def test_find_byte_exponent_kilobytes(self):
        name, exp = find_byte_exponent(2048)
        assert name == "Ko"

    def test_find_byte_exponent_megabytes(self):
        name, exp = find_byte_exponent(2 * 1024 ** 2)
        assert name == "Mo"

    def test_find_byte_exponent_gigabytes(self):
        name, exp = find_byte_exponent(2 * 1024 ** 3)
        assert name == "Go"

    def test_find_byte_exponent_zero(self):
        name, exp = find_byte_exponent(0)
        assert name == " o"

    def test_formatbyte_small(self):
        result = formatbyte(512)
        assert "512" in result
        assert "o" in result

    def test_formatbyte_kilobytes(self):
        result = formatbyte(2048)
        assert "Ko" in result

    def test_formatbyte_gigabytes(self):
        result = formatbyte(2 * 1024 ** 3)
        assert "Go" in result


# ---------------------------------------------------------------------------
# DashFormatter
# ---------------------------------------------------------------------------


class TestDashFormatter:
    @pytest.fixture()
    def dash(self):
        with patch("milabench.log.get_run_count", return_value=0):
            d = DashFormatter()
        return d

    def test_init_defaults(self, dash):
        assert dash.max_rows == 8
        assert dash.prune_delay == 60
        assert dash.current == 0

    @patch("milabench.log.get_run_count", return_value=10)
    def test_get_global_progress_bar_creates_once(self, mock_rc, dash):
        bar1 = dash._get_global_progress_bar()
        bar2 = dash._get_global_progress_bar()
        assert bar1 is bar2

    @patch("milabench.log.get_run_count", return_value=5)
    def test_update_global_increments(self, mock_rc, dash):
        dash._update_global(1)
        assert dash.current == 1
        dash._update_global(2)
        assert dash.current == 3

    @patch("milabench.log.get_run_count", return_value=0)
    def test_update_global_no_total(self, mock_rc, dash):
        dash._update_global(1)
        assert dash.current == 1

    def test_should_prune_old_run(self, dash):
        assert dash.should_prune("bench.0", 120) is True

    def test_should_prune_recent_run_still_active(self, dash):
        dash.benchcount["bench"] = 1
        assert dash.should_prune("bench.0", 5) is False

    def test_should_prune_too_many_rows(self, dash):
        dash.benchcount["bench"] = 0
        for i in range(dash.max_rows + 2):
            dash.rows[f"row_{i}"] = {}
        assert dash.should_prune("bench.0", 10) is True

    def test_should_prune_within_limits(self, dash):
        dash.benchcount["bench"] = 0
        dash.rows["row_0"] = {}
        assert dash.should_prune("bench.0", 10) is False

    def test_should_prune_max_rows_zero(self, dash):
        dash.max_rows = 0
        dash.benchcount["bench"] = 0
        assert dash.should_prune("bench.0", 10) is False

    def test_prune_removes_old_entries(self, dash):
        dash.rows["old.0"] = {"data": "x"}
        dash.endtimes["old.0"] = time.time() - 200
        dash.prune()
        assert "old.0" not in dash.rows

    def test_prune_keeps_recent_entries(self, dash):
        dash.rows["new.0"] = {"data": "x"}
        dash.benchcount["new"] = 1
        dash.endtimes["new.0"] = time.time()
        dash.prune()
        assert "new.0" in dash.rows

    @patch("milabench.log.get_run_count", return_value=0)
    def test_call_dispatches_to_on_method(self, mock_rc, dash):
        dash.on_start = MagicMock()
        entry = _entry("start", data={"command": [], "time": time.time()})
        dash(entry)
        dash.on_start.assert_called_once()

    @patch("milabench.log.get_run_count", return_value=0)
    def test_call_unknown_event_no_error(self, mock_rc, dash):
        entry = _entry("unknown_event", data={})
        dash(entry)

    def test_on_start_increments_benchcount(self, dash):
        entry = _entry("start", data={}, tag="bench.0")
        dash.on_start(entry, {}, {})
        assert dash.benchcount["bench"] == 1

    def test_on_stop_sets_early_stop(self, dash):
        entry = _entry("stop", data={}, tag="bench.0")
        dash.on_stop(entry, {}, {})
        assert dash.early_stop["bench.0"] is True

    @patch("milabench.log.get_run_count", return_value=5)
    def test_on_end_updates_global_and_endtimes(self, mock_rc, dash):
        entry = _entry("end", data={}, tag="bench.0")
        dash.benchcount["bench"] = 1
        dash.on_end(entry, {}, {})
        assert dash.current == 1
        assert "bench.0" in dash.endtimes
        assert dash.benchcount["bench"] == 0

    @patch("milabench.log.get_run_count", return_value=0)
    def test_start_and_end_lifecycle(self, mock_rc, dash):
        dash.live = MagicMock()
        dash.start()
        dash.live.__enter__.assert_called_once()
        dash.end()
        dash.live.__exit__.assert_called_once_with(None, None, None)

    @patch("milabench.log.get_run_count", return_value=0)
    def test_refresh_calls_make_table(self, mock_rc, dash):
        dash.make_table = MagicMock(return_value="table")
        dash.live = MagicMock()
        dash.refresh()
        dash.make_table.assert_called_once()
        dash.live.update.assert_called_once()


# ---------------------------------------------------------------------------
# ShortDashFormatter
# ---------------------------------------------------------------------------


class TestShortDashFormatter:
    @pytest.fixture()
    def sdash(self):
        with patch("milabench.log.get_run_count", return_value=0):
            d = ShortDashFormatter()
            d.live = MagicMock()
        return d

    def test_make_table_empty(self, sdash):
        table = sdash.make_table()
        assert table is not None

    def test_make_table_with_global_row(self, sdash):
        sdash.rows["GLOBAL"] = {"progress": "50%"}
        table = sdash.make_table()
        assert table is not None

    def test_make_table_with_bench_row(self, sdash):
        sdash.rows["bench.0"] = {
            "status": "RUNNING",
            "progress": "25%",
            "rate": "1.50",
            "loss": "0.30",
            "gpu_load": "80%",
            "gpu_mem": "4000/8000 MB",
            "gpu_temp": "65C",
        }
        table = sdash.make_table()
        assert table is not None

    def test_on_data_progress_early_stop(self, sdash):
        entry = _entry("data", data={"progress": [50, 100], "task": "early_stop"})
        row = {}
        sdash.on_data(entry, {"progress": [50, 100], "task": "early_stop"}, row)
        assert row["progress"] == "50%"

    def test_on_data_progress_done(self, sdash):
        entry = _entry("data", data={"progress": [100, 100], "task": "early_stop"})
        row = {}
        sdash.on_data(entry, {"progress": [100, 100], "task": "early_stop"}, row)
        assert row["progress"] == "DONE"

    def test_on_data_progress_zero_total(self, sdash):
        entry = _entry("data", data={"progress": [0, 0], "task": "early_stop"})
        row = {}
        sdash.on_data(entry, {"progress": [0, 0], "task": "early_stop"}, row)
        assert "progress" not in row

    def test_on_data_gpudata(self, sdash):
        entry = _entry("data", data={})
        gpudata = {"0": {"load": 0.85, "memory": [4000, 8000], "temperature": 72}}
        row = {}
        sdash.on_data(entry, {"gpudata": gpudata}, row)
        assert row["gpu_load"] == "85%"
        assert "4000/8000 MB" in row["gpu_mem"]
        assert "72C" in row["gpu_temp"]

    def test_on_data_rate_train(self, sdash):
        entry = _entry("data", data={})
        row = {}
        sdash.on_data(entry, {"rate": 3.14159, "task": "train"}, row)
        assert row["rate"] == "3.14"

    def test_on_data_rate_non_train(self, sdash):
        entry = _entry("data", data={})
        row = {}
        sdash.on_data(entry, {"rate": 3.14, "task": "eval"}, row)
        assert "rate" not in row

    def test_on_data_loss_train(self, sdash):
        entry = _entry("data", data={})
        row = {}
        sdash.on_data(entry, {"loss": 0.567, "task": "train"}, row)
        assert row["loss"] == "0.57"

    def test_on_data_loss_non_train(self, sdash):
        entry = _entry("data", data={})
        row = {}
        sdash.on_data(entry, {"loss": 0.5, "task": "eval"}, row)
        assert "loss" not in row

    def test_on_start(self, sdash):
        entry = _entry("start", data={}, tag="bench.0")
        row = {}
        sdash.on_start(entry, {}, row)
        assert "RUNNING" in str(row["status"])
        assert sdash.benchcount["bench"] == 1

    def test_on_error(self, sdash):
        entry = _entry("error", data={})
        row = {}
        sdash.on_error(entry, {}, row)
        assert "ERROR" in str(row["status"])

    @patch("milabench.log.get_run_count", return_value=5)
    def test_on_end_success(self, mock_rc, sdash):
        entry = _entry("end", data={"return_code": 0}, tag="bench.0")
        sdash.benchcount["bench"] = 1
        row = {}
        sdash.on_end(entry, {"return_code": 0}, row)
        assert "COMPLETED" in str(row["status"])

    @patch("milabench.log.get_run_count", return_value=5)
    def test_on_end_with_early_stop(self, mock_rc, sdash):
        entry = _entry("end", data={"return_code": -9}, tag="bench.0")
        sdash.early_stop["bench.0"] = True
        sdash.benchcount["bench"] = 1
        row = {}
        sdash.on_end(entry, {"return_code": -9}, row)
        assert "COMPLETED" in str(row["status"])

    @patch("milabench.log.get_run_count", return_value=5)
    def test_on_end_failure(self, mock_rc, sdash):
        entry = _entry("end", data={"return_code": 1}, tag="bench.0")
        sdash.benchcount["bench"] = 1
        row = {}
        sdash.on_end(entry, {"return_code": 1}, row)
        assert "FAIL:1" in str(row["status"])

    def test_on_data_no_matching_keys(self, sdash):
        entry = _entry("data", data={})
        row = {}
        sdash.on_data(entry, {"other_key": 42}, row)


# ---------------------------------------------------------------------------
# LongDashFormatter
# ---------------------------------------------------------------------------


class TestLongDashFormatter:
    @pytest.fixture()
    def ldash(self):
        with patch("milabench.log.get_run_count", return_value=0):
            d = LongDashFormatter()
            d.live = MagicMock()
        return d

    def test_make_table_empty(self, ldash):
        result = ldash.make_table()
        assert result is not None

    def test_make_table_with_progress(self, ldash):
        ldash.rows["bench.0"] = {"progress": new_progress_bar()}
        result = ldash.make_table()
        assert result is not None

    def test_make_table_with_values(self, ldash):
        ldash.rows["bench.0"] = {"rate": "1.50", "loss": "0.30"}
        result = ldash.make_table()
        assert result is not None

    def test_on_data_progress_creates_bar(self, ldash):
        entry = _entry("data", data={})
        row = {}
        ldash.on_data(entry, {"progress": [10, 100], "task": "early_stop"}, row)
        assert "progress" in row

    def test_on_data_progress_updates_existing_bar(self, ldash):
        entry = _entry("data", data={})
        row = {"progress": new_progress_bar()}
        ldash.on_data(entry, {"progress": [50, 100], "task": "early_stop"}, row)
        assert "progress" in row

    def test_on_data_progress_non_early_stop_ignored(self, ldash):
        entry = _entry("data", data={})
        row = {}
        ldash.on_data(entry, {"progress": [10, 100], "task": "train"}, row)
        assert "progress" not in row

    def test_on_data_gpudata(self, ldash):
        entry = _entry("data", data={})
        gpudata = {"0": {"load": 0.9, "memory": [6000, 8000], "temperature": 80}}
        row = {}
        ldash.on_data(entry, {"gpudata": gpudata}, row)
        assert "gpu:0" in row
        assert "90" in row["gpu:0"]

    def test_on_data_iodata(self, ldash):
        entry = _entry("data", data={})
        iodata = {
            "read_time": 100,
            "write_time": 200,
            "read_count": 50,
            "write_count": 30,
            "busy_time": 150,
        }
        row = {}
        ldash.on_data(entry, {"iodata": iodata}, row)
        assert "iodata" in row
        assert "rt=100" in row["iodata"]
        assert "wt=200" in row["iodata"]

    def test_on_data_process(self, ldash):
        entry = _entry("data", data={})
        process = {
            "pid": "1234",
            "memory": [4e9, 16e9],
            "load": 75.0,
            "read_bytes": 1024,
            "write_bytes": 2048,
            "read_chars": 4096,
            "write_chars": 8192,
        }
        row = {}
        ldash.on_data(entry, {"process": process}, row)
        assert "cpu.1234" in row
        assert "physical.io.1234" in row
        assert "virtual.io.1234" in row

    def test_on_data_cpudata(self, ldash):
        entry = _entry("data", data={})
        cpudata = {"memory": [8e9, 32e9], "load": 45.0}
        row = {}
        ldash.on_data(entry, {"cpudata": cpudata}, row)
        assert "cpudata" in row
        assert "45" in row["cpudata"]

    def test_on_data_netdata(self, ldash):
        entry = _entry("data", data={})
        netdata = {"bytes_sent": 1000, "bytes_recv": 2000}
        row = {}
        ldash.on_data(entry, {"netdata": netdata}, row)
        assert "netdata" in row
        assert "s=1000" in row["netdata"]
        assert "r=2000" in row["netdata"]

    def test_on_data_generic_with_time(self, ldash):
        entry = _entry("data", data={})
        row = {}
        ldash.on_data(
            entry,
            {"rate": 5.0, "task": "train", "units": "samples/s", "time": ldash.created_time + 10},
            row,
        )
        assert "train rate" in row
        assert "time" in row["train rate"]

    def test_on_data_generic_without_time(self, ldash):
        entry = _entry("data", data={})
        row = {}
        ldash.on_data(entry, {"rate": 5.0, "task": "train", "units": "samples/s"}, row)
        assert "train rate" in row

    def test_newlines_method(self, ldash):
        lines = ldash.newlines("train", "items/s", "", {"rate": 1.5})
        assert "train rate" in lines

    def test_newlines_method_with_time(self, ldash):
        lines = ldash.newlines("eval", "items/s", "time 10 s", {"acc": 0.95, "loss": 0.1})
        assert "eval acc" in lines
        assert "time 10 s" in lines["eval acc"]

    def test_time_method(self, ldash):
        t = ldash.created_time + 42
        result = ldash.time(t)
        assert result == 42

    def test_unit_method(self, ldash):
        assert ldash.unit("rate", "items/s") == "items/s"

    def test_format_numeric(self, ldash):
        assert ldash.format("rate", 3.14159) == "3.14"

    def test_format_non_numeric(self, ldash):
        assert ldash.format("status", "running") == "running"

    def test_format_none(self, ldash):
        result = ldash.format("key", None)
        assert result == "None"
