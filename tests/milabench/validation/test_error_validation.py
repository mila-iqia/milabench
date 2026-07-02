import pytest
from collections import defaultdict
from dataclasses import dataclass, field

from milabench.validation.error import (
    PackError,
    ParsedTraceback,
    _extract_traceback,
    Layer,
    split_on_4plus_spaces,
)
from milabench.validation.validation import Summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakePack:
    tag: str


class FakeEntry:
    """Lightweight stand-in for BenchLogEntry."""

    def __init__(self, *, tag, event, data=None, pipe=None):
        self.tag = tag
        self.event = event
        self.data = data
        self.pipe = pipe
        self.pack = FakePack(tag)


def _make_layer(**kw):
    return Layer(**kw)


# ---------------------------------------------------------------------------
# PackError
# ---------------------------------------------------------------------------

class TestPackError:
    def test_defaults(self):
        pe = PackError()
        assert pe.stderr == []
        assert pe.code == []
        assert pe.message is None
        assert pe.early_stop is False
        assert pe.trace is None

    def test_custom_values(self):
        pe = PackError(
            stderr=["err"],
            code=[1],
            message="RuntimeError: boom",
            early_stop=True,
            trace="Traceback ...",
        )
        assert pe.stderr == ["err"]
        assert pe.code == [1]
        assert pe.message == "RuntimeError: boom"
        assert pe.early_stop is True
        assert pe.trace == "Traceback ..."


# ---------------------------------------------------------------------------
# ParsedTraceback
# ---------------------------------------------------------------------------

class TestParsedTraceback:
    def test_find_raise_found(self):
        lines = [
            'File "foo.py", line 1',
            "    raise RuntimeError(",
            'RuntimeError: oops',
        ]
        tb = ParsedTraceback(lines)
        idx, name = tb.find_raise()
        assert idx == 1
        assert name == "RuntimeError"

    def test_find_raise_not_found(self):
        tb = ParsedTraceback(["no raise here\n"])
        idx, name = tb.find_raise()
        assert idx is None
        assert name is None

    def test_raised_exception_with_raise(self):
        lines = [
            'File "foo.py", line 1',
            "    raise ValueError(",
            "ValueError: bad value",
        ]
        tb = ParsedTraceback(lines)
        assert tb.raised_exception() == "ValueError: bad value"

    def test_raised_exception_without_raise(self):
        lines = ["line one", "line two", "final line"]
        tb = ParsedTraceback(lines)
        assert tb.raised_exception() == "final line"

    def test_raised_exception_raise_at_last_line(self):
        """When raise is the only element, min(idx+1, len) overshoots (source bug)."""
        lines = ["    raise KeyError("]
        tb = ParsedTraceback(lines)
        with pytest.raises(IndexError):
            tb.raised_exception()

    def test_append_line_normal(self):
        tb = ParsedTraceback(["first"])
        tb.append_line("second\n")
        assert tb.lines == ["first", "second"]

    def test_append_line_caret_merge(self):
        """Two consecutive caret/space-only lines get merged."""
        tb = ParsedTraceback(["^^ "])
        tb.append_line("  ^^")
        assert len(tb.lines) == 1
        assert tb.lines[0] == "^^ " + "  ^^"


# ---------------------------------------------------------------------------
# split_on_4plus_spaces  (currently identity)
# ---------------------------------------------------------------------------

class TestSplitOn4PlusSpaces:
    def test_identity(self):
        assert split_on_4plus_spaces("hello") == "hello"
        assert split_on_4plus_spaces("") == ""


# ---------------------------------------------------------------------------
# _extract_traceback
# ---------------------------------------------------------------------------

class TestExtractTraceback:
    def test_empty_input(self):
        assert _extract_traceback([], is_install=False) == []

    def test_single_traceback(self):
        lines = [
            "Traceback (most recent call last):\n",
            '  File "foo.py", line 1\n',
            "RuntimeError: boom\n",
        ]
        result = _extract_traceback(lines, is_install=False)
        assert len(result) == 1
        assert "RuntimeError: boom" in result[0].lines[-1]

    def test_multiple_tracebacks_keeps_all(self):
        lines = [
            "Traceback (most recent call last):\n",
            "  first error\n",
            "Traceback (most recent call last):\n",
            "  second error\n",
        ]
        result = _extract_traceback(lines, is_install=False)
        assert len(result) == 2

    def test_during_handling_stops_parsing(self):
        """'During handling of the above exception' causes a break."""
        lines = [
            "Traceback (most recent call last):\n",
            "  first\n",
            "During handling of the above exception\n",
            "Traceback (most recent call last):\n",
            "  should not appear\n",
        ]
        result = _extract_traceback(lines, is_install=False)
        assert len(result) == 1

    def test_exception_in_thread_stops_parsing(self):
        """'Exception in thread Thread' causes a break."""
        lines = [
            "Traceback (most recent call last):\n",
            "  first\n",
            "Exception in thread Thread-1\n",
            "Traceback (most recent call last):\n",
            "  should not appear\n",
        ]
        result = _extract_traceback(lines, is_install=False)
        assert len(result) == 1

    def test_pip_error_during_install(self):
        """During install, lines with ERROR start a pip-error traceback."""
        lines = [
            "ERROR: some pip failure\n",
            "  detail line\n",
        ]
        result = _extract_traceback(lines, is_install=True)
        assert len(result) == 1
        assert "PipInstallError" in result[0].lines[-1]

    def test_pip_error_not_during_install(self):
        """Outside install, ERROR lines are ignored."""
        lines = [
            "ERROR: some pip failure\n",
            "  detail line\n",
        ]
        result = _extract_traceback(lines, is_install=False)
        assert len(result) == 0

    def test_empty_line_becomes_newline(self):
        """Empty strings in lines become '\\n' internally."""
        lines = [
            "Traceback (most recent call last):\n",
            "",
            "  body\n",
            "RuntimeError: x\n",
        ]
        result = _extract_traceback(lines, is_install=False)
        assert len(result) == 1

    def test_lines_without_newline_accumulate(self):
        """Lines without \\n get accumulated until a \\n arrives."""
        lines = [
            "Trace",
            "back (most recent call last):\n",
            "  err\n",
            "Error: x\n",
        ]
        result = _extract_traceback(lines, is_install=False)
        assert len(result) == 1

    def test_pip_error_not_duplicated(self):
        """A second ERROR line while already parsing_pip_error doesn't start a new traceback."""
        lines = [
            "ERROR: first pip error\n",
            "ERROR: second pip error\n",
            "  detail\n",
        ]
        result = _extract_traceback(lines, is_install=True)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Layer – event handlers
# ---------------------------------------------------------------------------

class TestLayerEvents:
    @pytest.fixture
    def layer(self):
        return _make_layer()

    def test_on_stop(self, layer):
        entry = FakeEntry(tag="bench.0", event="stop")
        layer.on_stop(entry)
        assert layer.errors["bench.0"].early_stop is True

    def test_on_config(self, layer):
        layer.is_prepare = True
        layer.is_install = True
        entry = FakeEntry(tag="bench.0", event="config")
        layer.on_config(entry)
        assert layer.is_prepare is False
        assert layer.is_install is False

    def test_on_start_pip(self, layer):
        entry = FakeEntry(tag="bench.0", event="start", data={"command": ["pip", "install"]})
        layer.on_start(entry)
        assert layer.is_install is True

    def test_on_start_non_pip(self, layer):
        entry = FakeEntry(tag="bench.0", event="start", data={"command": ["python", "train.py"]})
        layer.on_start(entry)
        assert layer.is_install is False

    def test_on_line_stderr(self, layer):
        entry = FakeEntry(tag="bench.0", event="line", data="some error text", pipe="stderr")
        layer.on_line(entry)
        assert "some error text" in layer.errors["bench.0"].stderr

    def test_on_line_stdout_ignored(self, layer):
        entry = FakeEntry(tag="bench.0", event="line", data="stdout text", pipe="stdout")
        layer.on_line(entry)
        assert layer.errors["bench.0"].stderr == []

    def test_on_error(self, layer):
        entry = FakeEntry(
            tag="bench.0",
            event="error",
            data={"type": "RuntimeError", "message": "boom", "trace": "tb"},
        )
        layer.on_error(entry)
        err = layer.errors["bench.0"]
        assert err.code == [1]
        assert err.message == "RuntimeError: boom"
        assert err.trace == "tb"

    def test_on_error_no_trace(self, layer):
        entry = FakeEntry(
            tag="bench.0",
            event="error",
            data={"type": "ValueError", "message": "bad"},
        )
        layer.on_error(entry)
        assert layer.errors["bench.0"].trace is None

    def test_on_end(self, layer):
        entry = FakeEntry(tag="bench.0", event="end", data={"return_code": 42})
        layer.on_end(entry)
        assert layer.errors["bench.0"].code == [42]


# ---------------------------------------------------------------------------
# Layer.report_exceptions
# ---------------------------------------------------------------------------

class TestReportExceptions:
    def test_no_exceptions_found(self):
        layer = _make_layer()
        error = PackError(stderr=[], code=[1])
        summary = Summary()
        summary.newsection("test")
        layer.report_exceptions(summary, error, short=False)
        summary.endsection()
        output = []
        summary._show(summary.root.body, 0, output)
        text = "\n".join(output)
        assert "No traceback info" in text

    def test_with_trace_string(self):
        layer = _make_layer()
        error = PackError(
            code=[1],
            trace="Traceback (most recent call last):\n  raise RuntimeError(\nRuntimeError: fail",
        )
        summary = Summary()
        summary.newsection("test")
        layer.report_exceptions(summary, error, short=False)
        summary.endsection()
        output = []
        summary._show(summary.root.body, 0, output)
        text = "\n".join(output)
        assert "1 exceptions found" in text

    def test_short_mode_no_lines(self):
        layer = _make_layer()
        error = PackError(
            code=[1],
            trace="Traceback:\n  raise RuntimeError(\nRuntimeError: x",
        )
        summary = Summary()
        summary.newsection("test")
        layer.report_exceptions(summary, error, short=True)
        summary.endsection()
        output = []
        summary._show(summary.root.body, 0, output)
        text = "\n".join(output)
        assert "1 exceptions found" in text
        assert "|" not in text

    def test_from_stderr(self):
        layer = _make_layer()
        stderr = [
            "Traceback (most recent call last):\n",
            '  File "x.py", line 1\n',
            "KeyError: missing\n",
        ]
        error = PackError(stderr=stderr, code=[1])
        summary = Summary()
        summary.newsection("test")
        layer.report_exceptions(summary, error, short=False)
        summary.endsection()
        output = []
        summary._show(summary.root.body, 0, output)
        text = "\n".join(output)
        assert "1 exceptions found" in text


# ---------------------------------------------------------------------------
# Layer.group_errors
# ---------------------------------------------------------------------------

class TestGroupErrors:
    def test_groups_by_bench_name(self):
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(code=[0])
        layer.errors["bench.1"] = PackError(code=[1])
        groups = layer.group_errors()
        assert "bench" in groups
        assert groups["bench"].total == 2

    def test_unpartitioned_tag(self):
        """Tag without '.' uses the whole tag as name (line 193-194)."""
        layer = _make_layer()
        layer.errors["singlebench"] = PackError(code=[1])
        groups = layer.group_errors()
        assert "singlebench" in groups
        assert groups["singlebench"].total == 1

    def test_success_counting(self):
        """Return code sum == 0 counts as success (lines 202-204)."""
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(code=[0])
        groups = layer.group_errors()
        assert groups["bench"].success == 1
        assert groups["bench"].failures == 0

    def test_failure_counting(self):
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(code=[1])
        groups = layer.group_errors()
        assert groups["bench"].failures == 1

    def test_early_stop_counting(self):
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(code=[1], early_stop=True)
        groups = layer.group_errors()
        assert groups["bench"].early_stopped == 1
        assert groups["bench"].failures == 0

    def test_exceptions_from_trace(self):
        """When error.trace is set, uses ParsedTraceback on trace (line 211-212)."""
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(
            code=[1],
            trace="Traceback:\n  raise RuntimeError(\nRuntimeError: x",
        )
        groups = layer.group_errors()
        assert len(groups["bench"].exceptions) > 0

    def test_exceptions_from_stderr(self):
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(
            code=[1],
            stderr=[
                "Traceback (most recent call last):\n",
                "  File 'x.py', line 1\n",
                "KeyError: k\n",
            ],
        )
        groups = layer.group_errors()
        assert len(groups["bench"].exceptions) > 0


# ---------------------------------------------------------------------------
# Layer.display_grouped
# ---------------------------------------------------------------------------

class TestDisplayGrouped:
    def test_no_errors(self):
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(code=[0])
        summary = Summary()
        failures = layer.display_grouped(summary)
        assert failures == 0

    def test_failures_returned(self):
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(
            code=[1],
            trace="Traceback:\n  raise RuntimeError(\nRuntimeError: x",
        )
        summary = Summary()
        failures = layer.display_grouped(summary)
        assert failures == 1

    def test_early_stopped_section(self):
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(code=[1], early_stop=True)
        summary = Summary()
        layer.display_grouped(summary)
        output = []
        summary._show(summary.root.body, 0, output)
        text = "\n".join(output)
        assert "bench" in text

    def test_no_exception_message(self):
        """Failure with no traceback info shows 'No exception were found'."""
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(code=[1], stderr=[])
        summary = Summary()
        layer.display_grouped(summary)
        output = []
        summary._show(summary.root.body, 0, output)
        text = "\n".join(output)
        assert "No exception were found" in text


# ---------------------------------------------------------------------------
# Layer.display_extended  (lines 251-275)
# ---------------------------------------------------------------------------

class TestDisplayExtended:
    def test_success_skipped(self):
        """Entries with code sum == 0 are skipped (line 260-262)."""
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(code=[0])
        summary = Summary()
        failures = layer.display_extended(summary)
        assert failures == 0

    def test_early_stop_not_counted_as_failure(self):
        """Early-stopped entries print 'early stopped' and don't increment failures (lines 265-269)."""
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(code=[1], early_stop=True)
        summary = Summary()
        failures = layer.display_extended(summary)
        assert failures == 0
        output = []
        summary._show(summary.root.body, 0, output)
        text = "\n".join(output)
        assert "early stopped" in text

    def test_failure_with_error_codes(self):
        """Non-early-stop failure prints error codes (line 271)."""
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(
            code=[1, 2],
            stderr=[
                "Traceback (most recent call last):\n",
                "RuntimeError: x\n",
            ],
        )
        summary = Summary()
        failures = layer.display_extended(summary)
        assert failures == 1
        output = []
        summary._show(summary.root.body, 0, output)
        text = "\n".join(output)
        assert "Error codes = 1, 2" in text

    def test_sorted_output(self):
        """Errors are reported in sorted tag order (line 255)."""
        layer = _make_layer()
        layer.errors["zebra.0"] = PackError(code=[1])
        layer.errors["alpha.0"] = PackError(code=[1])
        summary = Summary()
        layer.display_extended(summary)
        output = []
        summary._show(summary.root.body, 0, output)
        text = "\n".join(output)
        alpha_pos = text.index("alpha")
        zebra_pos = text.index("zebra")
        assert alpha_pos < zebra_pos

    def test_failure_with_no_traceback(self):
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(code=[1], stderr=[])
        summary = Summary()
        failures = layer.display_extended(summary)
        assert failures == 1
        output = []
        summary._show(summary.root.body, 0, output)
        text = "\n".join(output)
        assert "No traceback info" in text


# ---------------------------------------------------------------------------
# Layer.report  (lines 277-289)
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_sets_error_code(self):
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(code=[0])
        summary = Summary()
        failures = layer.report(summary)
        assert failures == 0
        assert layer.error_code == 0

    def test_report_with_failures(self):
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(code=[1])
        summary = Summary()
        failures = layer.report(summary)
        assert failures == 1
        assert layer.error_code == 1

    def test_report_github_issues_no_failures(self):
        """github_issues=True but no failures should not crash (line 281)."""
        layer = _make_layer()
        layer.errors["bench.0"] = PackError(code=[0])
        summary = Summary()
        failures = layer.report(summary, github_issues=True)
        assert failures == 0


# ---------------------------------------------------------------------------
# Integration: full event flow
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_lifecycle_success(self):
        layer = _make_layer()
        layer.on_config(FakeEntry(tag="bench.0", event="config"))
        layer.on_start(FakeEntry(tag="bench.0", event="start", data={"command": ["python"]}))
        layer.on_line(FakeEntry(tag="bench.0", event="line", data="info", pipe="stdout"))
        layer.on_end(FakeEntry(tag="bench.0", event="end", data={"return_code": 0}))

        summary = Summary()
        assert layer.report(summary) == 0

    def test_full_lifecycle_failure(self):
        layer = _make_layer()
        layer.on_start(FakeEntry(tag="bench.0", event="start", data={"command": ["python"]}))
        layer.on_line(FakeEntry(
            tag="bench.0", event="line",
            data="Traceback (most recent call last):\n", pipe="stderr",
        ))
        layer.on_line(FakeEntry(
            tag="bench.0", event="line",
            data="RuntimeError: crash\n", pipe="stderr",
        ))
        layer.on_error(FakeEntry(
            tag="bench.0", event="error",
            data={"type": "RuntimeError", "message": "crash", "trace": None},
        ))
        layer.on_end(FakeEntry(tag="bench.0", event="end", data={"return_code": 1}))

        summary = Summary()
        failures = layer.report(summary)
        assert failures == 1

    def test_multiple_benches(self):
        layer = _make_layer()
        for i in range(3):
            tag = f"bench.{i}"
            layer.on_end(FakeEntry(tag=tag, event="end", data={"return_code": i}))

        groups = layer.group_errors()
        assert groups["bench"].total == 3
        assert groups["bench"].success == 1

    def test_pip_install_flow(self):
        layer = _make_layer()
        layer.on_start(FakeEntry(tag="pkg.0", event="start", data={"command": ["pip", "install"]}))
        layer.on_line(FakeEntry(
            tag="pkg.0", event="line",
            data="ERROR: could not install\n", pipe="stderr",
        ))
        layer.on_line(FakeEntry(
            tag="pkg.0", event="line",
            data="  some detail\n", pipe="stderr",
        ))
        layer.on_end(FakeEntry(tag="pkg.0", event="end", data={"return_code": 1}))

        summary = Summary()
        failures = layer.display_extended(summary)
        assert failures == 1
