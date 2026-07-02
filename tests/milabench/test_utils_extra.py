"""Comprehensive tests for milabench.utils — targeting untested functions.

Existing tests in tests/test_utils.py cover enumerate_rank and select_nodes,
so this file focuses on the remaining functions and classes.
"""

import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from milabench.utils import (
    MISSING,
    MultiLogger,
    Named,
    assemble_options,
    available_layers,
    blabla,
    deprecated,
    error_guard,
    make_constraints_file,
    multilogger,
    relativize,
    validation_layers,
)
from milabench.validation.validation import Summary


# ---------------------------------------------------------------------------
# deprecated decorator
# ---------------------------------------------------------------------------

class TestDeprecated:
    def test_emits_deprecation_warning(self):
        @deprecated
        def old_func():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()

        assert result == 42
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "old_func" in str(w[0].message)

    def test_preserves_return_value(self):
        @deprecated
        def add(a, b):
            return a + b

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            assert add(3, 4) == 7

    def test_preserves_function_name(self):
        @deprecated
        def my_function():
            pass

        assert my_function.__name__ == "my_function"


# ---------------------------------------------------------------------------
# Named & MISSING sentinel
# ---------------------------------------------------------------------------

class TestNamed:
    def test_repr(self):
        obj = Named("hello")
        assert repr(obj) == "hello"

    def test_missing_sentinel(self):
        assert repr(MISSING) == "MISSING"


# ---------------------------------------------------------------------------
# blabla — random syllable generator
# ---------------------------------------------------------------------------

class TestBlabla:
    def test_default_length(self):
        word = blabla()
        assert len(word) == 8  # 4 syllables × 2 chars

    def test_custom_length(self):
        word = blabla(n=2)
        assert len(word) == 4

    def test_zero_length(self):
        assert blabla(n=0) == ""

    def test_only_valid_chars(self):
        vowels = set("aeiou")
        consonants = set("bdfgjklmnprstvz")
        valid = vowels | consonants
        word = blabla(n=10)
        assert all(c in valid for c in word)


# ---------------------------------------------------------------------------
# error_guard decorator
# ---------------------------------------------------------------------------

class TestErrorGuard:
    def test_returns_normal_value_on_success(self):
        @error_guard(default_return=-1)
        def good():
            return 42

        assert good() == 42

    def test_returns_static_default_on_exception(self):
        @error_guard(default_return=-1)
        def bad():
            raise RuntimeError("boom")

        assert bad() == -1

    def test_returns_callable_default_on_exception(self):
        @error_guard(default_return=lambda x, y: x + y)
        def bad(x, y):
            raise ValueError("oops")

        assert bad(3, 4) == 7

    def test_prints_to_stderr_on_exception(self, capsys):
        @error_guard(default_return=0)
        def bad():
            raise RuntimeError("test error")

        bad()
        captured = capsys.readouterr()
        assert "non-fatal error" in captured.err.lower()

    def test_preserves_function_name(self):
        @error_guard(default_return=None)
        def my_func():
            pass

        assert my_func.__name__ == "my_func"

    def test_passes_kwargs_to_callable_default(self):
        @error_guard(default_return=lambda *, key=None: key)
        def bad(*, key=None):
            raise RuntimeError()

        assert bad(key="value") == "value"


# ---------------------------------------------------------------------------
# assemble_options — dict/list → CLI args
# ---------------------------------------------------------------------------

class TestAssembleOptions:
    def test_list_passthrough(self):
        opts = ["--foo", "bar"]
        assert assemble_options(opts) == ["--foo", "bar"]

    def test_empty_list(self):
        assert assemble_options([]) == []

    def test_none_values_skipped(self):
        opts = {"--keep": "yes", "--skip": None}
        result = assemble_options(opts)
        assert "--keep" in result
        assert "--skip" not in result

    def test_true_flag_with_dash(self):
        result = assemble_options({"--verbose": True})
        assert result == ["--verbose"]

    def test_true_positional_without_dash(self):
        result = assemble_options({"train": True})
        assert result == ["train"]

    def test_positional_comes_first(self):
        result = assemble_options({"train": True, "--lr": "0.01"})
        assert result[0] == "train"
        assert "--lr" in result
        assert "0.01" in result

    def test_double_dash_extends(self):
        result = assemble_options({"--": ["extra1", "extra2"]})
        assert "extra1" in result
        assert "extra2" in result

    def test_false_raises(self):
        with pytest.raises(ValueError, match="null"):
            assemble_options({"--bad": False})

    def test_string_value(self):
        result = assemble_options({"--lr": "0.001"})
        assert result == ["--lr", "0.001"]

    def test_numeric_value(self):
        result = assemble_options({"--epochs": 10})
        assert result == ["--epochs", "10"]

    def test_list_value_joined(self):
        result = assemble_options({"--gpus": [0, 1, 2]})
        assert result == ["--gpus", "0,1,2"]

    def test_empty_dict(self):
        assert assemble_options({}) == []


# ---------------------------------------------------------------------------
# relativize
# ---------------------------------------------------------------------------

class TestRelativize:
    def test_absolute_path_gets_relativized(self):
        result = relativize("/home/user/project/foo.txt", "/home/user/project")
        assert str(result) == "foo.txt"

    def test_relative_path_stays_relative(self):
        result = relativize("foo/bar.txt", "/home/user/project")
        assert str(result) == "foo/bar.txt"


# ---------------------------------------------------------------------------
# make_constraints_file
# ---------------------------------------------------------------------------

class TestMakeConstraintsFile:
    def test_creates_file_with_constraints(self, tmp_path):
        working_dir = str(tmp_path)
        constraints = [str(tmp_path / "c1.txt"), str(tmp_path / "c2.txt")]
        result = make_constraints_file(
            ".pin/constraints.txt", constraints, working_dir
        )
        assert len(result) == 1
        content = result[0].read_text()
        assert "-c ../c1.txt" in content
        assert "-c ../c2.txt" in content

    def test_creates_file_with_requirements(self, tmp_path):
        working_dir = str(tmp_path)
        constraints = [str(tmp_path / "c.txt")]
        requirements = [str(tmp_path / "req.txt")]
        result = make_constraints_file(
            ".pin/constraints.txt", constraints, working_dir, requirements
        )
        content = result[0].read_text()
        assert "-c" in content
        assert "-r ../req.txt" in content

    def test_empty_constraints_returns_empty_tuple(self, tmp_path):
        result = make_constraints_file(
            ".pin/constraints.txt", [], str(tmp_path)
        )
        assert result == ()

    def test_none_constraints_returns_empty_tuple(self, tmp_path):
        result = make_constraints_file(
            ".pin/constraints.txt", None, str(tmp_path)
        )
        assert result == ()


# ---------------------------------------------------------------------------
# validation_layers & available_layers
# ---------------------------------------------------------------------------

class TestValidationLayers:
    def test_available_layers_returns_known_names(self):
        names = list(available_layers())
        assert len(names) > 0
        assert "error" in names

    def test_instantiates_known_layers(self):
        layers = validation_layers("error")
        assert len(layers) == 1

    def test_unknown_layer_raises(self):
        with pytest.raises(RuntimeError, match="does not exist"):
            validation_layers("nonexistent_layer_xyz")

    def test_multiple_layers(self):
        names = list(available_layers())
        if len(names) >= 2:
            layers = validation_layers(names[0], names[1])
            assert len(layers) == 2

    def test_empty_call_returns_empty(self):
        layers = validation_layers()
        assert layers == []


# ---------------------------------------------------------------------------
# MultiLogger
# ---------------------------------------------------------------------------

class TestMultiLogger:
    def test_calls_all_functions(self):
        calls = []

        def f1(*args, **kw):
            calls.append(("f1", args, kw))

        def f2(*args, **kw):
            calls.append(("f2", args, kw))

        ml = MultiLogger({"a": f1, "b": f2})
        ml("hello", key="val")

        assert len(calls) == 2
        assert calls[0] == ("f1", ("hello",), {"key": "val"})
        assert calls[1] == ("f2", ("hello",), {"key": "val"})

    def test_exception_does_not_stop_others(self, capsys):
        calls = []

        def bad(*a, **kw):
            raise RuntimeError("fail")

        def good(*a, **kw):
            calls.append("good")

        ml = MultiLogger({"a": bad, "b": good})
        ml("data")

        assert "good" in calls
        captured = capsys.readouterr()
        assert "Error happened in logger" in captured.err

    def test_stop_on_exception_propagates(self):
        def bad(*a, **kw):
            raise RuntimeError("fatal")

        ml = MultiLogger({"a": bad}, stop_on_exception=True)
        with pytest.raises(RuntimeError, match="fatal"):
            ml("data")

    def test_result_combines_error_codes(self):
        f1 = MagicMock()
        f1.error_code = 0b01
        f2 = MagicMock()
        f2.error_code = 0b10

        ml = MultiLogger({"a": f1, "b": f2})
        assert ml.result() == 0b11

    def test_result_ignores_none_error_codes(self):
        f1 = MagicMock()
        f1.error_code = None

        ml = MultiLogger({"a": f1})
        assert ml.result() == 0

    def test_result_no_error_code_attr(self):
        def plain():
            pass

        ml = MultiLogger({"a": plain})
        assert ml.result() == 0

    def test_report_delegates_to_layers(self):
        layer = MagicMock()
        layer.report = MagicMock()
        ml = MultiLogger({"a": layer})
        ml.report(foo="bar")
        layer.report.assert_called_once()


# ---------------------------------------------------------------------------
# multilogger context manager
# ---------------------------------------------------------------------------

class _DummyLayer:
    """Minimal context-manager layer for testing multilogger.

    Mimics ValidationLayer: __enter__ returns self, __call__ records invocations.
    """

    def __init__(self, name="dummy"):
        self._name = name
        self.called_with = []
        self.error_code = None
        self._reported = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        self.called_with.append((args, kwargs))

    def report(self, summary, **kwargs):
        self._reported = True


class TestMultiloggerContextManager:
    def test_yields_callable(self):
        layer = _DummyLayer()
        with multilogger(layer) as ml:
            ml("test")
        assert len(layer.called_with) == 1

    def test_calls_report_on_exit(self):
        layer = _DummyLayer()
        with multilogger(layer) as ml:
            pass
        assert layer._reported

    def test_multiple_layers_different_types(self):
        """multilogger keys by type(log), so we need distinct types."""

        class _LayerA(_DummyLayer):
            pass

        class _LayerB(_DummyLayer):
            pass

        l1 = _LayerA("a")
        l2 = _LayerB("b")
        with multilogger(l1, l2) as ml:
            ml("event")
        assert len(l1.called_with) == 1
        assert len(l2.called_with) == 1


# ---------------------------------------------------------------------------
# Summary class
# ---------------------------------------------------------------------------

class TestSummary:
    def test_empty_summary_shows_nothing(self, capsys):
        s = Summary()
        s.show()
        assert s.is_empty()
        assert capsys.readouterr().out == ""

    def test_add_makes_non_empty(self):
        s = Summary()
        s.add("hello")
        assert not s.is_empty()

    def test_show_outputs_content(self, capsys):
        s = Summary()
        s.add("line1")
        s.show()
        out = capsys.readouterr().out
        assert "line1" in out

    def test_section_context_manager(self, capsys):
        s = Summary()
        with s.section("Section A"):
            s.add("detail")
        s.show()
        out = capsys.readouterr().out
        assert "Section A" in out
        assert "detail" in out

    def test_nested_sections(self, capsys):
        s = Summary()
        with s.section("Outer"):
            s.add("outer line")
            with s.section("Inner"):
                s.add("inner line")
        s.show()
        out = capsys.readouterr().out
        assert "Outer" in out
        assert "Inner" in out
        assert "inner line" in out

    def test_empty_section_not_shown(self, capsys):
        s = Summary()
        s.add("visible")
        with s.section("Empty"):
            pass
        s.show()
        out = capsys.readouterr().out
        assert "Empty" not in out

    def test_newline_adds_blank(self):
        s = Summary()
        s.newline()
        assert "" in s.root.body

    def test_underline(self):
        s = Summary()
        assert s.underline(5, char="=") == "====="
        assert len(s.underline(3, depth=0)) == 3

    def test_show_with_custom_printfun(self):
        s = Summary()
        s.add("custom")
        output = []
        s.show(printfun=output.append)
        assert any("custom" in line for line in output)


# ---------------------------------------------------------------------------
# get_available_ram (mocked — no real cgroup/psutil dependency)
# ---------------------------------------------------------------------------

class TestGetAvailableRam:
    def test_non_slurm_returns_none(self):
        """Without SLURM_JOB_ID and without exception, the function returns None
        (the psutil fallback is only in the except branch)."""
        from milabench.utils import get_available_ram

        env = os.environ.copy()
        env.pop("SLURM_JOB_ID", None)
        with patch.dict(os.environ, env, clear=True):
            result = get_available_ram(leeway=100)
            assert result is None

    def test_slurm_reads_cgroup(self, tmp_path):
        from milabench.utils import get_available_ram

        job_id = "12345"
        cgroup_dir = tmp_path / f"system.slice/slurmstepd.scope/job_{job_id}"
        cgroup_dir.mkdir(parents=True)

        (cgroup_dir / "memory.max").write_text("10000")
        (cgroup_dir / "memory.current").write_text("3000")

        fake_path = str(tmp_path) + "/system.slice/slurmstepd.scope/job_{}"

        with patch.dict(os.environ, {"SLURM_JOB_ID": job_id}):
            # Patch the filename construction inside get_available_ram
            # The function builds: f"/sys/fs/cgroup/system.slice/slurmstepd.scope/job_{jobid}"
            # We need to redirect that to tmp_path
            original_open = open

            def patched_open(path, *args, **kwargs):
                if isinstance(path, str) and "/sys/fs/cgroup/" in path:
                    path = path.replace(
                        "/sys/fs/cgroup",
                        str(tmp_path),
                    )
                return original_open(path, *args, **kwargs)

            with patch("builtins.open", side_effect=patched_open):
                result = get_available_ram(leeway=500)
                assert result == 10000 - 3000 - 500

    def test_slurm_cgroup_error_falls_back_to_psutil(self):
        from milabench.utils import get_available_ram

        mock_vm = MagicMock()
        mock_vm.available = 4096

        with patch.dict(os.environ, {"SLURM_JOB_ID": "99999"}):
            with patch("psutil.virtual_memory", return_value=mock_vm):
                result = get_available_ram(leeway=96)
                assert result == 4096 - 96
