import json
import os

import pytest

from milabench.status import serialize_exception, StatusTracker, resume_from_status


# ---------------------------------------------------------------------------
# Helpers – lightweight stand-ins that satisfy the interfaces expected by
# StatusTracker and resume_from_status without pulling in the full pack
# machinery.
# ---------------------------------------------------------------------------

class StrPath(str):
    """str subclass that supports the ``/`` operator so that
    ``logdir / "status.jsonl"`` returns something the ``+`` operator
    still works on (plain Path objects fail on ``path + ".lock"``).
    """

    def __truediv__(self, other):
        return StrPath(os.path.join(self, other))


class MockPack:
    def __init__(self, name, logdir_path):
        self.config = {"name": name}
        self._logdir = logdir_path

    def logdir(self):
        return StrPath(self._logdir)


class MockPacks:
    """Mapping-like object whose ``.values()`` returns an *iterator*
    (the production code calls ``next(packs.values())``).
    """

    def __init__(self, packs_dict):
        self._dict = packs_dict

    def __len__(self):
        return len(self._dict)

    def values(self):
        return iter(self._dict.values())


@pytest.fixture
def logdir(tmp_path):
    return tmp_path / "logs"


@pytest.fixture
def pack(logdir):
    logdir.mkdir(parents=True, exist_ok=True)
    return MockPack("bench_a", str(logdir))


@pytest.fixture
def packs(pack):
    return MockPacks({"bench_a": pack})


@pytest.fixture
def tracker(packs):
    return StatusTracker(packs, repeat=1)


# ===================================================================
# serialize_exception
# ===================================================================


class TestSerializeException:
    def test_with_real_exception(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_type, exc_value, exc_tb = sys.exc_info()
            result = serialize_exception(exc_type, exc_value, exc_tb)

        assert result["type"] == "ValueError"
        assert result["message"] == "boom"
        assert isinstance(result["traceback"], list)
        assert any("boom" in line for line in result["traceback"])

    def test_all_none(self):
        result = serialize_exception(None, None, None)
        assert result == {"type": None, "message": None, "traceback": None}

    def test_type_only(self):
        result = serialize_exception(RuntimeError, None, None)
        assert result["type"] == "RuntimeError"
        assert result["message"] is None
        assert result["traceback"] is None

    def test_type_and_message(self):
        exc = KeyError("missing_key")
        result = serialize_exception(type(exc), exc, None)
        assert result["type"] == "KeyError"
        assert "missing_key" in result["message"]
        assert result["traceback"] is None

    def test_nested_exception(self):
        try:
            try:
                raise OSError("disk full")
            except OSError:
                raise RuntimeError("wrapper") from None
        except RuntimeError:
            import sys
            info = sys.exc_info()
            result = serialize_exception(*info)

        assert result["type"] == "RuntimeError"
        assert "wrapper" in result["message"]


# ===================================================================
# StatusTracker.__init__
# ===================================================================


class TestStatusTrackerInit:
    def test_basic_init(self, tracker, pack):
        assert tracker.repeat == 1
        expected = os.path.join(pack._logdir, "status.jsonl")
        assert tracker.sentinel == expected

    def test_repeat_value_preserved(self, packs):
        t = StatusTracker(packs, repeat=5)
        assert t.repeat == 5

    def test_empty_packs_raises(self):
        empty = MockPacks({})
        with pytest.raises(AssertionError, match="No packs available"):
            StatusTracker(empty, repeat=1)


# ===================================================================
# StatusTracker.append  (direct call with a dict)
# ===================================================================


class TestStatusTrackerAppend:
    def test_append_writes_json_line(self, tracker):
        tracker.append({"phase": "run", "status": "started"})

        with open(tracker.sentinel) as fp:
            lines = fp.read().strip().splitlines()

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["phase"] == "run"
        assert data["status"] == "started"

    def test_append_multiple(self, tracker):
        tracker.append({"a": 1})
        tracker.append({"b": 2})
        tracker.append({"c": 3})

        with open(tracker.sentinel) as fp:
            lines = fp.read().strip().splitlines()

        assert len(lines) == 3
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[2]) == {"c": 3}

    def test_append_creates_file_if_missing(self, tracker):
        assert not os.path.exists(tracker.sentinel)
        tracker.append({"x": "y"})
        assert os.path.exists(tracker.sentinel)

    def test_append_serialises_nested_data(self, tracker):
        payload = {"nested": {"key": [1, 2, 3]}, "flag": True}
        tracker.append(payload)

        with open(tracker.sentinel) as fp:
            data = json.loads(fp.readline())

        assert data == payload


# ===================================================================
# StatusTracker.__enter__ / __exit__
# The production code calls self.append(phase=..., status=..., ...)
# but append(self, obj) only accepts a positional argument, so these
# raise TypeError.  We verify the failure mode explicitly.
# ===================================================================


class TestStatusTrackerContextManager:
    def test_enter_raises_due_to_append_signature(self, tracker):
        with pytest.raises(TypeError):
            tracker.__enter__()

    def test_exit_no_exception_raises_due_to_append_signature(self, tracker):
        with pytest.raises(TypeError):
            tracker.__exit__(None, None, None)

    def test_exit_with_exception_raises_due_to_append_signature(self, tracker):
        try:
            raise RuntimeError("fail")
        except RuntimeError:
            import sys
            exc_info = sys.exc_info()
            with pytest.raises(TypeError):
                tracker.__exit__(*exc_info)


# ===================================================================
# StatusTracker.__iter__
# Same append-signature issue as the context manager: iteration will
# fail at the first self.append() call inside the generator.
# ===================================================================


class TestStatusTrackerIter:
    def test_iter_raises_due_to_append_signature(self, tracker):
        it = iter(tracker)
        with pytest.raises(TypeError):
            next(it)

    def test_iter_empty_repeat(self, packs):
        t = StatusTracker(packs, repeat=0)
        assert list(t) == []


# ===================================================================
# resume_from_status
# ===================================================================


class TestResumeFromStatus:
    def test_repeat_greater_than_one_prints_warning(self, packs, logdir, capsys):
        sentinel = os.path.join(str(logdir), "status.jsonl")
        os.makedirs(os.path.dirname(sentinel), exist_ok=True)
        with open(sentinel, "w") as fp:
            json.dump({"bench": "a", "status": "ended"}, fp)
            fp.write("\n")

        with pytest.raises((NameError, Exception)):
            resume_from_status(packs, runfolder=None, repeat=2)

        captured = capsys.readouterr()
        assert "repeat > 1 is not supported" in captured.out

    def test_repeat_one_no_warning(self, packs, logdir, capsys):
        sentinel = os.path.join(str(logdir), "status.jsonl")
        os.makedirs(os.path.dirname(sentinel), exist_ok=True)
        with open(sentinel, "w") as fp:
            json.dump({"bench": "a", "status": "ended"}, fp)
            fp.write("\n")

        with pytest.raises(NameError, match="missing_packs"):
            resume_from_status(packs, runfolder=None, repeat=1)

        captured = capsys.readouterr()
        assert "repeat > 1" not in captured.out

    def test_empty_packs_raises(self):
        empty = MockPacks({})
        with pytest.raises(AssertionError, match="No packs available"):
            resume_from_status(empty, runfolder=None, repeat=1)

    def test_reads_status_lines(self, packs, logdir):
        sentinel = os.path.join(str(logdir), "status.jsonl")
        os.makedirs(os.path.dirname(sentinel), exist_ok=True)
        with open(sentinel, "w") as fp:
            for entry in [
                {"bench": "a", "status": "started"},
                {"bench": "a", "status": "ended"},
            ]:
                json.dump(entry, fp)
                fp.write("\n")

        with pytest.raises(NameError, match="missing_packs"):
            resume_from_status(packs, runfolder=None, repeat=1)

    def test_missing_status_file_raises(self, packs, logdir):
        sentinel = os.path.join(str(logdir), "status.jsonl")
        os.makedirs(os.path.dirname(sentinel), exist_ok=True)
        with pytest.raises(FileNotFoundError):
            resume_from_status(packs, runfolder=None, repeat=1)
