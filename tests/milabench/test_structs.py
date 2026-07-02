"""Tests for milabench.structs module."""

import asyncio
from dataclasses import fields, asdict
from unittest.mock import patch

import pytest

from milabench.structs import BenchLogEntry, Job, TaskLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakePack:
    """Minimal stand-in for a pack object with a .tag attribute."""

    def __init__(self, tag="fake-tag"):
        self.tag = tag


def _run(coro):
    """Run async code without pytest-asyncio."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# BenchLogEntry
# ---------------------------------------------------------------------------


class TestBenchLogEntry:
    def test_basic_construction(self):
        pack = _FakePack("my-tag")
        entry = BenchLogEntry(pack=pack, event="start", data={"key": "val"})

        assert entry.pack is pack
        assert entry.event == "start"
        assert entry.data == {"key": "val"}

    def test_tag_property_delegates_to_pack(self):
        pack = _FakePack("benchmark-xyz")
        entry = BenchLogEntry(pack=pack, event="e", data=None)

        assert entry.tag == "benchmark-xyz"

    def test_tag_property_reflects_pack_changes(self):
        pack = _FakePack("old")
        entry = BenchLogEntry(pack=pack, event="e", data=None)
        pack.tag = "new"

        assert entry.tag == "new"

    def test_inherits_logentry_methods(self):
        pack = _FakePack()
        entry = BenchLogEntry(pack=pack, event="ev", data=42, pipe="stdout")

        assert entry.get("pipe", None) == "stdout"
        assert entry.get("nonexistent", "default") == "default"

        d = entry.dict()
        assert d["event"] == "ev"
        assert d["data"] == 42
        assert d["pipe"] == "stdout"
        assert d["pack"] is pack

    def test_default_pipe_is_none(self):
        entry = BenchLogEntry(pack=_FakePack(), event="e", data=None)
        assert entry.pipe is None

    def test_extra_kwargs_passed_to_logentry(self):
        entry = BenchLogEntry(pack=_FakePack(), event="e", data="d", pipe="stderr")
        assert entry.pipe == "stderr"


# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------


class TestJob:
    def test_required_field_argv(self):
        job = Job(argv=["python", "script.py"])
        assert job.argv == ["python", "script.py"]

    def test_default_values_are_none(self):
        job = Job(argv=[])
        assert job.info is None
        assert job.env is None
        assert job.cwd is None
        assert job.preexec_fn is None
        assert job.properties is None

    def test_all_fields_assigned(self):
        fn = lambda: None
        job = Job(
            argv=["cmd"],
            info={"k": "v"},
            env={"PATH": "/usr/bin"},
            cwd="/tmp",
            preexec_fn=fn,
            properties={"gpu": 0},
        )
        assert job.argv == ["cmd"]
        assert job.info == {"k": "v"}
        assert job.env == {"PATH": "/usr/bin"}
        assert job.cwd == "/tmp"
        assert job.preexec_fn is fn
        assert job.properties == {"gpu": 0}

    def test_field_names_and_count(self):
        names = [f.name for f in fields(Job)]
        assert names == ["argv", "info", "env", "cwd", "preexec_fn", "properties"]

    def test_equality(self):
        a = Job(argv=["a"], cwd="/x")
        b = Job(argv=["a"], cwd="/x")
        assert a == b

    def test_inequality(self):
        a = Job(argv=["a"])
        b = Job(argv=["b"])
        assert a != b

    def test_repr(self):
        job = Job(argv=["run"])
        r = repr(job)
        assert "Job" in r
        assert "run" in r

    def test_asdict(self):
        job = Job(argv=["x"], info={"a": 1})
        d = asdict(job)
        assert d == {
            "argv": ["x"],
            "info": {"a": 1},
            "env": None,
            "cwd": None,
            "preexec_fn": None,
            "properties": None,
        }

    def test_empty_argv(self):
        job = Job(argv=[])
        assert job.argv == []

    def test_env_with_non_string_values(self):
        job = Job(argv=[], env={"NUM_GPUS": 4, "DEBUG": True})
        assert job.env["NUM_GPUS"] == 4
        assert job.env["DEBUG"] is True


# ---------------------------------------------------------------------------
# TaskLogger  (covers lines 33-57)
# ---------------------------------------------------------------------------


class TestTaskLogger:
    def test_stores_common_args(self):
        logger = TaskLogger(pack=_FakePack(), extra="value")
        assert logger._common["extra"] == "value"
        assert isinstance(logger._common["pack"], _FakePack)

    def test_getattr_returns_callable(self):
        logger = TaskLogger(pack=_FakePack())
        cm_factory = logger.some_task
        assert callable(cm_factory)

    def test_task_sends_start_and_end_events(self):
        pack = _FakePack("tag-1")
        logger = TaskLogger(pack=pack)

        sent = []

        async def fake_send(msg):
            sent.append(msg)

        async def run():
            with patch("milabench.structs.send", side_effect=fake_send):
                async with logger.download("downloading model"):
                    pass

        _run(run())

        assert len(sent) == 2

        start_entry = sent[0]
        assert isinstance(start_entry, BenchLogEntry)
        assert start_entry.event == "start-task"
        assert start_entry.data["task"] == "download"
        assert start_entry.data["message"] == "downloading model"
        assert "token" in start_entry.data

        end_entry = sent[1]
        assert isinstance(end_entry, BenchLogEntry)
        assert end_entry.event == "end-task"
        assert end_entry.data["task"] == "download"
        assert end_entry.data["token"] == start_entry.data["token"]

    def test_task_passes_extra_kwargs_in_start_event(self):
        pack = _FakePack()
        logger = TaskLogger(pack=pack)

        sent = []

        async def fake_send(msg):
            sent.append(msg)

        async def run():
            with patch("milabench.structs.send", side_effect=fake_send):
                async with logger.install("installing deps", retries=3, url="http://x"):
                    pass

        _run(run())

        start_data = sent[0].data
        assert start_data["retries"] == 3
        assert start_data["url"] == "http://x"

    def test_task_tokens_are_unique(self):
        pack = _FakePack()
        logger = TaskLogger(pack=pack)

        tokens = []

        async def fake_send(msg):
            if msg.event == "start-task":
                tokens.append(msg.data["token"])

        async def run():
            with patch("milabench.structs.send", side_effect=fake_send):
                async with logger.task_a("msg"):
                    pass
                async with logger.task_b("msg"):
                    pass

        _run(run())

        assert len(tokens) == 2
        assert tokens[0] != tokens[1]

    def test_common_args_forwarded_to_benchlogentry(self):
        pack = _FakePack("common-tag")
        logger = TaskLogger(pack=pack)

        sent = []

        async def fake_send(msg):
            sent.append(msg)

        async def run():
            with patch("milabench.structs.send", side_effect=fake_send):
                async with logger.build("building"):
                    pass

        _run(run())

        for entry in sent:
            assert entry.pack is pack
            assert entry.tag == "common-tag"

    def test_different_task_names_via_getattr(self):
        logger = TaskLogger(pack=_FakePack())

        sent = []

        async def fake_send(msg):
            sent.append(msg)

        async def run():
            with patch("milabench.structs.send", side_effect=fake_send):
                async with logger.alpha("a"):
                    pass
                async with logger.beta("b"):
                    pass

        _run(run())

        task_names = [e.data["task"] for e in sent if e.event == "start-task"]
        assert task_names == ["alpha", "beta"]

    def test_body_executes_between_start_and_end(self):
        """Verify the yield happens between start-task and end-task sends."""
        logger = TaskLogger(pack=_FakePack())

        timeline = []

        async def fake_send(msg):
            timeline.append(msg.event)

        async def run():
            with patch("milabench.structs.send", side_effect=fake_send):
                async with logger.step("working"):
                    timeline.append("body")

        _run(run())

        assert timeline == ["start-task", "body", "end-task"]
