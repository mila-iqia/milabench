import asyncio
import io

import pytest
from types import SimpleNamespace

import milabench.syslog as _syslog_mod
from milabench.capability import (
    _failure,
    is_system_capable,
    is_system_capable_with_reasons,
    is_system_capable_report,
)


def _make_pack(system=None, requires_capabilities=None, name="test_bench"):
    """Build a minimal pack-like object accepted by capability functions."""
    config = {"name": name, "system": system or {"gpu_count": 8, "memory": 64}}
    if requires_capabilities is not None:
        config["requires_capabilities"] = requires_capabilities

    messages = []

    async def message(msg):
        messages.append(msg)

    return SimpleNamespace(config=config, message=message), messages


def _run(coro):
    """Run an async coroutine synchronously without pytest-asyncio."""
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


@pytest.fixture
def capture_syslog(monkeypatch):
    """Capture syslog output by replacing the module-level stderr reference."""
    buf = io.StringIO()
    monkeypatch.setattr(_syslog_mod, "stderr", buf)
    return buf


# ── is_system_capable_with_reasons ────────────────────────────────────


class TestIsSystemCapableWithReasons:
    def test_no_requirements(self):
        pack, _ = _make_pack(requires_capabilities=[])
        ok, whys = is_system_capable_with_reasons(pack)
        assert ok is True
        assert whys == []

    def test_missing_requires_key(self):
        """When requires_capabilities is absent from config, default to []."""
        pack, _ = _make_pack()
        ok, whys = is_system_capable_with_reasons(pack)
        assert ok is True
        assert whys == []

    def test_all_conditions_met(self):
        pack, _ = _make_pack(
            system={"gpu_count": 8, "memory": 64},
            requires_capabilities=["gpu_count >= 4", "memory >= 32"],
        )
        ok, whys = is_system_capable_with_reasons(pack)
        assert ok is True
        assert whys == []

    def test_single_condition_fails(self):
        pack, _ = _make_pack(
            system={"gpu_count": 2, "memory": 64},
            requires_capabilities=["gpu_count >= 4"],
        )
        ok, whys = is_system_capable_with_reasons(pack)
        assert ok is False
        assert whys == ["gpu_count >= 4"]

    def test_multiple_conditions_some_fail(self):
        pack, _ = _make_pack(
            system={"gpu_count": 2, "memory": 16},
            requires_capabilities=["gpu_count >= 4", "memory >= 32", "memory >= 8"],
        )
        ok, whys = is_system_capable_with_reasons(pack)
        assert ok is False
        assert "gpu_count >= 4" in whys
        assert "memory >= 32" in whys
        assert "memory >= 8" not in whys

    def test_all_conditions_fail(self):
        pack, _ = _make_pack(
            system={"gpu_count": 1, "memory": 4},
            requires_capabilities=["gpu_count >= 4", "memory >= 32"],
        )
        ok, whys = is_system_capable_with_reasons(pack)
        assert ok is False
        assert len(whys) == 2

    def test_does_not_mutate_config(self):
        """eval() injects __builtins__; verify config['system'] stays clean."""
        pack, _ = _make_pack(
            system={"gpu_count": 8},
            requires_capabilities=["gpu_count >= 1"],
        )
        original_keys = set(pack.config["system"].keys())
        is_system_capable_with_reasons(pack)
        assert set(pack.config["system"].keys()) == original_keys

    def test_boolean_capability(self):
        pack, _ = _make_pack(
            system={"has_fp16": True},
            requires_capabilities=["has_fp16"],
        )
        ok, whys = is_system_capable_with_reasons(pack)
        assert ok is True
        assert whys == []

    def test_boolean_capability_false(self):
        pack, _ = _make_pack(
            system={"has_fp16": False},
            requires_capabilities=["has_fp16"],
        )
        ok, whys = is_system_capable_with_reasons(pack)
        assert ok is False
        assert whys == ["has_fp16"]

    def test_expression_with_arithmetic(self):
        pack, _ = _make_pack(
            system={"gpu_count": 4, "mem_per_gpu": 16},
            requires_capabilities=["gpu_count * mem_per_gpu >= 64"],
        )
        ok, whys = is_system_capable_with_reasons(pack)
        assert ok is True

    def test_condition_referencing_missing_key_raises(self):
        """A condition using an undefined variable should raise NameError."""
        pack, _ = _make_pack(
            system={"gpu_count": 8},
            requires_capabilities=["nonexistent_var >= 4"],
        )
        with pytest.raises(NameError):
            is_system_capable_with_reasons(pack)

    def test_equality_condition(self):
        pack, _ = _make_pack(
            system={"gpu_count": 4},
            requires_capabilities=["gpu_count == 4"],
        )
        ok, whys = is_system_capable_with_reasons(pack)
        assert ok is True
        assert whys == []

    def test_string_capability(self):
        pack, _ = _make_pack(
            system={"arch": "cuda"},
            requires_capabilities=["arch == 'cuda'"],
        )
        ok, whys = is_system_capable_with_reasons(pack)
        assert ok is True

    def test_string_capability_mismatch(self):
        pack, _ = _make_pack(
            system={"arch": "rocm"},
            requires_capabilities=["arch == 'cuda'"],
        )
        ok, whys = is_system_capable_with_reasons(pack)
        assert ok is False
        assert whys == ["arch == 'cuda'"]

    def test_returns_tuple(self):
        pack, _ = _make_pack(requires_capabilities=[])
        result = is_system_capable_with_reasons(pack)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ── is_system_capable ────────────────────────────────────────────────


class TestIsSystemCapable:
    def test_capable_returns_true(self):
        pack, _ = _make_pack(
            system={"gpu_count": 8},
            requires_capabilities=["gpu_count >= 4"],
        )
        assert is_system_capable(pack) is True

    def test_not_capable_returns_false(self, capture_syslog):
        pack, _ = _make_pack(
            system={"gpu_count": 2},
            requires_capabilities=["gpu_count >= 4"],
        )
        result = is_system_capable(pack)
        assert result is False
        assert "gpu_count >= 4" in capture_syslog.getvalue()

    def test_logs_each_failing_reason(self, capture_syslog):
        pack, _ = _make_pack(
            system={"gpu_count": 1, "memory": 4},
            requires_capabilities=["gpu_count >= 4", "memory >= 32"],
        )
        is_system_capable(pack)
        output = capture_syslog.getvalue()
        assert "gpu_count >= 4" in output
        assert "memory >= 32" in output

    def test_no_requirements(self):
        pack, _ = _make_pack(requires_capabilities=[])
        assert is_system_capable(pack) is True

    def test_log_contains_bench_name(self, capture_syslog):
        pack, _ = _make_pack(
            name="my_bench",
            system={"gpu_count": 1},
            requires_capabilities=["gpu_count >= 4"],
        )
        is_system_capable(pack)
        assert "my_bench" in capture_syslog.getvalue()

    def test_no_log_when_capable(self, capture_syslog):
        pack, _ = _make_pack(
            system={"gpu_count": 8},
            requires_capabilities=["gpu_count >= 4"],
        )
        is_system_capable(pack)
        assert capture_syslog.getvalue() == ""


# ── _failure (async) ─────────────────────────────────────────────────


class TestFailure:
    def test_sends_message(self):
        pack, messages = _make_pack(name="bench_x")
        _run(_failure(pack, "gpu_count >= 4"))
        assert len(messages) == 1
        assert "bench_x" in messages[0]
        assert "gpu_count >= 4" in messages[0]

    def test_message_format(self):
        pack, messages = _make_pack(name="bench_y")
        _run(_failure(pack, "memory >= 64"))
        assert messages[0].startswith("Skip bench_y")
        assert "capability is not satisfied" in messages[0]


# ── is_system_capable_report (async) ─────────────────────────────────


class TestIsSystemCapableReport:
    def test_capable(self):
        pack, messages = _make_pack(
            system={"gpu_count": 8},
            requires_capabilities=["gpu_count >= 4"],
        )
        result = _run(is_system_capable_report(pack))
        assert result is True
        assert messages == []

    def test_not_capable_sends_messages(self):
        pack, messages = _make_pack(
            system={"gpu_count": 1},
            requires_capabilities=["gpu_count >= 4"],
        )
        result = _run(is_system_capable_report(pack))
        assert result is False
        assert len(messages) == 1
        assert "gpu_count >= 4" in messages[0]

    def test_multiple_failures(self):
        pack, messages = _make_pack(
            system={"gpu_count": 1, "memory": 4},
            requires_capabilities=["gpu_count >= 4", "memory >= 32"],
        )
        result = _run(is_system_capable_report(pack))
        assert result is False
        assert len(messages) == 2

    def test_no_requirements(self):
        pack, messages = _make_pack(requires_capabilities=[])
        result = _run(is_system_capable_report(pack))
        assert result is True
        assert messages == []

    def test_message_includes_pack_name(self):
        pack, messages = _make_pack(
            name="special_bench",
            system={"gpu_count": 1},
            requires_capabilities=["gpu_count >= 4"],
        )
        _run(is_system_capable_report(pack))
        assert "special_bench" in messages[0]
