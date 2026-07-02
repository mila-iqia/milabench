"""Tests for pure-logic functions in milabench.sizer."""

import time
from collections import defaultdict
from unittest.mock import MagicMock

import pytest
import yaml
from cantilever.core.statstream import StatStream

from milabench.sizer import (
    BenchStats,
    arch_to_device,
    broadcast,
    compact_dump,
    deduplicate_observation,
    to_octet,
)


# ---------------------------------------------------------------------------
# to_octet: metric-prefix byte string → float
# ---------------------------------------------------------------------------

class TestToOctet:
    """Tests for to_octet(), the byte-string parser."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("1 GiB", 1 * 1024**3),
            ("2 GiB", 2 * 1024**3),
            ("0.5 GiB", 0.5 * 1024**3),
        ],
    )
    def test_gibibytes(self, value, expected):
        # to_octet doesn't handle spaces — strip them as the callers do
        assert to_octet(value.replace(" ", "")) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("1GB", 1 * 10**9),
            ("2GB", 2 * 10**9),
            ("0.5GB", 0.5 * 10**9),
        ],
    )
    def test_gigabytes_si(self, value, expected):
        assert to_octet(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("12Go", 12 * 10**9),
            ("48Go", 48 * 10**9),
        ],
    )
    def test_gigaoctets(self, value, expected):
        assert to_octet(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("1024MiB", 1024 * 1024**2),
            ("41920MiB", 41920 * 1024**2),
            ("512MiB", 512 * 1024**2),
        ],
    )
    def test_mebibytes(self, value, expected):
        assert to_octet(value) == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("500MB", 500 * 10**6),
            ("1MB", 1 * 10**6),
        ],
    )
    def test_megabytes_si(self, value, expected):
        assert to_octet(value) == expected

    def test_terabytes(self):
        assert to_octet("1TB") == 10**12

    def test_tebibytes(self):
        assert to_octet("1TiB") == 1024**4

    def test_kilobytes(self):
        assert to_octet("4kB") == 4 * 10**3

    def test_kibibytes(self):
        assert to_octet("4kiB") == 4 * 1024**1

    def test_plain_octets_with_suffix(self):
        assert to_octet("1000o") == 1000.0

    def test_plain_ioctets(self):
        assert to_octet("2048io") == 2048.0

    def test_bare_number(self):
        assert to_octet("42") == 42.0

    def test_zero(self):
        assert to_octet("0") == 0.0


# ---------------------------------------------------------------------------
# arch_to_device: arch string → device type
# ---------------------------------------------------------------------------

class TestArchToDevice:
    @pytest.mark.parametrize(
        "arch, expected",
        [
            ("cuda", "cuda"),
            ("cpu", "cpu"),
            ("rocm", "cuda"),
            ("xpu", "xpu"),
            ("hpu", "hpu"),
            ("mps", "mps"),
            ("meta", "meta"),
        ],
    )
    def test_known_architectures(self, arch, expected):
        assert arch_to_device(arch) == expected

    def test_unknown_arch_defaults_to_cpu(self):
        assert arch_to_device("nonexistent_arch") == "cpu"

    def test_empty_string_defaults_to_cpu(self):
        assert arch_to_device("") == "cpu"


# ---------------------------------------------------------------------------
# broadcast: call a list of delegates with error resilience
# ---------------------------------------------------------------------------

class TestBroadcast:
    def test_calls_all_delegates(self):
        results = []
        broadcast([lambda x: results.append(x), lambda x: results.append(x * 2)], 5)
        assert results == [5, 10]

    def test_empty_delegates(self):
        broadcast([], 1, 2, 3)

    def test_error_in_one_does_not_stop_others(self, capsys):
        results = []

        def bad(_):
            raise ValueError("boom")

        def good(x):
            results.append(x)

        broadcast([bad, good], 42)
        assert results == [42]
        captured = capsys.readouterr()
        assert "boom" in captured.out

    def test_passes_kwargs(self):
        received = {}

        def capture(**kw):
            received.update(kw)

        broadcast([capture], key="val")
        assert received == {"key": "val"}


# ---------------------------------------------------------------------------
# BenchStats: dataclass helpers
# ---------------------------------------------------------------------------

class TestBenchStats:
    def test_initial_state(self):
        bs = BenchStats("test_bench")
        assert bs.benchname == "test_bench"
        assert bs.active_count == 0
        assert bs.rc == []
        assert bs.early_stopped == []

    def test_max_memory_usage_uses_peak_when_available(self):
        bs = BenchStats("b")
        bs.peak_usage += 500
        bs.max_usage += 100
        assert bs.max_memory_usage() == 500

    def test_max_memory_usage_falls_back_to_max_usage(self):
        bs = BenchStats("b")
        bs.max_usage += 300
        bs.max_usage += 400
        assert bs.max_memory_usage() == 400

    def test_max_memory_usage_no_data(self):
        bs = BenchStats("b")
        assert bs.max_memory_usage() == float("-inf")

    def test_has_stopped_early_false_when_empty(self):
        bs = BenchStats("b")
        assert bs.has_stopped_early() is False

    def test_has_stopped_early_true(self):
        bs = BenchStats("b")
        bs.early_stopped.append(True)
        assert bs.has_stopped_early() is True

    def test_has_stopped_early_last_false(self):
        bs = BenchStats("b")
        bs.early_stopped.append(True)
        bs.early_stopped.append(False)
        assert bs.has_stopped_early() is False

    def test_statstream_fields_accumulate(self):
        bs = BenchStats("b")
        bs.perf += 10.0
        bs.perf += 20.0
        assert bs.perf.avg == 15.0
        assert bs.perf.current_count == 2


# ---------------------------------------------------------------------------
# deduplicate_observation: the core deduplication logic (0% coverage)
# ---------------------------------------------------------------------------

def _obs(batch_size, cpu, memory_mib, perf, t=None):
    """Helper to build an observation dict."""
    return {
        "batch_size": batch_size,
        "cpu": cpu,
        "memory": f"{memory_mib} MiB",
        "perf": perf,
        "time": t or int(time.time()),
    }


class TestDeduplicateObservation:

    def test_empty_input(self):
        assert deduplicate_observation({}) == {}

    def test_version_key_preserved(self):
        result = deduplicate_observation({"version": 2.0})
        assert result["version"] == 2.0

    def test_bench_with_no_observations(self):
        scaling = {"mybench": {"observations": []}}
        result = deduplicate_observation(scaling)
        assert result["mybench"]["observations"] == []

    def test_single_observation_kept(self):
        obs = _obs(32, 4, 1000, 100.0)
        scaling = {"bench": {"observations": [obs]}}
        result = deduplicate_observation(scaling)
        assert len(result["bench"]["observations"]) == 1
        assert result["bench"]["observations"][0]["batch_size"] == 32

    def test_single_observation_zero_perf_dropped(self):
        obs = _obs(32, 4, 1000, 0)
        scaling = {"bench": {"observations": [obs]}}
        result = deduplicate_observation(scaling)
        assert result["bench"]["observations"] == []

    def test_unique_observations_all_kept(self):
        scaling = {
            "bench": {
                "observations": [
                    _obs(16, 4, 500, 50.0),
                    _obs(32, 4, 1000, 100.0),
                    _obs(64, 8, 2000, 200.0),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        sizes = [o["batch_size"] for o in result["bench"]["observations"]]
        assert sorted(sizes) == [16, 32, 64]

    def test_duplicates_merged_when_similar(self):
        t = int(time.time())
        scaling = {
            "bench": {
                "observations": [
                    _obs(32, 4, 1000, 100.0, t),
                    _obs(32, 4, 1005, 101.0, t + 10),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        obs = result["bench"]["observations"]
        assert len(obs) == 1
        assert obs[0]["batch_size"] == 32
        assert obs[0]["time"] == t + 10

    def test_duplicates_not_merged_when_very_different(self):
        t = int(time.time())
        scaling = {
            "bench": {
                "observations": [
                    _obs(32, 4, 1000, 100.0, t),
                    _obs(32, 4, 5000, 500.0, t + 10),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        obs = result["bench"]["observations"]
        assert len(obs) == 2

    def test_zero_perf_entries_excluded_from_merge(self):
        t = int(time.time())
        scaling = {
            "bench": {
                "observations": [
                    _obs(32, 4, 1000, 100.0, t),
                    _obs(32, 4, 1000, 0, t + 1),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        obs = result["bench"]["observations"]
        # Only the perf>0 entry remains (single valid → should_generate_single)
        assert len(obs) == 1
        assert obs[0]["perf"] > 0

    def test_all_duplicate_entries_zero_perf_dropped(self):
        t = int(time.time())
        scaling = {
            "bench": {
                "observations": [
                    _obs(32, 4, 1000, 0, t),
                    _obs(32, 4, 1000, 0, t + 1),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        assert result["bench"]["observations"] == []

    def test_output_sorted_by_batch_size(self):
        scaling = {
            "bench": {
                "observations": [
                    _obs(64, 4, 2000, 200.0),
                    _obs(16, 4, 500, 50.0),
                    _obs(32, 4, 1000, 100.0),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        sizes = [o["batch_size"] for o in result["bench"]["observations"]]
        assert sizes == sorted(sizes)

    def test_multiple_benchmarks_handled_independently(self):
        scaling = {
            "version": 2.0,
            "bench_a": {"observations": [_obs(32, 4, 1000, 100.0)]},
            "bench_b": {"observations": [_obs(64, 8, 2000, 200.0)]},
        }
        result = deduplicate_observation(scaling)
        assert "bench_a" in result
        assert "bench_b" in result
        assert result["version"] == 2.0

    def test_three_similar_duplicates_merged(self):
        t = int(time.time())
        scaling = {
            "bench": {
                "observations": [
                    _obs(32, 4, 1000, 100.0, t),
                    _obs(32, 4, 1002, 100.5, t + 5),
                    _obs(32, 4, 1001, 100.2, t + 10),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        obs = result["bench"]["observations"]
        assert len(obs) == 1
        assert obs[0]["batch_size"] == 32
        assert obs[0]["time"] == t + 10

    def test_mixed_unique_and_duplicate(self):
        t = int(time.time())
        scaling = {
            "bench": {
                "observations": [
                    _obs(16, 4, 500, 50.0, t),
                    _obs(32, 4, 1000, 100.0, t),
                    _obs(32, 4, 1003, 101.0, t + 10),
                    _obs(64, 8, 2000, 200.0, t),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        obs = result["bench"]["observations"]
        sizes = [o["batch_size"] for o in obs]
        assert sorted(sizes) == [16, 32, 64]

    def test_merged_observation_memory_format(self):
        t = int(time.time())
        scaling = {
            "bench": {
                "observations": [
                    _obs(32, 4, 1000, 100.0, t),
                    _obs(32, 4, 1010, 101.0, t + 10),
                ]
            }
        }
        result = deduplicate_observation(scaling)
        obs = result["bench"]["observations"][0]
        assert "MiB" in obs["memory"]
        assert isinstance(obs["perf"], float)


# ---------------------------------------------------------------------------
# compact_dump: YAML formatting
# ---------------------------------------------------------------------------

class TestCompactDump:
    def test_returns_dumper_class(self):
        dumper = compact_dump()
        assert issubclass(dumper, yaml.SafeDumper)

    def test_roundtrip_preserves_data(self):
        data = {
            "bench": {
                "observations": [
                    {"batch_size": 32, "memory": "1000 MiB", "perf": 100.0},
                    {"batch_size": 64, "memory": "2000 MiB", "perf": 200.0},
                ]
            }
        }
        dumped = yaml.dump(data, Dumper=compact_dump())
        reloaded = yaml.safe_load(dumped)
        assert reloaded == data


# ---------------------------------------------------------------------------
# StatStream integration (used heavily in sizer)
# ---------------------------------------------------------------------------

class TestStatStreamUsage:
    """Verify StatStream behaves as sizer.py assumes."""

    def test_basic_accumulation(self):
        s = StatStream(drop_first_obs=0)
        s += 10
        s += 20
        assert s.current_count == 2
        assert s.avg == 15.0
        assert s.max == 20
        assert s.min == 10

    def test_empty_stream_defaults(self):
        s = StatStream(drop_first_obs=0)
        assert s.current_count == 0
        assert s.max == float("-inf")

    def test_single_value(self):
        s = StatStream(drop_first_obs=0)
        s += 42
        assert s.avg == 42.0
        assert s.current_count == 1
        assert s.sd == 0.0
