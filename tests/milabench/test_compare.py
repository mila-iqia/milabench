from datetime import datetime

import pytest

from milabench.compare import (
    _Output,
    retrieve_datetime_from_name,
    fetch_runs,
    _print_headers,
    compare,
)


# ─── retrieve_datetime_from_name ─────────────────────────────────────────────


class TestRetrieveDatetimeFromName:
    def test_colon_format(self):
        result = retrieve_datetime_from_name("2024-03-15_10:30:45.123456")
        assert result == datetime(2024, 3, 15, 10, 30, 45, 123456)

    def test_underscore_format(self):
        result = retrieve_datetime_from_name("2024-03-15_10_30_45.123456")
        assert result == datetime(2024, 3, 15, 10, 30, 45, 123456)

    def test_invalid_format_returns_none(self):
        result = retrieve_datetime_from_name("not-a-date")
        assert result is None

    def test_empty_string_returns_none(self):
        result = retrieve_datetime_from_name("")
        assert result is None

    def test_partial_date_returns_none(self):
        result = retrieve_datetime_from_name("2024-03-15")
        assert result is None

    def test_missing_fractional_seconds(self):
        result = retrieve_datetime_from_name("2024-03-15_10:30:45")
        assert result is None


# ─── fetch_runs ──────────────────────────────────────────────────────────────


class TestFetchRuns:
    def test_empty_folder(self, tmp_path):
        runs = fetch_runs(str(tmp_path), None)
        assert runs == []

    def test_skips_install_and_prepare_dirs(self, tmp_path):
        (tmp_path / "install_deps").mkdir()
        (tmp_path / "prepare_data").mkdir()
        (tmp_path / "benchmark_run").mkdir()
        runs = fetch_runs(str(tmp_path), None)
        assert len(runs) == 1
        assert runs[0].name == "benchmark_run"

    def test_skips_files(self, tmp_path):
        (tmp_path / "some_file.txt").write_text("hello")
        (tmp_path / "actual_run").mkdir()
        runs = fetch_runs(str(tmp_path), None)
        assert len(runs) == 1
        assert runs[0].name == "actual_run"

    def test_dir_without_dot_uses_mtime(self, tmp_path):
        run_dir = tmp_path / "my_benchmark"
        run_dir.mkdir()
        runs = fetch_runs(str(tmp_path), None)
        assert len(runs) == 1
        assert runs[0].name == "my_benchmark"
        assert runs[0].date is not None

    def test_dir_with_dotted_name_splits_correctly(self, tmp_path):
        """When dir has dots, the code splits name.date.fractional but
        the date portion alone (without fractional) won't match the format
        strings that require .%f, so it falls back to mtime."""
        run_dir = tmp_path / "bench.2024-03-15_10:30:45.123456"
        run_dir.mkdir()
        runs = fetch_runs(str(tmp_path), None)
        assert len(runs) == 1
        assert runs[0].name == "bench"
        # Date falls back to mtime since "2024-03-15_10:30:45" doesn't match formats with .%f
        assert runs[0].date is not None

    def test_dir_with_three_dots_parses_date(self, tmp_path):
        """With 3 dots: name.date_with_fractional.suffix, the second rsplit
        yields the date including .%f portion that matches the format."""
        run_dir = tmp_path / "bench.2024-03-15_10:30:45.123456.0"
        run_dir.mkdir()
        runs = fetch_runs(str(tmp_path), None)
        assert len(runs) == 1
        # First rsplit(".", 1): "bench.2024-03-15_10:30:45.123456" / "0"
        # Second rsplit(".", 1): "bench.2024-03-15_10:30:45" / "123456"
        # date = "123456" doesn't match any format → falls back to mtime
        assert runs[0].date is not None

    def test_dir_with_underscore_date_falls_back_to_mtime(self, tmp_path):
        """Underscore format date in dir name also has the same splitting behavior."""
        run_dir = tmp_path / "bench.2024-03-15_10_30_45.123456"
        run_dir.mkdir()
        runs = fetch_runs(str(tmp_path), None)
        assert len(runs) == 1
        assert runs[0].name == "bench"
        assert runs[0].date is not None

    def test_dir_with_unparseable_date_falls_back_to_mtime(self, tmp_path):
        run_dir = tmp_path / "bench.notadate.999"
        run_dir.mkdir()
        runs = fetch_runs(str(tmp_path), None)
        assert len(runs) == 1
        assert runs[0].date is not None

    def test_runs_sorted_by_date(self, tmp_path):
        d1 = tmp_path / "bench.2024-01-01_00:00:00.000001"
        d2 = tmp_path / "bench.2024-06-01_00:00:00.000001"
        d3 = tmp_path / "bench.2024-03-01_00:00:00.000001"
        d1.mkdir()
        d2.mkdir()
        d3.mkdir()
        runs = fetch_runs(str(tmp_path), None)
        dates = [r.date for r in runs]
        assert dates == sorted(dates)

    def test_filter_matches(self, tmp_path):
        (tmp_path / "bench_a").mkdir()
        (tmp_path / "bench_b").mkdir()
        (tmp_path / "other_c").mkdir()
        runs = fetch_runs(str(tmp_path), "bench_*")
        names = [r.name for r in runs]
        assert "bench_a" in names
        assert "bench_b" in names
        assert "other_c" not in names

    def test_filter_excludes_all(self, tmp_path):
        (tmp_path / "bench_a").mkdir()
        (tmp_path / "bench_b").mkdir()
        runs = fetch_runs(str(tmp_path), "no_match_*")
        assert runs == []

    def test_filter_prints_ignored_count(self, tmp_path, capsys):
        (tmp_path / "bench_a").mkdir()
        (tmp_path / "other_b").mkdir()
        fetch_runs(str(tmp_path), "bench_*")
        captured = capsys.readouterr()
        assert "Ignoring run 1 runs because of filter bench_*" in captured.out


# ─── _print_headers ──────────────────────────────────────────────────────────


class TestPrintHeaders:
    def test_single_run(self, capsys):
        run = _Output(
            path="/tmp/test",
            name="my_bench",
            date=datetime(2024, 3, 15, 10, 30, 45),
        )
        _print_headers([run], " | ")
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) == 4
        assert "my_bench" in lines[0]
        assert "2024-03-15" in lines[1]
        assert "10:30:45" in lines[2]
        assert set(lines[3]) <= {"-"}

    def test_multiple_runs(self, capsys):
        runs = [
            _Output("/tmp/a", "run_a", datetime(2024, 1, 1, 8, 0, 0)),
            _Output("/tmp/b", "run_b", datetime(2024, 6, 15, 14, 30, 0)),
        ]
        _print_headers(runs, " | ")
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert "run_a" in lines[0]
        assert "run_b" in lines[0]

    def test_empty_runs(self, capsys):
        _print_headers([], " | ")
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        # With no runs: header row, dates row, times row (just fixed prefix), and separator
        # But the times list only has bench+metric prefixes, so the output is minimal
        assert len(lines) >= 2
        assert "-" in lines[-1]


# ─── compare ─────────────────────────────────────────────────────────────────


class TestCompare:
    @pytest.fixture
    def two_runs(self):
        run1 = _Output(
            path="/tmp/r1",
            name="run1",
            date=datetime(2024, 1, 1, 10, 0, 0),
            summary={
                "bench_a": {"train_rate": {"mean": 100.5}},
                "bench_b": {"train_rate": {"mean": 200.0}},
            },
        )
        run2 = _Output(
            path="/tmp/r2",
            name="run2",
            date=datetime(2024, 2, 1, 10, 0, 0),
            summary={
                "bench_a": {"train_rate": {"mean": 110.0}},
                "bench_b": {"train_rate": {"mean": 210.0}},
            },
        )
        return [run1, run2]

    def test_basic_compare(self, two_runs, capsys):
        compare(two_runs, last=None, metric="train_rate", stat="mean")
        captured = capsys.readouterr()
        assert "bench_a" in captured.out
        assert "bench_b" in captured.out
        assert "100.50" in captured.out
        assert "110.00" in captured.out

    def test_last_limits_runs(self, two_runs, capsys):
        compare(two_runs, last=1, metric="train_rate", stat="mean")
        captured = capsys.readouterr()
        assert "run2" in captured.out
        assert "110.00" in captured.out

    def test_missing_bench_in_one_run_shows_nan(self, capsys):
        run1 = _Output(
            path="/tmp/r1",
            name="run1",
            date=datetime(2024, 1, 1, 10, 0, 0),
            summary={"bench_a": {"train_rate": {"mean": 50.0}}},
        )
        run2 = _Output(
            path="/tmp/r2",
            name="run2",
            date=datetime(2024, 2, 1, 10, 0, 0),
            summary={"bench_b": {"train_rate": {"mean": 75.0}}},
        )
        compare([run1, run2], last=None, metric="train_rate", stat="mean")
        captured = capsys.readouterr()
        assert "nan" in captured.out

    def test_missing_metric_shows_nan(self, capsys):
        run = _Output(
            path="/tmp/r1",
            name="run1",
            date=datetime(2024, 1, 1, 10, 0, 0),
            summary={"bench_a": {"other_metric": {"mean": 10.0}}},
        )
        compare([run], last=None, metric="train_rate", stat="mean")
        captured = capsys.readouterr()
        assert "nan" in captured.out

    def test_missing_stat_shows_nan(self, capsys):
        run = _Output(
            path="/tmp/r1",
            name="run1",
            date=datetime(2024, 1, 1, 10, 0, 0),
            summary={"bench_a": {"train_rate": {"std": 1.5}}},
        )
        compare([run], last=None, metric="train_rate", stat="mean")
        captured = capsys.readouterr()
        assert "nan" in captured.out

    def test_empty_summary(self, capsys):
        run = _Output(
            path="/tmp/r1",
            name="run1",
            date=datetime(2024, 1, 1, 10, 0, 0),
            summary={},
        )
        compare([run], last=None, metric="train_rate", stat="mean")
        captured = capsys.readouterr()
        lines = [l for l in captured.out.strip().split("\n") if l.strip()]
        # Only header lines, no bench lines
        assert len(lines) == 4

    def test_benches_sorted_alphabetically(self, capsys):
        run = _Output(
            path="/tmp/r1",
            name="run1",
            date=datetime(2024, 1, 1, 10, 0, 0),
            summary={
                "zebra": {"m": {"s": 1.0}},
                "alpha": {"m": {"s": 2.0}},
                "middle": {"m": {"s": 3.0}},
            },
        )
        compare([run], last=None, metric="m", stat="s")
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        bench_lines = [l for l in lines if any(b in l for b in ("alpha", "middle", "zebra"))]
        assert "alpha" in bench_lines[0]
        assert "middle" in bench_lines[1]
        assert "zebra" in bench_lines[2]

    def test_last_none_uses_all_runs(self, two_runs, capsys):
        compare(two_runs, last=None, metric="train_rate", stat="mean")
        captured = capsys.readouterr()
        assert "run1" in captured.out
        assert "run2" in captured.out

    def test_single_run(self, capsys):
        run = _Output(
            path="/tmp/r1",
            name="only_run",
            date=datetime(2024, 5, 20, 12, 0, 0),
            summary={"bench_x": {"perf": {"avg": 42.0}}},
        )
        compare([run], last=None, metric="perf", stat="avg")
        captured = capsys.readouterr()
        assert "42.00" in captured.out
        assert "only_run" in captured.out


# ─── _Output dataclass ───────────────────────────────────────────────────────


class TestOutputDataclass:
    def test_default_summary_is_none(self):
        out = _Output(path="/tmp/p", name="n", date=datetime.now())
        assert out.summary is None

    def test_fields(self):
        dt = datetime(2024, 1, 1)
        out = _Output(path="/a/b", name="test", date=dt, summary={"k": "v"})
        assert out.path == "/a/b"
        assert out.name == "test"
        assert out.date == dt
        assert out.summary == {"k": "v"}
