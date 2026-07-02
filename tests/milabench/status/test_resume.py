from unittest.mock import MagicMock, patch

import pytest

from milabench.status.resume import (
    _expected_logfiles,
    find_matching_runfolder,
    full_run,
    is_run_complete,
    resume_as_bench_selector,
    resume_from_files,
    schedule_run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pack(name="bench1"):
    """Create a mock pack with the minimal interface needed."""
    pack = MagicMock()
    pack.config = {"name": name, "system": {"gpu": {"count": 1}}}
    return pack


def _make_plan(logfile_names):
    """Create a mock plan with executors whose packs return given logfile names."""
    plan = MagicMock()
    executors = []
    for fname in logfile_names:
        executor = MagicMock()
        logfile_mock = MagicMock()
        logfile_mock.name = fname
        executor.pack.logfile.return_value = logfile_mock
        executors.append(executor)
    plan.executors = executors
    return plan


# ---------------------------------------------------------------------------
# Tests for is_run_complete
# ---------------------------------------------------------------------------

class TestIsRunComplete:
    def test_all_logs_present_and_successful(self, tmp_path):
        logfiles = ["bench1.data", "bench2.data"]
        for f in logfiles:
            (tmp_path / f).write_text("")

        with patch(
            "milabench.report.read.fetch_benchmark_status", return_value="success"
        ):
            assert is_run_complete(str(tmp_path), logfiles) is True

    def test_missing_logfile_returns_false(self, tmp_path):
        logfiles = ["bench1.data", "bench2.data"]
        (tmp_path / "bench1.data").write_text("")

        with patch(
            "milabench.report.read.fetch_benchmark_status", return_value="success"
        ):
            assert is_run_complete(str(tmp_path), logfiles) is False

    def test_failed_status_returns_false(self, tmp_path):
        logfiles = ["bench1.data"]
        (tmp_path / "bench1.data").write_text("")

        with patch(
            "milabench.report.read.fetch_benchmark_status", return_value="failure"
        ):
            assert is_run_complete(str(tmp_path), logfiles) is False

    def test_missing_list_collects_failed_logs(self, tmp_path):
        logfiles = ["bench1.data", "bench2.data"]
        for f in logfiles:
            (tmp_path / f).write_text("")

        missing = []
        with patch(
            "milabench.report.read.fetch_benchmark_status", return_value="failure"
        ):
            result = is_run_complete(str(tmp_path), logfiles, missing=missing)
        assert result is False
        assert missing == ["bench1.data", "bench2.data"]

    def test_partial_failure_collects_only_failed(self, tmp_path):
        logfiles = ["bench1.data", "bench2.data"]
        for f in logfiles:
            (tmp_path / f).write_text("")

        missing = []

        def side_effect(path):
            if "bench1" in path:
                return "success"
            return "failure"

        with patch(
            "milabench.report.read.fetch_benchmark_status", side_effect=side_effect
        ):
            result = is_run_complete(str(tmp_path), logfiles, missing=missing)
        assert result is False
        assert missing == ["bench2.data"]

    def test_empty_logfiles_returns_true(self, tmp_path):
        assert is_run_complete(str(tmp_path), []) is True

    def test_missing_defaults_to_empty_list(self, tmp_path):
        logfiles = ["bench1.data"]
        (tmp_path / "bench1.data").write_text("")

        with patch(
            "milabench.report.read.fetch_benchmark_status", return_value="success"
        ):
            assert is_run_complete(str(tmp_path), logfiles) is True


# ---------------------------------------------------------------------------
# Tests for find_matching_runfolder
# ---------------------------------------------------------------------------

class TestFindMatchingRunfolder:
    def test_finds_single_match(self, tmp_path):
        (tmp_path / "project.run1.2026-01-01_10-00-00").mkdir()
        result = find_matching_runfolder(str(tmp_path), "project.run1.{time}")
        assert result == "project.run1.2026-01-01_10-00-00"

    def test_pattern_substitution(self, tmp_path):
        (tmp_path / "p600.o350.2026-01-07_14-27-23").mkdir()
        result = find_matching_runfolder(str(tmp_path), "p600.o350.{time}")
        assert result == "p600.o350.2026-01-07_14-27-23"

    def test_raises_on_multiple_matches(self, tmp_path):
        (tmp_path / "p600.o350.2026-01-01_10-00-00").mkdir()
        (tmp_path / "p600.o350.2026-01-02_10-00-00").mkdir()
        with pytest.raises(RuntimeError, match="Cannot reusme"):
            find_matching_runfolder(str(tmp_path), "p600.o350.{time}")

    def test_raises_on_no_match(self, tmp_path):
        with pytest.raises(IndexError):
            find_matching_runfolder(str(tmp_path), "nonexistent.{time}")

    def test_multiple_placeholders(self, tmp_path):
        (tmp_path / "proj.run.data").mkdir()
        result = find_matching_runfolder(str(tmp_path), "{name}.{id}.{ext}")
        assert result == "proj.run.data"


# ---------------------------------------------------------------------------
# Tests for _expected_logfiles
# ---------------------------------------------------------------------------

class TestExpectedLogfiles:
    @patch("milabench.status.resume.is_system_capable", return_value=True)
    @patch("milabench.multi.make_execution_plan")
    def test_returns_plans_and_logfiles(self, mock_plan_fn, mock_capable):
        plan = _make_plan(["bench1.D0.data", "bench1.D1.data"])
        mock_plan_fn.return_value = plan

        packs = {"bench1": _make_pack("bench1")}
        result = _expected_logfiles(packs, repeat=1)

        assert len(result) == 1
        assert result[0][0] is plan
        assert result[0][1] == ["bench1.D0.data", "bench1.D1.data"]

    @patch("milabench.status.resume.is_system_capable", return_value=False)
    @patch("milabench.multi.make_execution_plan")
    def test_skips_incapable_systems(self, mock_plan_fn, mock_capable):
        packs = {"bench1": _make_pack("bench1")}
        result = _expected_logfiles(packs, repeat=1)

        assert result == []
        mock_plan_fn.assert_not_called()

    @patch("milabench.status.resume.is_system_capable", return_value=True)
    @patch("milabench.multi.make_execution_plan")
    def test_repeat_multiplies_plans(self, mock_plan_fn, mock_capable):
        plan = _make_plan(["bench1.data"])
        mock_plan_fn.return_value = plan

        packs = {"bench1": _make_pack("bench1")}
        result = _expected_logfiles(packs, repeat=3)

        assert len(result) == 3
        assert mock_plan_fn.call_count == 3

    @patch("milabench.status.resume.is_system_capable", return_value=True)
    @patch("milabench.multi.make_execution_plan")
    def test_multiple_packs(self, mock_plan_fn, mock_capable):
        plan1 = _make_plan(["bench1.data"])
        plan2 = _make_plan(["bench2.data"])
        mock_plan_fn.side_effect = [plan1, plan2]

        packs = {"bench1": _make_pack("bench1"), "bench2": _make_pack("bench2")}
        result = _expected_logfiles(packs, repeat=1)

        assert len(result) == 2


# ---------------------------------------------------------------------------
# Tests for resume_from_files
# ---------------------------------------------------------------------------

class TestResumeFromFiles:
    @patch("milabench.status.resume._expected_logfiles")
    @patch("milabench.status.resume.is_run_complete")
    def test_returns_incomplete_plans(self, mock_complete, mock_expected):
        plan1 = MagicMock(name="plan1")
        plan2 = MagicMock(name="plan2")
        mock_expected.return_value = [
            (plan1, ["f1.data"]),
            (plan2, ["f2.data"]),
        ]
        mock_complete.side_effect = [True, False]

        packs = {"b1": _make_pack(), "b2": _make_pack()}
        result = resume_from_files(packs, "/some/path", repeat=1)

        assert result == [plan2]

    @patch("milabench.status.resume._expected_logfiles")
    @patch("milabench.status.resume.is_run_complete")
    def test_all_complete_returns_empty(self, mock_complete, mock_expected):
        plan1 = MagicMock()
        mock_expected.return_value = [(plan1, ["f1.data"])]
        mock_complete.return_value = True

        result = resume_from_files({"b1": _make_pack()}, "/path", repeat=1)
        assert result == []

    @patch("milabench.status.resume._expected_logfiles")
    @patch("milabench.status.resume.is_run_complete")
    def test_prints_warning_for_repeat_gt_1(self, mock_complete, mock_expected, capsys):
        mock_expected.return_value = []
        resume_from_files({"b1": _make_pack()}, "/path", repeat=2)
        captured = capsys.readouterr()
        assert "repeat > 1 is not supported" in captured.out


# ---------------------------------------------------------------------------
# Tests for full_run
# ---------------------------------------------------------------------------

class TestFullRun:
    @patch("milabench.status.resume._expected_logfiles")
    def test_returns_all_plans(self, mock_expected):
        plan1 = MagicMock()
        plan2 = MagicMock()
        mock_expected.return_value = [(plan1, ["f1.data"]), (plan2, ["f2.data"])]

        packs = {"b1": _make_pack(), "b2": _make_pack()}
        result = full_run(packs, repeat=1)

        assert result == [plan1, plan2]

    @patch("milabench.status.resume._expected_logfiles")
    def test_empty_packs_returns_empty(self, mock_expected):
        mock_expected.return_value = []
        result = full_run({}, repeat=1)
        assert result == []


# ---------------------------------------------------------------------------
# Tests for schedule_run
# ---------------------------------------------------------------------------

class TestScheduleRun:
    @patch("milabench.status.resume.full_run")
    def test_no_resume_calls_full_run(self, mock_full_run, tmp_path):
        mock_full_run.return_value = ["plan1"]
        runfolder = tmp_path / "myrun"
        runfolder.mkdir()

        result = schedule_run({}, str(tmp_path), "myrun", repeat=1, resume=False)
        assert result == ["plan1"]
        mock_full_run.assert_called_once()

    @patch("milabench.status.resume.resume_from_files")
    def test_resume_calls_resume_from_files(self, mock_resume, tmp_path):
        mock_resume.return_value = ["plan1"]
        runfolder = tmp_path / "myrun"
        runfolder.mkdir()

        result = schedule_run({}, str(tmp_path), "myrun", repeat=1, resume=True)
        assert result == ["plan1"]
        mock_resume.assert_called_once()

    @patch("milabench.status.resume.resume_from_files")
    def test_resume_nonexistent_folder_logs_warning(self, mock_resume, tmp_path):
        mock_resume.return_value = []
        result = schedule_run({}, str(tmp_path), "nonexistent", repeat=1, resume=True)
        assert result == []

    @patch("milabench.status.resume.find_matching_runfolder")
    @patch("milabench.status.resume.full_run")
    def test_pattern_in_run_name_resolves(self, mock_full_run, mock_find, tmp_path):
        mock_find.return_value = "resolved_name"
        mock_full_run.return_value = ["plan"]

        result = schedule_run({}, str(tmp_path), "proj.{time}", repeat=1, resume=False)
        mock_find.assert_called_once_with(str(tmp_path), "proj.{time}")
        assert result == ["plan"]

    @patch("milabench.status.resume.find_matching_runfolder")
    @patch("milabench.status.resume.resume_from_files")
    def test_pattern_with_resume(self, mock_resume, mock_find, tmp_path):
        mock_find.return_value = "resolved_run"
        mock_resume.return_value = ["plan_x"]

        result = schedule_run({}, str(tmp_path), "proj.{time}", repeat=1, resume=True)
        mock_find.assert_called_once()
        mock_resume.assert_called_once()
        assert result == ["plan_x"]


# ---------------------------------------------------------------------------
# Tests for resume_as_bench_selector
# ---------------------------------------------------------------------------

class TestResumeAsBenchSelector:
    @patch("milabench.status.resume.is_system_capable", return_value=True)
    @patch("milabench.multi.make_execution_plan")
    @patch("milabench.report.read.fetch_benchmark_status", return_value="success")
    def test_filters_completed_packs(self, mock_status, mock_plan_fn, mock_capable, tmp_path):
        plan = _make_plan(["bench1.data"])
        mock_plan_fn.return_value = plan

        runfolder = tmp_path / "myrun"
        runfolder.mkdir()
        (runfolder / "bench1.data").write_text("")

        packs = {"bench1": _make_pack("bench1")}
        result = resume_as_bench_selector(packs, str(tmp_path), "myrun")
        assert result == {}

    @patch("milabench.status.resume.is_system_capable", return_value=True)
    @patch("milabench.multi.make_execution_plan")
    @patch("milabench.report.read.fetch_benchmark_status", return_value="failure")
    def test_keeps_incomplete_packs(self, mock_status, mock_plan_fn, mock_capable, tmp_path):
        plan = _make_plan(["bench1.data"])
        mock_plan_fn.return_value = plan

        runfolder = tmp_path / "myrun"
        runfolder.mkdir()
        (runfolder / "bench1.data").write_text("")

        pack = _make_pack("bench1")
        packs = {"bench1": pack}
        result = resume_as_bench_selector(packs, str(tmp_path), "myrun")
        assert result == {"bench1": pack}

    @patch("milabench.status.resume.is_system_capable", return_value=False)
    @patch("milabench.multi.make_execution_plan")
    def test_skips_incapable_packs(self, mock_plan_fn, mock_capable, tmp_path):
        packs = {"bench1": _make_pack("bench1")}
        result = resume_as_bench_selector(packs, str(tmp_path), "myrun")
        assert result == {}
        mock_plan_fn.assert_not_called()

    @patch("milabench.status.resume.is_system_capable", return_value=True)
    @patch("milabench.multi.make_execution_plan")
    def test_mixed_completion_status(self, mock_plan_fn, mock_capable, tmp_path):
        plan = _make_plan(["b.data"])
        mock_plan_fn.return_value = plan

        runfolder = tmp_path / "myrun"
        runfolder.mkdir()
        (runfolder / "b.data").write_text("")

        call_count = [0]
        def status_side_effect(path):
            call_count[0] += 1
            if call_count[0] == 1:
                return "success"
            return "failure"

        with patch("milabench.report.read.fetch_benchmark_status", side_effect=status_side_effect):
            pack1 = _make_pack("bench1")
            pack2 = _make_pack("bench2")
            packs = {"bench1": pack1, "bench2": pack2}
            result = resume_as_bench_selector(packs, str(tmp_path), "myrun")
        assert result == {"bench2": pack2}
