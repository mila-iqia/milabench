import os
import time
from unittest.mock import MagicMock, patch

import pytest

from benchmate.timings import StepTimer, getenv, total_observations


class TestGetenv:
    def test_returns_default_when_var_not_set(self):
        assert getenv("NONEXISTENT_VAR_XYZ", int, 42) == 42

    def test_returns_converted_value_when_var_set(self, monkeypatch):
        monkeypatch.setenv("TEST_GETENV_VAR", "10")
        assert getenv("TEST_GETENV_VAR", int, 0) == 10

    def test_returns_default_on_conversion_error(self, monkeypatch):
        monkeypatch.setenv("TEST_GETENV_VAR", "not_an_int")
        assert getenv("TEST_GETENV_VAR", int, 99) == 99

    def test_float_conversion(self, monkeypatch):
        monkeypatch.setenv("TEST_GETENV_VAR", "3.14")
        assert getenv("TEST_GETENV_VAR", float, 0.0) == pytest.approx(3.14)

    def test_string_type(self, monkeypatch):
        monkeypatch.setenv("TEST_GETENV_VAR", "hello")
        assert getenv("TEST_GETENV_VAR", str, "") == "hello"


class TestTotalObservations:
    def test_default_values(self, monkeypatch):
        monkeypatch.delenv("VOIR_EARLYSTOP_COUNT", raising=False)
        monkeypatch.delenv("VOIR_EARLYSTOP_SKIP", raising=False)
        assert total_observations() == 65  # 60 + 5

    def test_custom_count(self, monkeypatch):
        monkeypatch.setenv("VOIR_EARLYSTOP_COUNT", "100")
        monkeypatch.delenv("VOIR_EARLYSTOP_SKIP", raising=False)
        assert total_observations() == 105  # 100 + 5

    def test_custom_skip(self, monkeypatch):
        monkeypatch.delenv("VOIR_EARLYSTOP_COUNT", raising=False)
        monkeypatch.setenv("VOIR_EARLYSTOP_SKIP", "10")
        assert total_observations() == 70  # 60 + 10

    def test_both_custom(self, monkeypatch):
        monkeypatch.setenv("VOIR_EARLYSTOP_COUNT", "20")
        monkeypatch.setenv("VOIR_EARLYSTOP_SKIP", "3")
        assert total_observations() == 23


class TestStepTimer:
    def make_timer(self, sync=None):
        pusher = MagicMock()
        sync_fn = sync if sync is not None else MagicMock()
        timer = StepTimer(pusher, sync=sync_fn)
        return timer, pusher, sync_fn

    def test_init_defaults(self):
        pusher = MagicMock()
        timer = StepTimer(pusher)
        assert timer.n_size == 0
        assert timer.n_obs == 0
        assert timer.timesteps == 0
        assert timer.pusher is pusher

    def test_step_accumulates_size(self):
        timer, pusher, _ = self.make_timer()
        timer.step(16)
        timer.step(8)
        assert timer.n_size == 24

    def test_end_calls_sync(self):
        timer, pusher, sync_fn = self.make_timer()
        timer.step(10)
        timer.end()
        sync_fn.assert_called_once()

    def test_end_pushes_rate(self):
        timer, pusher, _ = self.make_timer()
        timer.step(32)
        timer.end()

        rate_call = pusher.call_args_list[0]
        kwargs = rate_call.kwargs
        assert kwargs["units"] == "items/s"
        assert kwargs["task"] == "train"
        assert kwargs["rate"] > 0

    def test_end_pushes_progress(self):
        timer, pusher, _ = self.make_timer()
        timer.step(1)
        timer.end()

        progress_call = pusher.call_args_list[1]
        kwargs = progress_call.kwargs
        assert kwargs["task"] == "early_stop"
        progress = kwargs["progress"]
        assert progress[0] == 0  # n_obs before increment
        assert progress[1] == timer.total_obs

    def test_end_increments_n_obs(self):
        timer, pusher, _ = self.make_timer()
        timer.step(1)
        timer.end()
        assert timer.n_obs == 1
        timer.step(1)
        timer.end()
        assert timer.n_obs == 2

    def test_end_resets_n_size(self):
        timer, pusher, _ = self.make_timer()
        timer.step(64)
        timer.end()
        assert timer.n_size == 0

    def test_end_rate_reflects_step_size(self):
        timer, pusher, _ = self.make_timer()

        fake_start = 1000.0
        fake_end = 1001.0
        timer.start_time = fake_start

        with patch("benchmate.timings.time.perf_counter", return_value=fake_end):
            timer.step(100)
            timer.end()

        rate_call = pusher.call_args_list[0]
        assert rate_call.kwargs["rate"] == pytest.approx(100.0)  # 100 items / 1 second

    def test_end_resets_start_time(self):
        timer, pusher, _ = self.make_timer()
        timer.step(1)
        timer.end()
        # start_time should be updated to end_time after end()
        assert timer.start_time == timer.end_time

    def test_log_forwards_kwargs_to_pusher(self):
        timer, pusher, _ = self.make_timer()
        timer.log(loss=0.5, acc=0.9)
        pusher.assert_called_once_with(loss=0.5, acc=0.9)

    def test_multiple_steps_between_ends(self):
        timer, pusher, _ = self.make_timer()

        fake_start = 1000.0
        fake_end = 1002.0
        timer.start_time = fake_start

        with patch("benchmate.timings.time.perf_counter", return_value=fake_end):
            timer.step(16)
            timer.step(16)
            timer.step(16)
            timer.end()

        rate_call = pusher.call_args_list[0]
        assert rate_call.kwargs["rate"] == pytest.approx(24.0)  # 48 items / 2 seconds

    def test_default_sync_is_callable(self):
        """StepTimer can be created without explicit sync and end() works."""
        pusher = MagicMock()
        timer = StepTimer(pusher)
        timer.step(10)
        timer.end()  # should not raise
        assert timer.n_obs == 1
