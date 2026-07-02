import asyncio
import time
import warnings
from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from milabench.alt_async import (
    FeedbackEventLoop,
    _loop_type,
    feedback_runner,
    proceed,
    send,
)


class TestLoopType:
    """Covers lines 17-20: module-level _loop_type resolution."""

    def test_loop_type_is_valid(self):
        assert _loop_type is not None
        assert issubclass(_loop_type, asyncio.BaseEventLoop)

    def test_feedback_loop_inherits_loop_type(self):
        assert issubclass(FeedbackEventLoop, _loop_type)


class TestFeedbackEventLoopInit:
    def test_default_state(self):
        loop = FeedbackEventLoop()
        try:
            assert isinstance(loop._yieldable_messages, deque)
            assert loop._yieldable_generators == []
            assert loop._multiplexers == []
            assert loop._callbacks == []
        finally:
            loop.close()

    def test_queue_message(self):
        loop = FeedbackEventLoop()
        try:
            loop.queue_message("a")
            loop.queue_message("b")
            assert list(loop._yieldable_messages) == ["a", "b"]
        finally:
            loop.close()


class TestCallback:
    """Covers line 124: _callback dispatches to registered callbacks."""

    def test_no_callbacks(self):
        loop = FeedbackEventLoop()
        try:
            assert loop._callback("msg") == "msg"
        finally:
            loop.close()

    def test_single_callback(self):
        loop = FeedbackEventLoop()
        try:
            received = []
            loop._callbacks.append(lambda m: received.append(m))
            result = loop._callback("hello")
            assert result == "hello"
            assert received == ["hello"]
        finally:
            loop.close()

    def test_multiple_callbacks(self):
        loop = FeedbackEventLoop()
        try:
            r1, r2 = [], []
            loop._callbacks.append(lambda m: r1.append(m))
            loop._callbacks.append(lambda m: r2.append(m))
            loop._callback("x")
            assert r1 == ["x"]
            assert r2 == ["x"]
        finally:
            loop.close()


class TestSendFunction:
    """Covers line 216: send() with a non-FeedbackEventLoop warns."""

    def test_send_with_feedback_loop(self):
        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            async def do_send():
                await send("payload")

            gen = loop.run_until_complete(do_send())
            results = list(gen)
            assert "payload" in results
        finally:
            loop.close()

    def test_send_with_standard_loop_warns(self):
        async def do_send():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                await send("nope")
                assert len(w) == 1
                assert "Could not send message" in str(w[0].message)

        asyncio.run(do_send())


class TestRunUntilComplete:
    """Covers lines 46-82 including exception branches (70-76, 80)."""

    def test_simple_coroutine_returns_value(self):
        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            async def answer():
                return 42

            gen = loop.run_until_complete(answer())
            result = None
            try:
                for _ in gen:
                    pass
            except StopIteration as e:
                result = e.value
            else:
                result = None
            # Generator might return via StopIteration or just end
            # The generator protocol: list(gen) captures yielded items,
            # and the return value of run_until_complete is the future result.
            # Re-run cleanly:
        finally:
            loop.close()

        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            async def answer():
                return 42

            gen = loop.run_until_complete(answer())
            yielded = []
            return_val = None
            while True:
                try:
                    yielded.append(next(gen))
                except StopIteration as stop:
                    return_val = stop.value
                    break
            assert return_val == 42
        finally:
            loop.close()

    def test_exception_in_coroutine(self):
        """Covers lines 70-76: exception branch in run_until_complete."""
        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            async def boom():
                raise ValueError("kaboom")

            gen = loop.run_until_complete(boom())
            with pytest.raises(ValueError, match="kaboom"):
                for _ in gen:
                    pass
        finally:
            loop.close()

    def test_base_exception_in_coroutine(self):
        """Covers lines 70-76: BaseException (new_task=True, future.done(), not cancelled)."""
        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            async def interrupt():
                raise KeyboardInterrupt()

            gen = loop.run_until_complete(interrupt())
            with pytest.raises(KeyboardInterrupt):
                for _ in gen:
                    pass
        finally:
            loop.close()

    def test_loop_stopped_before_future_done(self):
        """Covers line 80: RuntimeError when loop stops prematurely."""
        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            async def hang():
                await asyncio.sleep(999)

            loop.call_soon(loop.stop)
            gen = loop.run_until_complete(hang())
            with pytest.raises(RuntimeError, match="Event loop stopped before Future completed"):
                for _ in gen:
                    pass
        finally:
            loop.close()


class TestProceedMethod:
    """Covers lines 127-158: proceed() on the loop, including
    error propagation through multiplexers (130-158)."""

    def test_proceed_simple(self):
        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            async def simple():
                return "ok"

            gen = loop.proceed(simple())
            results = list(gen)
        finally:
            loop.close()

    def test_proceed_yields_messages(self):
        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            async def chatty():
                await send("m1")
                await send("m2")

            gen = loop.proceed(chatty())
            results = list(gen)
            assert "m1" in results
            assert "m2" in results
        finally:
            loop.close()

    @patch("milabench.alt_async.destroy")
    def test_proceed_exception_with_multiplexers(self, mock_destroy):
        """Covers lines 130-158: error/end events yielded per process."""
        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            mock_proc = MagicMock()
            info_dict = {"task": "bench1"}
            mock_mx = MagicMock()
            mock_mx.processes = {
                mock_proc: (
                    ("stdin", "stdout"),        # streams
                    ["python", "train.py"],      # argv
                    info_dict,                   # info
                ),
            }
            mock_mx.constructor.side_effect = lambda **kw: kw

            loop._multiplexers.append(mock_mx)

            async def explode():
                raise RuntimeError("gpu on fire")

            gen = loop.proceed(explode())
            results = []
            with pytest.raises(RuntimeError, match="gpu on fire"):
                for item in gen:
                    results.append(item)

            error_events = [
                r for r in results
                if isinstance(r, dict) and r.get("event") == "error"
            ]
            end_events = [
                r for r in results
                if isinstance(r, dict) and r.get("event") == "end"
            ]
            assert len(error_events) == 1
            assert error_events[0]["data"]["type"] == "RuntimeError"
            assert error_events[0]["data"]["message"] == "gpu on fire"
            assert error_events[0]["task"] == "bench1"

            assert len(end_events) == 1
            assert end_events[0]["data"]["command"] == ["python", "train.py"]
            assert end_events[0]["data"]["return_code"] == "ERROR"

            mock_destroy.assert_called_once_with(mock_proc)
        finally:
            loop.close()

    @patch("milabench.alt_async.destroy")
    def test_proceed_exception_multiple_processes(self, mock_destroy):
        """Multiple processes in multiplexer each get error/end events."""
        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            proc_a = MagicMock(name="proc_a")
            proc_b = MagicMock(name="proc_b")
            mock_mx = MagicMock()
            mock_mx.processes = {
                proc_a: ((), ["cmd_a"], {"rank": 0}),
                proc_b: ((), ["cmd_b"], {"rank": 1}),
            }
            mock_mx.constructor.side_effect = lambda **kw: kw

            loop._multiplexers.append(mock_mx)

            async def fail():
                raise ValueError("oops")

            gen = loop.proceed(fail())
            results = []
            with pytest.raises(ValueError, match="oops"):
                for item in gen:
                    results.append(item)

            error_events = [r for r in results if isinstance(r, dict) and r.get("event") == "error"]
            end_events = [r for r in results if isinstance(r, dict) and r.get("event") == "end"]
            assert len(error_events) == 2
            assert len(end_events) == 2
            assert mock_destroy.call_count == 2
        finally:
            loop.close()

    @patch("milabench.alt_async.destroy")
    def test_proceed_exception_with_callbacks(self, mock_destroy):
        """Callbacks receive error/end messages during proceed exception."""
        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            cb_received = []
            loop._callbacks.append(lambda m: cb_received.append(m))

            mock_proc = MagicMock()
            mock_mx = MagicMock()
            mock_mx.processes = {mock_proc: ((), ["run"], {"name": "t"})}
            mock_mx.constructor.side_effect = lambda **kw: kw
            loop._multiplexers.append(mock_mx)

            async def fail():
                raise RuntimeError("err")

            gen = loop.proceed(fail())
            with pytest.raises(RuntimeError):
                list(gen)

            assert len(cb_received) == 2
        finally:
            loop.close()


class TestModuleProceed:
    """Covers lines 205-208: module-level proceed() function."""

    def test_proceed_creates_loop_and_yields(self):
        async def greet():
            loop = asyncio.get_running_loop()
            assert isinstance(loop, FeedbackEventLoop)
            await send("hi")

        gen = proceed(greet())
        results = list(gen)
        assert "hi" in results

    def test_proceed_with_return_value(self):
        async def compute():
            return 99

        gen = proceed(compute())
        list(gen)


class TestFeedbackRunner:
    """Covers lines 161-175: feedback_runner decorator."""

    def test_wraps_generator_into_async(self):
        @feedback_runner
        def gen_func():
            yield "x"
            yield "y"
            return "done"

        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            gen = loop.run_until_complete(gen_func())
            results = list(gen)
            assert "x" in results
            assert "y" in results
        finally:
            loop.close()

    def test_none_yield_triggers_await(self):
        @feedback_runner
        def gen_func():
            yield None
            yield "after"
            return "fin"

        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            gen = loop.run_until_complete(gen_func())
            results = list(gen)
            assert "after" in results
        finally:
            loop.close()

    def test_passes_args_kwargs(self):
        @feedback_runner
        def gen_func(a, b, c=10):
            yield a + b + c
            return a * b

        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            gen = loop.run_until_complete(gen_func(1, 2, c=3))
            results = list(gen)
            assert 6 in results  # 1 + 2 + 3
        finally:
            loop.close()

    def test_preserves_function_name(self):
        @feedback_runner
        def my_special_func():
            yield 1

        assert my_special_func.__name__ == "my_special_func"

    def test_empty_generator(self):
        @feedback_runner
        def gen_func():
            return "empty"
            yield  # noqa: unreachable — makes it a generator

        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            gen = loop.run_until_complete(gen_func())
            results = list(gen)
        finally:
            loop.close()


class TestRunFunction:
    """Covers line 199: destroy called on 'stop' event."""

    @patch("milabench.alt_async.destroy")
    @patch("milabench.alt_async.voir_run")
    def test_stop_event_triggers_destroy(self, mock_voir_run, mock_destroy):
        normal_entry = MagicMock()
        normal_entry.event = "data"

        stop_entry = MagicMock()
        stop_entry.event = "stop"

        mock_proc = MagicMock()
        mock_mx = MagicMock()
        mock_mx.processes = {mock_proc: ((), ["cmd"], {})}
        mock_mx.__iter__ = MagicMock(return_value=iter([normal_entry, stop_entry]))
        mock_voir_run.return_value = mock_mx

        from milabench.alt_async import run

        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            gen = loop.run_until_complete(run(["echo", "hi"], info={}))
            results = list(gen)
            mock_destroy.assert_called_once_with(mock_proc)
        finally:
            loop.close()

    @patch("milabench.alt_async.destroy")
    @patch("milabench.alt_async.voir_run")
    def test_no_stop_event_no_destroy(self, mock_voir_run, mock_destroy):
        entry = MagicMock()
        entry.event = "data"

        mock_mx = MagicMock()
        mock_mx.processes = {}
        mock_mx.__iter__ = MagicMock(return_value=iter([entry]))
        mock_voir_run.return_value = mock_mx

        from milabench.alt_async import run

        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            gen = loop.run_until_complete(run(["echo"], info={}))
            list(gen)
            mock_destroy.assert_not_called()
        finally:
            loop.close()

    @patch("milabench.alt_async.voir_run")
    def test_process_accumulator(self, mock_voir_run):
        mock_proc = MagicMock()
        mock_mx = MagicMock()
        mock_mx.processes = {mock_proc: ()}
        mock_mx.__iter__ = MagicMock(return_value=iter([]))
        mock_voir_run.return_value = mock_mx

        from milabench.alt_async import run

        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            accumulator = []
            gen = loop.run_until_complete(
                run(["cmd"], process_accumulator=accumulator, info={})
            )
            list(gen)
            assert mock_proc in accumulator
        finally:
            loop.close()

    @patch("milabench.alt_async.voir_run")
    def test_setsid_flag(self, mock_voir_run):
        mock_proc = MagicMock()
        mock_mx = MagicMock()
        mock_mx.processes = {mock_proc: ()}
        mock_mx.__iter__ = MagicMock(return_value=iter([]))
        mock_voir_run.return_value = mock_mx

        from milabench.alt_async import run

        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            gen = loop.run_until_complete(run(["cmd"], setsid=True, info={}))
            list(gen)
            assert mock_proc.did_setsid is True
        finally:
            loop.close()


class TestRunOnceYieldsMessages:
    def test_queued_messages_yielded(self):
        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            async def enqueue():
                loop.queue_message("q1")
                loop.queue_message("q2")
                await asyncio.sleep(0)

            gen = loop.run_until_complete(enqueue())
            results = list(gen)
            assert "q1" in results
            assert "q2" in results
        finally:
            loop.close()


class TestLoopSendMethod:
    def test_send_queues_and_yields(self):
        loop = FeedbackEventLoop()
        asyncio.set_event_loop(loop)
        try:
            async def multi_send():
                await loop.send("s1")
                await loop.send("s2")

            gen = loop.proceed(multi_send())
            results = list(gen)
            assert "s1" in results
            assert "s2" in results
        finally:
            loop.close()
