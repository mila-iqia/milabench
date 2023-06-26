import asyncio
import os
import signal
import sys
import threading
import time
from asyncio import events, futures, tasks
from asyncio.base_events import _run_until_complete_cb
from collections import deque
from functools import wraps
import warnings


from voir.proc import run as voir_run


class FeedbackEventLoop(type(asyncio.get_event_loop())):
    """Fork of the default event loop that can be iterated upon.

    Iterating over `run(...)` will run the loop and yield logs or events
    (mostly) as they happen. Use the `feedback_runner` decorator to change
    an iterable into an awaitable that will yield all of the iterated values
    from the `run(...)` call.
    """

    def __init__(self):
        super().__init__()
        self._yieldable_messages = deque()
        self._yieldable_generators = []
        self._multiplexers = []
        self._callbacks = []

    def queue_message(self, message):
        self._yieldable_messages.append(message)

    async def send(self, message):
        self.queue_message(message)
        await asyncio.sleep(0)

    def run_until_complete(self, future):
        """Run until the Future is done.

        If the argument is a coroutine, it is wrapped in a Task.

        WARNING: It would be disastrous to call run_until_complete()
        with the same coroutine twice -- it would wrap it in two
        different Tasks and that can't be good.

        Return the Future's result, or raise its exception.
        """
        self._check_closed()
        self._check_running()

        new_task = not futures.isfuture(future)
        future = tasks.ensure_future(future, loop=self)
        if new_task:
            # An exception is raised if the future didn't complete, so there
            # is no need to log the "destroy pending task" message
            future._log_destroy_pending = False

        future.add_done_callback(_run_until_complete_cb)
        try:
            yield from self.run_forever()
        except:
            if new_task and future.done() and not future.cancelled():
                # The coroutine raised a BaseException. Consume the exception
                # to not log a warning, the caller doesn't have access to the
                # local task.
                future.exception()
            raise
        finally:
            future.remove_done_callback(_run_until_complete_cb)
        if not future.done():
            raise RuntimeError("Event loop stopped before Future completed.")

        return future.result()

    def run_forever(self):
        """Run until stop() is called."""
        self._check_closed()
        self._check_running()
        self._set_coroutine_origin_tracking(self._debug)
        self._thread_id = threading.get_ident()

        old_agen_hooks = sys.get_asyncgen_hooks()
        sys.set_asyncgen_hooks(
            firstiter=self._asyncgen_firstiter_hook,
            finalizer=self._asyncgen_finalizer_hook,
        )
        try:
            events._set_running_loop(self)
            while True:
                yield from self._run_once()
                if self._stopping:
                    break
        finally:
            self._stopping = False
            self._thread_id = None
            events._set_running_loop(None)
            self._set_coroutine_origin_tracking(False)
            sys.set_asyncgen_hooks(*old_agen_hooks)

    def _run_once(self):
        """Run one full iteration of the event loop.

        This calls all currently ready callbacks, polls for I/O,
        schedules the resulting callbacks, and finally schedules
        'call_later' callbacks.
        """

        while self._yieldable_messages:
            yield self._callback(self._yieldable_messages.popleft())

        super()._run_once()


    def _callback(self, msg):
        for callback in self._callbacks:
            callback(msg)
        return msg
    
    def proceed(self, coro):
        try:
            yield from self.run_until_complete(coro)
        except BaseException as exc:
            for mx in self._multiplexers:
                for proc, (streams, argv, info) in mx.processes.items():
                    yield self._callback(mx.constructor(
                        event="error",
                        data={
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                        **info,
                    ))
                    destroy(proc)
                    yield self._callback(mx.constructor(
                        event="end",
                        data={
                            "command": argv,
                            "time": time.time(),
                            "return_code": "ERROR",
                        },
                        **info,
                    ))
            raise


def feedback_runner(gen):
    @wraps(gen)
    async def wrapped(*args, **kwargs):
        loop = asyncio.get_running_loop()
        g = gen(*args, **kwargs)
        while True:
            try:
                while (x := next(g)) is not None:
                    loop.queue_message(x)
                await asyncio.sleep(0)
            except StopIteration as stop:
                await asyncio.sleep(0)
                return stop.value

    return wrapped


def destroy(*processes):
    for proc in processes:
        if getattr(proc, "did_setsid", False):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.kill()
        # TODO: send SIGKILL after a certain time if the process is not dead


@feedback_runner
def run(argv, setsid=None, info={}, process_accumulator=None, **kwargs):
    if setsid:
        kwargs["preexec_fn"] = os.setsid
    mx = voir_run(argv, info=info, **kwargs, timeout=0)
    if process_accumulator is not None:
        process_accumulator.extend(mx.processes)
    if setsid:
        for proc in mx.processes:
            proc.did_setsid = True
    loop = asyncio.get_running_loop()
    loop._multiplexers.append(mx)
    for entry in mx:
        if entry and entry.event == "stop":
            destroy(*mx.processes)
        yield entry


def proceed(coro):
    loop = FeedbackEventLoop()
    asyncio.set_event_loop(loop)
    yield from loop.proceed(coro)


async def send(message):
    loop = asyncio.get_running_loop()
    if isinstance(loop, FeedbackEventLoop):
        await loop.send(message)
    else:
        warnings.warn("Could not send message, wrong event loop used")
