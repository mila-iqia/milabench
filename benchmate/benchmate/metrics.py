import json
import os
import sys
import time

from voir.helpers import current_overseer
from voir.phase import StopProgram
from voir.smuggle import SmuggleWriter

TORCH_ERROR = None
try:
    import torch
    import torch.distributed as dist
except ImportError as err:
    TORCH_ERROR = err


def file_push(fp=sys.stdout):
    def message(**kwargs):
        kwargs.setdefault("task", "train")
        msg = json.dumps(kwargs)
        print(msg, file=fp)

    return message


def sumggle_push():
    fp = SmuggleWriter(sys.stdout)
    return file_push(fp)


def is_running_with_voir():
    return current_overseer.get() is not None


def give_push():
    ov = current_overseer.get()

    if ov is not None:
        return ov.give

    return file_push()


def earlystop_count():
    return int(os.getenv("VOIR_EARLYSTOP_COUNT", 60)) + int(
        os.getenv("VOIR_EARLYSTOP_SKIP", 10)
    )


class LazyMetricPusher:
    def __init__(self, task):
        self.task = task
        self.delayed = []

    def append(self, *args, **kwargs):
        self.delayed.append((args, kwargs))

    def record(self, *args, **kwargs):
        """Record data for a future metric.

        No synchronization here.
        """
        self.append(*args, **kwargs)

    def materialize(self, *args, **kwargs):
        """Transform raw data into a metric.

        Synchronization happens here.
        """
        return *args, kwargs

    def push(self, pusher):
        """Iterate through data and push metrics."""
        for args, kwargs in self.delayed:
            pusher(**self.materialize(*args, **kwargs))
        self.delayed = []


class LazyLossPusher(LazyMetricPusher):
    def record(self, loss):
        value = loss
        # no .item() we do not want to sync
        if hasattr(loss, "detach"):
            value = loss.detach()
        self.append(value)

    def materialize(self, loss):
        value = loss
        # synch here is fine
        if hasattr(loss, "item"):
            value = loss.item()

        return {"loss": value, "task": self.task}


class CPUTimer:
    def __init__(self):
        self._start = None
        self._end = None

    def start(self):
        self._start = -time.time()

    def end(self):
        self._end = time.time()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.end()

    def elapsed(self):
        return self._end + self._start


class DeviceTimer:
    def __init__(self, event_fn):
        self._start = event_fn(enable_timing=True)
        self._end = event_fn(enable_timing=True)

    def start(self):
        self._start.record()

    def end(self):
        self._end.record()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.end()

    def elapsed(self):
        self._end.synchronize()
        return self._start.elapsed_time(self._end)


class TimedIterator:
    """Time the body of a loop, ignoring the time it took to initialize the iterator.`
    The timings are measured using `torch.cuda.Event` to avoid explicit sync.

    An explicit sync is done at the end of an epoch or if the max number of observation is reached.

    Because the timings are async, voir only gets triggered when the explicit sync happens which
    is outside of the scope of performance measure, so no matter how long voir takes to process
    the events it will not impact the measures.

    The wrapper also works in multi-gpu, multi-node setups and computes the real batch-size
    by reducing the batch size on all processes. Only rank 0 logs data.

    Notes
    -----
    The event progress is the only one that is feed synchronously so
    this event should be handled quickly.

    Arguments
    ---------
    loader: Dataloader
        original pytorch dataloader

    event_fn:
        event constructor (torch.cuda.Evemt, toch.xpu.Event, etc...)

    rank: int
        rank of the current process, only required if distributed

    device:
        device used, only required if distributed

    earlystop: int
        number of observation to produce before raising StopProgram

    push: function:
        function used to message/push metrics

    Examples
    --------

    .. code-block::

       loader = TimedIterator(loader, torch.cuda.Event, earlystop=60)   # < here

       for e in range(epochs):
           for i in loader:
               loss = criterion(model(x), y)
    """

    @classmethod
    def with_sumggler(cls, *args, push=None, **kwargs):
        return cls(*args, push=sumggle_push(), **kwargs)

    @classmethod
    def with_stdout(cls, *args, push=None, **kwargs):
        return cls(*args, push=file_push(), **kwargs)

    @classmethod
    def with_give(cls, *args, push=None, **kwargs):
        return cls(*args, push=give_push(), **kwargs)

    def __init__(
        self,
        loader,
        event_fn,
        rank=0,
        push=file_push(),
        device=None,
        earlystop=earlystop_count(),
        raise_stop_program=False,
        batch_size_fn=None,
    ):
        self.loader = loader  # original iterator
        self.events = []  # accumulated events to be pushed

        self.task = "train"  # voir task usually train but could be validation/test
        self.total_obs = 0  # Number of "pushed" observations
        self.event_fn = event_fn  # function to create a device event
        self.early_stop = earlystop  # Number of observation to target
        self.unit = 1000  # device timer is ms

        self.message_push = push  # How to push the metrics usually voir or stdout

        # Number of times we broke out of the iterator for early stopping
        # we should really only do this onece
        self.break_count = 0
        self.batch_size_fn = batch_size_fn

        # Multi-GPU setup
        self.rank = None
        self.device = device
        self.world_size = 1

        # Options
        self.raise_stop_program = (
            raise_stop_program  # Does TimedIterator raise StopProgram
        )
        self.profile_instrumentation = False
        self.overhead = []
        self.previous_overhead = 0
        self.loader_init_time = []
        self.sub_overhead = 0

        if not TORCH_ERROR and dist.is_initialized():
            self.rank = rank
            assert (
                self.device is not None
            ), "device is required to compute the final batch size"

    def __getattr__(self, item):
        return getattr(self.loader, item)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.log_progress()

        # This takes much more time than expected good thing to keep track of it
        with CPUTimer() as ct:
            iterator = iter(self.loader)

        self.loader_init_time.append(ct.elapsed())
        return self.wrapped(iterator)

    def wrapped(self, iterator):
        # Time IO wait + batch compute
        start = self.event_fn(enable_timing=True)
        start.record()
        self.previous_overhead = 0

        for data in iterator:
            yield data

            with CPUTimer() as ct:
                end = self.event_fn(enable_timing=True)
                end.record()

                bs = self.deduce_batch_size(data)
                self.events.append((start, end, bs, self.previous_overhead))

                # Log progress so it looks somewhat responsive
                self.log_progress()

                # check for early stopping to avoid doing the full epoch
                if self.is_done() and self.break_count == 0:
                    self.break_count += 1
                    break

                start = end

            # Note: first step does not have overhead because end event is recorded
            # before the overhead starts
            # Note: It is not sure if the CPU overhead impacst the device at all
            # since we avoid sync it is possible the device is working during
            # the overhead section and that the effective overhead ends up being minimal
            self.previous_overhead = ct.elapsed()
            self.overhead.append(self.previous_overhead)

        self._push()
        self.earlystop()

    def deduce_batch_size(self, elem):
        if self.batch_size_fn:
            return self.batch_size_fn(elem)

        try:
            if len(elem) == 2:
                return len(elem[0])
            return len(elem)
        except ValueError:
            return 0

    def progress(self):
        return len(self.events) + self.total_obs

    def is_done(self):
        return self.early_stop is not None and self.progress() >= self.early_stop

    def earlystop(self, exception=StopProgram):
        if self.is_done():
            self._push()

            if self.raise_stop_program:
                raise exception()

    def on_iterator_stop_iterator(self):
        """Extension point called when timers are off"""
        pass

    def batch_size(self, bs):
        # multi GPU, batch size count
        if not TORCH_ERROR and dist.is_initialized():
            bs = torch.tensor([bs], dtype=torch.int64, device=self.device)
            dist.reduce(bs, dst=0)
            return bs.item()
        return bs

    def _push_time_steps(self):
        for start, end, bs, overhead in self.events:
            end.synchronize()
            elapsed = (start.elapsed_time(end)) / self.unit
            rate = self.batch_size(bs) / elapsed
            self.log_rate(rate)

        self.total_obs += len(self.events)
        self.events = []

    def _push_profile_metrics(self):
        if self.profile_instrumentation:
            for ov in self.overhead:
                self.message(overhead=ov, units="s", task=self.task)

            for iterinit in self.loader_init_time:
                self.message(__iter__=iterinit, units="s", task=self.task)
        self.previous_overhead = 0
        self.overhead = []
        self.loader_init_time = []

    def _push(self):
        """Push all the accumulated metrics"""

        with CPUTimer() as sync_time:
            event = self.event_fn()
            event.record()
            event.synchronize()

        with CPUTimer() as process_time:
            self.on_iterator_stop_iterator()

            # Push synchronize to have the final compute times
            self._push_time_steps()

            # Optional
            self._push_profile_metrics()

        self.message(sync_time=sync_time.elapsed(), units="s", task=self.task)
        self.message(process_time=process_time.elapsed(), units="s", task=self.task)

    def log_rate(self, rate):
        self.message(rate=rate, units="items/s", task=self.task)

    def log_progress(self):
        if self.early_stop is not None:
            progress = self.progress()
            self.message(progress=(progress, self.early_stop), task="early_stop")

    def message(self, **kwargs):
        if self.rank is None or self.rank == 0:
            self.message_push(**kwargs)
