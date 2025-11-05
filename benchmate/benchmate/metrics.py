from dataclasses import dataclass
import json
import os
import sys
import time
from contextlib import contextmanager
from threading import get_native_id

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

    print("No overseer found")
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
            data = self.materialize(*args, **kwargs)
            pusher(**data)
        self.delayed = []


class LazyLossPusher(LazyMetricPusher):
    def record(self, loss):
        value = loss
        # no .item() we do not want to sync
        if hasattr(loss, "detach"):
            value = loss.detach()
        self.append(value, time.time())

    def materialize(self, loss, timing=None):
        value = loss

        # synch here is fine
        if hasattr(loss, "item"):
            value = loss.item()

        return {"loss": value, "task": self.task, "time": timing}


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
    
    def synchronize(self):
        pass


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


def default_event():
    try:
        import torchcompat.core as accelerator
        return accelerator.Event
    except:
        print("Could not find a device timer")
        return CPUTimer()


def default_device():
    try:
        import torchcompat.core as accelerator
        return accelerator.fetch_device(int(os.getenv("LOCAL_RANK", 0)))
    except:
        print("Could not find a device")
        return None


def sync(device):
    try:
        import torchcompat.core as accelerator
        accelerator.synchronize(device)
    except:
        print("Could not find a device")
        return None

@contextmanager
def record_timing(event_fn):
    start = event_fn(enable_timing=True)
    end = event_fn(enable_timing=True)
    start.record()

    yield TimedEvent(None, start, end)

    end.record()
    end.synchronize()


@contextmanager
def lazy_record_timing(name, event_fn):
    start = event_fn(enable_timing=True)
    end = event_fn(enable_timing=True)

    start.record()
    yield TimedEvent(name, start, end)
    end.record()


@dataclass
class TimedEvent:
    name: str
    start: 'Event'
    end: 'Event'
    batch_size: int = None
    batch_id: int = None

    def elapsed_time(self):
        self.end.synchronize()
        return self.start.elapsed_time(self.end)


@dataclass
class LapEvent:
    name: str
    time: float


def _get_flag(name, type, default):
    return type(os.getenv(name, default))


@dataclass
class Flags:
    record_laps: int = _get_flag("BENCHMATE_RECORD_LAPS", int, 1)
    record_overhead: int = _get_flag("BENCHMATE_RECORD_OVERHEAD", int, 0)
    record_fine_grained: int  = _get_flag("BENCHMATE_RECORD_FINE", int, 0)
    eager_sync: int  = _get_flag("BENCHMATE_EAGER_SYNC", int, 0)

_flags = Flags()


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
    TIMED_ITERATOR_INSTANCE = 0
    ACTIVE_WRAPPED_ITERATOR = 0

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
        event_fn=None,
        rank=int(os.getenv("RANK", 0)),
        push=file_push(),
        device=default_device(),
        earlystop=earlystop_count(),
        raise_stop_program=False,
        batch_size_fn=None,
    ):
        self.recored_laps = [LapEvent('first', time.time())]
        self.loader = loader  # original iterator
        self.events: list[TimedEvent] = []  # accumulated events to be pushed

        self.task = "train"  # voir task usually train but could be validation/test
        self.total_obs = 0  # Number of "pushed" observations
        self.event_fn = event_fn  # function to create a device event
        self.early_stop = earlystop  # Number of observation to target
        self.unit = 1000  # device timer is ms
        self.last_time = time.time()
        self.message_push = push  # How to push the metrics usually voir or stdout

        # Number of times we broke out of the iterator for early stopping
        # we should really only do this onece
        self.break_count = 0
        self.batch_size_fn = batch_size_fn

        # Multi-GPU setup
        self.rank = rank
        self.device = device

        # Options
        self.raise_stop_program = (
            raise_stop_program  # Does TimedIterator raise StopProgram
        )
        self.overhead = []
        self.previous_overhead = 0
        self.loader_init_time = []
        self.recorded_timers = []

        self.unique_iterator = 0
        self.multi_call_guard = 0
        
        self.sub_overhead = 0
        self.batch_id = 0

        if event_fn is None:
            self.event_fn = default_event()

        # assert TimedIterator.TIMED_ITERATOR_INSTANCE == 0, f"One timed iterator only {TimedIterator.TIMED_ITERATOR_INSTANCE}"
        # TimedIterator.TIMED_ITERATOR_INSTANCE += 1

        if not TORCH_ERROR and dist.is_initialized():
            self.rank = rank
            assert (
                self.device is not None
            ), "device is required to compute the final batch size"

    def record_lap(self, name):
        """A lap event is a unique event that gets recorded when the code execute a particular path.
        This allow us to compute the lap time.
        """
        if _flags.record_laps:
            self.recored_laps.append(LapEvent(name, time.time()))

    def __getattr__(self, item):
        return getattr(self.loader, item)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.record_lap("iter_create")
        self.log_progress()

        # This takes much more time than expected good thing to keep track of it
        with CPUTimer() as ct:
            iterator = iter(self.loader)

        self.unique_iterator = id(iterator)
        self.loader_init_time.append(ct.elapsed())
        return self.wrapped(iterator)

    def _record_event(self, event: TimedEvent):
        """A TimedEvent is something that records a particular code duration"""
        if _flags.record_fine_grained:
            event.batch_id = self.batch_id
            self.recorded_timers.append(event)

    def wrapped(self, iterator):
        # Time IO wait + batch compute
        self.record_lap("iter_start")
        self.last_time = time.time()
        self.start = self.event_fn(enable_timing=True)
        self.start.record()
        self.previous_overhead = 0

        try:
            while True:
                assert self.unique_iterator == id(iterator)

                try:
                    with lazy_record_timing("next", self.event_fn) as next_time:
                        data = next(iterator)

                    # Simple one iteration = one backward
                    # ... huggingface ... is changing the batch sometimes...
                    bs = self.deduce_batch_size(data)

                    with lazy_record_timing("work", self.event_fn) as work_time: 
                        yield data

                    with lazy_record_timing("step", self.event_fn) as step_time: 
                        if should_break := self.step(bs):
                            self.break_count += 1
                            break
                        
                    self._record_event(next_time)
                    self._record_event(work_time)
                    self._record_event(step_time)
                    self.batch_id += 1
                
                except StopIteration:
                    break
        finally:
            self.multi_call_guard -= 1
            self.unique_iterator = None
            self.record_lap("iter_end")
            self._push()
            self._push_total_time_elapsed()
            self.earlystop()

    def _make_rate(self, event: TimedEvent):
        event.end.synchronize()
        sync(self.device)

        elapsed = (event.elapsed_time()) / self.unit
        rate = self.batch_size(event.batch_size) / elapsed

        return rate, elapsed

    def step(self, batch_size):
        should_break = False

        with CPUTimer() as ct:
            end = self.event_fn(enable_timing=True)
            end.record()

            # We could use event.query() to push events without blocking
            event = TimedEvent("rate", self.start, end, batch_size, self.batch_id)
            
            if _flags.eager_sync:
                print("HERE")
                rate, elapsed = self._make_rate(event)
                self.log_rate(rate, time=time.time(), elapsed=elapsed, batch_id=event.batch_id)
                self.total_obs += 1
            else:
                self.events.append(event)

            # Log progress so it looks somewhat responsive
            self.log_progress()

            # check for early stopping to avoid doing the full epoch
            if self.is_done() and self.break_count == 0:
                should_break = True

            self.start = end

        # Note: first step does not have overhead because end event is recorded
        # before the overhead starts
        # Note: It is not sure if the CPU overhead impacst the device at all
        # since we avoid sync it is possible the device is working during
        # the overhead section and that the effective overhead ends up being minimal
        self.previous_overhead = ct.elapsed()
       
        if _flags.record_overhead:
            self.overhead.append(self.previous_overhead)
    
        return should_break

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
        for event in self.events:
            rate, elapsed = self._make_rate(event)
            self.last_time += elapsed
            self.log_rate(rate, time=self.last_time, elapsed=elapsed, batch_id=event.batch_id)

        self.total_obs += len(self.events)
        self.events = []

    def _push_profile_metrics(self):
        for ov in self.overhead:
            self.message(overhead=ov, units="s", task=self.task)

        for iterinit in self.loader_init_time:
            self.message(__iter__=iterinit, units="s", task=self.task)
        
        for event in self.recorded_timers:
            kwargs = {
                event.name: event.elapsed_time() / self.unit,
                "units": "s",
                "task": self.task,
                "batch_id": event.batch_id
            }
            self.message(**kwargs)

        self.previous_overhead = 0
        self.overhead = []
        self.loader_init_time = []
        self.recorded_timers = []

    def __del__(self):
        try:
            if self.events:
                self._push()

        except ValueError as err:
            print("Some events could not be pushed because: ", str(err))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._push()

    def _push_lap_events(self):
        start = self.recored_laps[0]
        for lap in self.recored_laps[1:]:
            self.message(**{lap.name: lap.time - start.time}, task=self.task, units="s", time=lap.time)
        self.recored_laps = [start]

    def _push_total_time_elapsed(self):
        self.message(total_elapsed=time.time() - self.recored_laps[0].time, task=self.task, units="s")

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

            self._push_lap_events()

        if False:
            self.message(sync_time=sync_time.elapsed(), units="s", task=self.task)
            self.message(process_time=process_time.elapsed(), units="s", task=self.task)

    def log_rate(self, rate, time=None, elapsed=None, batch_id=None):
        self.message(
            rate=rate, units="items/s", task=self.task,
            time=time, elapsed=elapsed, batch_id=batch_id
        )

    def log_progress(self):
        if self.early_stop is not None:
            progress = self.progress()
            self.message(progress=(progress, self.early_stop), task="early_stop")

    def message(self, **kwargs):
        if self.rank is None or self.rank == 0:
            self.message_push(**kwargs)


class ManualTimedIterator(TimedIterator):
    """Used when the model is trained with graident accumulation steps.
    
    Users need to call iterator.step() manually.


    Examples
    --------


    .. code-block:: python

        for e in range(epochs):
            for j, _ in enumerate(loader):
                if (j + 1) % accumulation_steps == 0:
                    # work here
                    # ...
                    #
                    loader.step()

    """
    def __init__(self, loader, event_fn=default_event(), rank=int(os.getenv("RANK", 0)), push=file_push(), device=default_device(), earlystop=earlystop_count(), raise_stop_program=False, batch_size_fn=None):
        super().__init__(loader, event_fn, rank, push, device, earlystop, raise_stop_program, batch_size_fn)
        self.acc_batch_size = 0
        self.should_stop = False
        self.accumulation_steps = 0
        self.epochs = 0

    def step(self, batch_override=None):
        if batch_override is None:
            batch_override = self.acc_batch_size

        self.should_stop = super().step(batch_override)
        self.acc_batch_size = 0
        self.accumulation_steps = 0
        return self.should_stop
       
    def wrapped(self, iterator):
        # Time IO wait + batch compute
        self.start = self.event_fn(enable_timing=True)
        self.start.record()
        self.previous_overhead = 0
        
        for data in iterator:
            # we have to compute the batch size now because 
            # step might be call before the execution returns to this  block
            self.accumulation_steps += 1
            self.acc_batch_size += self.deduce_batch_size(data)
          
            yield data # step will probably be called right before the control
                       # returns to this block

            # step was called, should we stop ?
            if self.should_stop:
                break

        self._push()
        self.earlystop()
        self.epochs += 1