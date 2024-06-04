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


def give_push():
    ov = current_overseer.get()
    return ov.give


def earlystop_count():
    return int(os.getenv("VOIR_EARLYSTOP_COUNT", 60)) + int(
        os.getenv("VOIR_EARLYSTOP_SKIP", 10)
    )


class DataloaderWrapper:
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

       loader = DataloaderWrapper(loader, torch.cuda.Event, earlystop=60)   # < here

       for e in range(epochs):
           for i in loader:
               loss = criterion(model(x), y)

               loader.add_loss(loss)                                        # < here
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
        self.loader = loader
        self.events = []
        self.losses = []
        self.overhead = []
        self.loader_init_time = []

        self.total_obs = 0
        self.event_fn = event_fn
        self.world_size = 1
        self.early_stop = earlystop
        self.rank = None
        self.device = device
        self.datafile = sys.stdout
        self.n = len(loader)
        self.unit = 1000  # timer is ms
        self.profile_instrumentation = False
        self.message_push = push
        self.raise_stop_program = raise_stop_program
        self.break_count = 0
        self.batch_size_fn = batch_size_fn

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
        start = -time.time()
        iterator = iter(self.loader)
        end = time.time()

        self.loader_init_time.append(start + end)
        return self.wrapped(iterator)

    def wrapped(self, iterator):
        # Time IO wait + batch compute
        start = self.event_fn(enable_timing=True)
        start.record()

        for data in iterator:
            yield data

            overhead_start = -time.time()

            end = self.event_fn(enable_timing=True)
            end.record()
            bs = self.deduce_batch_size(data)
            self.events.append((start, end, bs))

            # check for early stopping to avoid doing the full epoch
            self.log_progress()
            if self.is_done() and self.break_count == 0:
                self.break_count += 1
                break

            start = end
            overhead_end = time.time()
            self.overhead.append(overhead_start + overhead_end)

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

    def extra_work(self):
        pass

    def batch_size(self, bs):
        # multi GPU, batch size count
        if not TORCH_ERROR and dist.is_initialized():
            bs = torch.tensor([bs], dtype=torch.int64, device=self.device)
            dist.reduce(bs, dst=0)
            return bs.item()
        return bs

    def _push(self):
        """Push all the accumulated metrics"""
        event = self.event_fn()
        event.record()
        event.synchronize()

        s = -time.time()
        self.extra_work()

        # Push synchronize to have the final compute times
        for start, end, bs in self.events:
            end.synchronize()

            elapsed = start.elapsed_time(end) / self.unit
            rate = self.batch_size(bs) / elapsed

            self.log_rate(rate)

        for loss in self.losses:
            self.log_loss(loss.item())

        if self.profile_instrumentation:
            for ov in self.overhead:
                self.message(overhead=ov, units="s", task="train")

            for iterinit in self.loader_init_time:
                self.message(__iter__=iterinit, units="s", task="train")

        self.total_obs += len(self.events)
        self.events = []
        self.losses = []
        self.overhead = []
        self.loader_init_time = []
        e = time.time()

        self.message(process_time=(e + s), units="s", task="train")

    def add_loss(self, loss):
        # avoid .item() that cause sync
        self.losses.append(loss.detach())
        return loss

    def log_rate(self, rate):
        self.message(rate=rate, units="items/s", task="train")

    def log_loss(self, loss):
        self.message(loss=loss, task="train")

    def log_progress(self):
        if self.early_stop is not None:
            progress = self.progress()
            self.message(progress=(progress, self.early_stop), task="early_stop")

    def message(self, **kwargs):
        if self.rank is None or self.rank == 0:
            self.message_push(**kwargs)


class Wrapper:
    """Helper class to create override function for ptera

    Examples
    --------

    .. code-block::

       probe = ov.probe("//dataloader() as loader", overridable=True)
       probe['loader'].override(wrapper.loader)

       probe = ov.probe("//train_epoch > criterion", overridable=True)
       probe['criterion'].override(wrapper.criterion)

    """

    def __init__(
        self, *args, backward_callback=None, step_callback=None, stdout=False, **kwargs
    ):
        self.wrapped = None
        self.args = args
        self.kwargs = kwargs
        self.backward_callback = backward_callback
        self.optimizer_step_callback = step_callback
        self.stdout = stdout

    def loader(self, loader):
        """Wrap a dataloader or an iterable which enable accurate measuring of time spent in the loop's body"""
        ctor = DataloaderWrapper.with_give
        if self.stdout:
            ctor = DataloaderWrapper.with_sumggler

        self.wrapped = ctor(loader, *self.args, **self.kwargs)
        return self.wrapped

    def criterion(self, criterion):
        """Wrap a loss value to  log and enable a .backward callback"""

        def wrapped(*args):
            loss = criterion(*args)

            if self.backward_callback:
                original = loss.backward

                def new_backward(*args, **kwargs):
                    original(*args, **kwargs)
                    self.backward_callback()

                loss.backward = new_backward

            self.wrapped.add_loss(loss)
            return loss

        return wrapped

    def optimizer(self, optimizer):
        """Wrap an optimizer to enable a .step callback"""
        if self.optimizer_step_callback:
            original = optimizer.step

            def new_step(*args, **kwargs):
                original(*args, **kwargs)
                self.optimizer_step_callback()

            optimizer.step = new_step
        return optimizer
