import os

from voir.helpers import current_overseer

from .metrics import LazyLossPusher, TimedIterator, ManualTimedIterator, give_push, sumggle_push


class BenchObserver:
    """Helper class to create override function for ptera

    Examples
    --------

    .. code-block::


       observer = BenchObserver()

       probe = ov.probe("//dataloader() as loader", overridable=True)
       probe['loader'].override(observer.loader)

       probe = ov.probe("//train_epoch > criterion", overridable=True)
       probe['criterion'].override(observer.criterion)

    """

    def __init__(
        self,
        *args,
        backward_callback=None,
        step_callback=None,
        stdout=False,
        rank=int(os.getenv("RANK", 0)),
        **kwargs,
    ):
        self.wrapped = None
        self.args = args
        self.kwargs = kwargs
        self.backward_callback = backward_callback
        self.optimizer_step_callback = step_callback
        self.stdout = stdout
        self.task = "train"
        self.rank = rank
        self.losses = LazyLossPusher(self.task)
        self.instance = None
        self.pusher = give_push()
        if self.stdout:
            self.pusher = sumggle_push()

    def on_iterator_stop_iterator(self):
        """Called when the timed iterator stops, used to do extra work when timers are off"""
        self.losses.push(self.pusher)

    def record_loss(self, loss):
        if self.rank is None or self.rank == 0:
            self.losses.record(loss)
        return loss

    def override_return_value(self, function, override):
        import ptera

        refstring = ptera.refstring(function)

        ov = current_overseer.get()

        if ov is not None:
            probe = ov.probe(f"{refstring}() as retval", overridable=True)
            probe["retval"].override(override)
        else:
            raise RuntimeError("Not running through voir")

    def iterate(self, iterator, custom_step=False):
        return self.loader(iterator, custom_step=custom_step)
    
    def step(self):
        self.instance.step()
    
    def original_dataloader(self):
        return self.instance
    
    def loader(self, loader, custom_step=False):
        """Wrap a dataloader or an iterable which enable accurate measuring of time spent in the loop's body"""
        if self.instance is not None:
            return self.instance.loader
        
        cls = TimedIterator
        if custom_step:
            cls = ManualTimedIterator
        
        self.wrapped = cls(
            loader, *self.args, rank=self.rank, push=self.pusher, **self.kwargs
        )
        self.wrapped.task = self.task
        self.wrapped.on_iterator_stop_iterator = self.on_iterator_stop_iterator
        self.instance = self.wrapped
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

            self.record_loss(loss)
            return loss

        return wrapped

    def optimizer(self, optimizer):
        """Wrap an optimizer to enable a .step callback"""
        if self.optimizer_step_callback:
            original = optimizer.step

            def new_step(*args, **kwargs):
                original(*args, **kwargs)
                self.optimizer_step_callback()

            # wow
            # pytorch does
            #   instance_ref = weakref.ref(method.__self__)
            #   where
            #       method == self.optimizer.step
            new_step.__self__ = optimizer.step.__self__
            new_step.__func__ = optimizer.step.__func__
            optimizer.step = new_step
        return optimizer
