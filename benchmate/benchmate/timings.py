import os
import time


def getenv(name, type, default):
    try:
        return type(os.getenv(name, default))
    except:
        return default


def total_observations():
    return (
        getenv("VOIR_EARLYSTOP_COUNT", int, 60) +
        getenv("VOIR_EARLYSTOP_SKIP", int, 5) 
    )


class StepTimer:
    """

    Examples
    --------

    .. code-block:: python

       step_timer = StepTimer()
       for i in range(epochs):
       
           for i, batch in enumerate(data):
               step_timer.step(batch.shape[0])
               step_timer.log(loss=...)

               if (i + 1) % grad_acc == 0:
                   optimizer.step()
                   step_timer.end()
           
    """
    def __init__(self, pusher, sync = lambda: None):
        self.start_time = time.perf_counter()
        self.end_time = 0
        self.n_size = 0
        self.n_obs = 0
        self.total_obs = total_observations()
        self.pusher = pusher
        self.sync = sync
        self.timesteps = 0

    def step(self, step_size):
        """Log a batch size or work that was been done"""
        self.n_size += step_size

    def end(self):
        """Push a new perf observation"""
        self.sync()
        self.end_time = time.perf_counter()
        self.pusher(rate=self.n_size/(self.end_time - self.start_time), units="items/s", task="train")
        self.pusher(progress=(self.n_obs, self.total_obs), task="early_stop")
        self.size = 0
        self.n_obs += 1
        self.start_time = self.end_time

    def log(self, **kwargs):
        self.pusher(**kwargs)
