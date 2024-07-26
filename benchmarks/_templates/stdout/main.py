# This is the script run by milabench run (by default)

# It is possible to use a script from a GitHub repo if it is cloned using
# clone_subtree in the benchfile.py, in which case this file can simply
# be deleted.

import time
import random

import torchcompat.core as accelerator
from benchmate.observer import BenchObserver


def criterion(*args, **kwargs):
    return random.normalvariate(0, 1)


def prepare_voir():
    from benchmate.observer import BenchObserver
    from benchmate.monitor import bench_monitor

    observer = BenchObserver(
        accelerator.Event, 
        earlystop=65,
        batch_size_fn=lambda x: len(x[0]),
        raise_stop_program=False,
        stdout=True,
    )

    return observer, bench_monitor


def main():
    device = accelerator.fetch_device(0) # <= This is your cuda device

    observer, monitor = prepare_voir()

    # optimizer = observer.optimizer(optimizer)
    
    dataloader = list(range(6000))
    
    with monitor():
        for epoch in range(10000):
            for i in observer.iterate(dataloader):
                # avoid .item()
                # avoid torch.cuda; use accelerator from torchcompat instead
                # avoid torch.cuda.synchronize or accelerator.synchronize
                
                # y = model(i)
                loss = criterion()
                # loss.backward()
                # optimizer.step()

                observer.record_loss(loss)
                
                time.sleep(0.1)

    assert epoch < 2, "milabench stopped the train script before the end of training"
    assert i < 72, "milabench stopped the train script before the end of training"


if __name__ == "__main__":
    main()
