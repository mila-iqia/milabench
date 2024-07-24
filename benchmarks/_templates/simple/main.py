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


def main():
    device = accelerator.fetch_device(0) # <= This is your cuda device

    observer = BenchObserver(
        batch_size_fn=lambda batch: 1
    )
    # optimizer = observer.optimizer(optimizer)
    # criterion = observer.criterion(criterion)
    
    dataloader = list(range(6000))
    
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
