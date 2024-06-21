# This is the script run by milabench run (by default)

# It is possible to use a script from a GitHub repo if it is cloned using
# clone_subtree in the benchfile.py, in which case this file can simply
# be deleted.

import time

import torchcompat.core as accelerator
from benchmate.observer import BenchObserver


def main():
    device = accelerator.fetch_device(0) # <= This is your cuda device
    
    observer = BenchObserver(batch_size_fn=lambda batch: 1)
    # optimizer = observer.optimizer(optimizer)
    # criterion = observer.optimizer(criterion)
    
    dataloader = [1, 2, 3, 4]
    
    for epoch in range(10):
        for i in observer.iterate(dataloader):
            # avoid .item()
            # avoid torch.cuda; use accelerator from torchcompat instead
            # avoid torch.cuda.synchronize or accelerator.synchronize
            
            # y = model(i)
            # loss = criterion(y)
            # loss.backward()
            # optimizer.step()
            
            time.sleep(0.1)


if __name__ == "__main__":
    main()
