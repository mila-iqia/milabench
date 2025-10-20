from dataclasses import dataclass

import torch
import torchcompat.core as accelerator


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


@dataclass
class Arguments:
    pass

def load_model():
    pass


def load_dataset():
    pass


def main(argv):
    global _log 

    observer, monitor = prepare_voir()

    with monitor():
        with torch.no_grad():
            model = load_model()

            dataset = load_dataset()

            for batch in dataset:

                output = model(batch)


if __name__ == "__main__":
    import sys

    main(sys.argv)
