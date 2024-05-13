import argparse
import contextlib
import os

import torch
import torch.cuda.amp
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as tvmodels
import torchvision.transforms as transforms
import torchcompat.core as accelerator

import voir
from giving import give, given
from cantilever.core.timer import timeit, timeiterator, show_timings

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def is_tf32_allowed(args):
    return "tf32" in args.precision


def is_fp16_allowed(args):
    return "fp16" in args.precision or "bf16" in args.precision


def float_dtype(precision):
    if "fp16" in precision:
        if accelerator.device_type == "cuda":
            return torch.float16
        else:
            return torch.bfloat16
        
    if "bf16" in precision:
        return torch.bfloat16
        
    return torch.float


class NoScale:
    def scale(self, loss):
        return loss
    def step(self, optimizer):
        optimizer.step()
    def update(self):
        pass


data_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)


@contextlib.contextmanager
def scaling(enable, dtype):
    if enable:
        with accelerator.amp.autocast(dtype=dtype):
            yield
    else:
        yield


def train_epoch(model, criterion, optimizer, loader, device, dtype, scaler=None):
    model.train()

    def toiterator(loader):
        with timeit("loader"):
            return iter(loader)
            
    for inp, target in timeiterator(voir.iterate("train", toiterator(loader), True)):
        
        with timeit("batch"):
            inp = inp.to(device, dtype=dtype)
            target = target.to(device)
            optimizer.zero_grad()

            with scaling(scaler is not None, dtype):
                output = model(inp)
                loss = criterion(output, target)
                give(loss=loss.item())

            if scaler:
                scaler.scale(loss).backward()
                accelerator.mark_step()

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                accelerator.mark_step()
                optimizer.step()

            accelerator.mark_step()
            accelerator.synchronize()


class SyntheticData:
    def __init__(self, model, device, batch_size, n, fixed_batch):
        self.n = n
        self.inp = torch.randn((batch_size, 3, 224, 224)).to(device)
        self.out = torch.rand_like(model(self.inp))
        self.fixed_batch = fixed_batch

    def __iter__(self):
        inp, out = self.inp, self.out
        for i in range(self.n):
            if not self.fixed_batch:
                inp = torch.rand_like(self.inp)
                out = torch.rand_like(self.out)
            yield (inp, out)

    def __len__(self):
        return self.n



def dataloader(args):
    train = datasets.ImageFolder(os.path.join(args.data, "train"), data_transforms)
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=torch.utils.data.RandomSampler(
            train, 
            replacement=True, 
            num_samples=len(train) * args.epochs
        )
    )
    return train_loader


def iobench(args):
    data_directory = os.environ.get("MILABENCH_DIR_DATA", None)
    if args.data is None and data_directory:
        args.data = os.path.join(data_directory, "FakeImageNet")
        
    loader = dataloader(args)
    device = accelerator.fetch_device(0)
    dtype = float_dtype(args.precision)

    def toiterator(loader):
        with timeit("loader"):
            return iter(loader)

    with given() as gv:
        for epoch in voir.iterate("main", range(args.epochs)):
            with timeit("epoch"):
                for inp, target in timeiterator(voir.iterate("train", toiterator(loader), True)):
                    with timeit("batch"):
                        inp = inp.to(device, dtype=dtype)
                        target = target.to(device)

                accelerator.synchronize()


def main():
    from voir.phase import StopProgram

    try:
        _main()
    except StopProgram:
        show_timings(True)
        raise

def _main():
    parser = argparse.ArgumentParser(description="Torchvision models")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--model", type=str, help="torchvision model name", required=True
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        metavar="S",
        help="random seed (default: 1234)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="number of workers for data loading",
    )
    parser.add_argument("--data", type=str, help="data directory")
    parser.add_argument(
        "--synthetic-data", action="store_true", help="whether to use synthetic data"
    )
    parser.add_argument(
        "--fixed-batch", action="store_true", help="use a fixed batch for training"
    )
    # parser.add_argument(
    #     "--with-amp",
    #     action="store_true",
    #     help="whether to use mixed precision with amp",
    # )
    parser.add_argument(
        "--no-stdout",
        action="store_true",
        help="do not display the loss on stdout",
    )
    # parser.add_argument(
    #     "--no-tf32",
    #     dest="allow_tf32",
    #     action="store_false",
    #     help="do not allow tf32",
    # )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "fp32", "tf32", "tf32-fp16"],
        default="fp32",
        help="Precision configuration",
    )
    parser.add_argument(
        "--iobench",
        action="store_true",
        default=False,
        help="Precision configuration",
    )

    args = parser.parse_args()

    if args.iobench:
        iobench(args)
    else:
        trainbench()

def trainbench(args):
    if args.fixed_batch:
        args.synthetic_data = True

    if args.synthetic_data:
        args.data = None
    else:
        if not args.data:
            data_directory = os.environ.get("MILABENCH_DIR_DATA", None)
            if data_directory:
                args.data = os.path.join(data_directory, "FakeImageNet")

    torch.manual_seed(args.seed)

    accelerator.set_enable_tf32(is_tf32_allowed(args))
    device = accelerator.fetch_device(0)

    model = getattr(tvmodels, args.model)()
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr)

    model, optimizer = accelerator.optimizer(model, optimizer=optimizer, dtype=float_dtype(args.precision))

    if args.data:
        train_loader = dataloader(args)
    else:
        train_loader = SyntheticData(
            model=model,
            device=device,
            batch_size=args.batch_size,
            n=1000,
            fixed_batch=args.fixed_batch,
        )

    scaler = NoScale()
    if torch.cuda.is_available():
        scaler = accelerator.amp.GradScaler(enabled=is_fp16_allowed(args))

    with given() as gv:
        if not args.no_stdout:
            gv.where("loss").display()

        for epoch in voir.iterate("main", range(args.epochs)):
            with timeit("epoch"):
                if not args.no_stdout:
                    print(f"Begin training epoch {epoch}/{args.epochs}")
                train_epoch(
                    model, criterion, optimizer, train_loader, device, scaler=scaler, dtype=float_dtype(args.precision)
                )


if __name__ == "__main__":
    main()
