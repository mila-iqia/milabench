import argparse
import contextlib
import os
import time

import torch
import torch.cuda.amp
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as tvmodels
import torchvision.transforms as transforms
import voir
from giving import give, given
from cantilever.core.timer import timeit

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class Stats:
    def __init__(self):
        self.count = 0
        self.epoch_count = 0
        
    def newbatch(self, bs):
        self.count += bs.shape[0]
        self.epoch_count += bs.shape[0]
        
    def newepoch(self):
        self.epoch_count = 0
        


stats = Stats()


def is_tf32_allowed(args):
    return "tf32" in args.precision


def is_fp16_allowed(args):
    return "fp16" in args.precision


data_transforms = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)


@contextlib.contextmanager
def scaling(enable):
    if enable:
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


def train_epoch(model, criterion, optimizer, loader, device, scaler=None, timer=None):
    global stats
    
    stats.newepoch()
    model.train()

    s = time.time()
    p = time.time()
    
    # this is what computes the batch size
    for inp, target in voir.iterate("train", loader, True):
        
        with timer.timeit("batch"):
            inp = inp.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            
            with scaling(scaler is not None):
                output = model(inp)
                loss = criterion(output, target)
                # this force sync
                # give(loss=loss.item())

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            torch.cuda.synchronize()


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


def main():
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

    args = parser.parse_args()
    if args.fixed_batch:
        args.synthetic_data = True

    if args.synthetic_data:
        args.data = None
    else:
        if not args.data:
            data_directory = os.environ.get("MILABENCH_DIR_DATA", None)
            if data_directory:
                args.data = os.path.join(data_directory, "FakeImageNet")

    if not args.no_cuda:
        assert torch.cuda.is_available(), "Why is CUDA not available"

    use_cuda = not args.no_cuda

    torch.manual_seed(args.seed)
    if use_cuda:
        if is_tf32_allowed(args):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = getattr(tvmodels, args.model)()
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr)

    if args.data:
        train = datasets.ImageFolder(os.path.join(args.data, "train"), data_transforms)
        train_loader = torch.utils.data.DataLoader(
            train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    else:
        train_loader = SyntheticData(
            model=model,
            device=device,
            batch_size=args.batch_size,
            n=1000,
            fixed_batch=args.fixed_batch,
        )

    if is_fp16_allowed(args):
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    with timeit("train") as train_timer:
        with given() as gv:
            if not args.no_stdout:
                gv.where("loss").display()

            for epoch in voir.iterate("main", range(args.epochs)):
                
                with train_timer.timeit("epoch") as epoch_timer:
                    model.train()
                    es = time.time()
                    
                    if not args.no_stdout:
                        print(f"Begin training epoch {epoch}/{args.epochs}")
                    
                    train_epoch(
                        model, criterion, optimizer, train_loader, device, scaler=scaler, timer=epoch_timer
                    )
                        
    train_timer.show()

if __name__ == "__main__":
    main()
