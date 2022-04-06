import argparse
import os
import contextlib

import torch
import torch.cuda.amp
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as tvmodels
import torchvision.transforms as transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


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


def train_epoch(model, criterion, optimizer, loader, device, scaler=None):
    for inp, target in loader:
        inp = inp.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        with scaling(scaler is not None):
            output = model(inp)
            loss = criterion(output, target)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()


def main():
    parser = argparse.ArgumentParser(description="Torchvision models")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
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
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
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
    parser.add_argument("--data", type=str, help="data directory")
    parser.add_argument(
        "--synthetic-data", action="store_true", help="whether to use synthetic data"
    )
    parser.add_argument(
        "--with-amp",
        action="store_true",
        help="whether to use mixed precision with amp",
    )

    args = parser.parse_args()

    if args.synthetic_data:
        args.data = None
    else:
        if not args.data:
            data_directory = os.environ.get("MILABENCH_DIR_DATA", None)
            if data_directory:
                args.data = os.path.join(data_directory, "FakeImageNet")

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = getattr(tvmodels, args.model)()
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr)

    if args.data:
        train = datasets.ImageFolder(os.path.join(args.data, "train"), data_transforms)
    else:
        train = datasets.FakeData(
            size=1000,
            image_size=(3, 512, 512),
            num_classes=1000,
            transform=data_transforms,
        )

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
    )

    if args.with_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(args.epochs):
        train_epoch(model, criterion, optimizer, train_loader, device, scaler=scaler)


if __name__ == "__main__":
    main()
