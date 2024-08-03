import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchcompat.core as accelerator
from bench.models import models
from pcqm4m_subset import PCQM4Mv2Subset
from torch_geometric.loader import DataLoader

from benchmate.observer import BenchObserver


def parser():
    parser = argparse.ArgumentParser(description="Geometric GNN")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="Number of epochs to train (default: 20)",
    )
    parser.add_argument("--model", type=str, help="GNN name", required=True)
    parser.add_argument(
        "--num-samples",
        type=str,
        help="Number of samples to process in the dataset",
        default=10000,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 0.0001)",
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
        default=0,
        help="number of workers for data loading",
    )
    return parser


def main():
    args = parser().parse_args()

    observer = BenchObserver(batch_size_fn=lambda batch: 1)

    train_dataset = PCQM4Mv2Subset(args.num_samples, os.environ["MILABENCH_DIR_DATA"])

    print(train_dataset.mean().shape)

    TRAIN_mean, TRAIN_std = (
        train_dataset.mean().item(),
        train_dataset.std().item(),
    )
    print("Train mean: {}\tTrain std: {}".format(TRAIN_mean, TRAIN_std))

    DataLoaderClass = DataLoader
    dataloader_kwargs = {}

    train_loader = DataLoaderClass(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        **dataloader_kwargs
    )

    info = models[args.model](args)

    device = accelerator.fetch_device(0)
    model = info.model.to(device)

    print(model)

    criterion = nn.L1Loss()

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = [{"params": model.parameters(), "lr": args.lr}]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=0)

    lr_scheduler = None
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    num_batches = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_acc = 0

        for step, batch in enumerate(observer.iterate(train_loader)):
            batch = batch.to(device)

            molecule_3D_repr = model(batch.x, batch.pos, batch.batch)

            pred = molecule_3D_repr.squeeze()

            B = pred.size()[0]
            y = batch.y.view(B, -1)
            # normalize
            y = (y - TRAIN_mean) / TRAIN_std

            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_acc += loss.cpu().detach().item()

            lr_scheduler.step(epoch - 1 + step / num_batches)

            observer.record_loss(loss.item())

        loss_acc /= num_batches
        lr_scheduler.step()

        print("Epoch: {}\nLoss: {}".format(epoch, loss_acc))


if __name__ == "__main__":
    main()
