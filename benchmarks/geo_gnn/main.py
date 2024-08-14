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
        type=int,
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
    parser.add_argument(
        "--use3d",
        action="store_true",
        default=False,
        help="Use 3D coordinates with data",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.environ["MILABENCH_DIR_DATA"],
        help="Dataset path",
    )
    return parser


def train_degree(train_dataset):
    from torch_geometric.utils import degree

    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    return deg


def main():
    args = parser().parse_args()

    def batch_size(x):
        print(type(x))
        return 1

    observer = BenchObserver(batch_size_fn=batch_size)

    train_dataset = PCQM4Mv2Subset(args.num_samples, args.root)

    info = models[args.model](args, degree=lambda: train_degree(train_dataset))

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

        for step, batch in enumerate(observer.iterate(train_loader)):
            # DataBatch(x=[229, 9], edge_index=[2, 476], edge_attr=[476, 3], y=[16], pos=[229, 3], smiles=[16], batch=[229], ptr=[17])
            batch = batch.to(device)
            
            if args.use3d:
                molecule_repr = model(z=batch.x, pos=batch.pos, batch=batch.batch)
            else:
                molecule_repr = model(x=batch.x, batch=batch.batch, edge_index=batch.edge_index)

            pred = molecule_repr.squeeze()

            B = pred.size()[0]
            y = batch.y.view(B, -1)
            # normalize
            y = (y - TRAIN_mean) / TRAIN_std

            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step(epoch - 1 + step / num_batches)

            observer.record_loss(loss)

        lr_scheduler.step()

        print("Epoch: {}\nLoss: {}".format(epoch))


if __name__ == "__main__":
    main()
