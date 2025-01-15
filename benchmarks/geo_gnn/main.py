import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchcompat.core as accelerator
from bench.models import models
from pcqm4m_subset import PCQM4Mv2Subset
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool

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
        try:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))
        except TypeError:
            pass

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        try:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        except TypeError:
            pass

    return deg


def mean(self):
    import numpy as np
    return np.mean([self.get(i).y for i in range(len(self))])

def std(self):
    import numpy as np
    return np.std([self.get(i).y for i in range(len(self))])


def find_bad_sample(args):
    if not args.use3d:
        return 

    DataLoaderClass = DataLoader
    dataloader_kwargs = {}

    train_dataset = PCQM4Mv2Subset(
        args.num_samples, 
        args.root,
    )
    
    train_loader = DataLoaderClass(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        **dataloader_kwargs,
    )
    
    sample = next(iter(train_dataset))

    info = models[args.model](
        args,
        sample=sample,
        degree=lambda: degree,
    )

    device = accelerator.fetch_device(0)
    model = info.model.to(device)
    
    from tqdm import tqdm
    bad = []
    for step, batch in enumerate(tqdm(train_loader)):
        batch = batch.to(device)

        try:
            with torch.no_grad():
                molecule_repr = model(z=batch.z, pos=batch.pos, batch=batch.batch)
        except AssertionError:
            print(step)
            bad.append(step)
            
    with open("bad.txt", "a") as fp:
        fp.write(f"{bad}\n")

    print(bad)

# those sample cause an assertion error inside torch-cluster
BAD_SAMPLES = [23847, 33055, 33056, 57908, 57909, 86016, 86488, 86872, 87929, 96913]

def main():
    args = parser().parse_args()

    # find_bad_sample(args)

    def batch_size(x):
        # assert len(x.batch.unique()) == int(x.batch[-1] - x.batch[0] + 1)
        return int(x.batch[-1] - x.batch[0] + 1)

    observer = BenchObserver(batch_size_fn=batch_size)

    bad_samples = []
    if args.use3d:
        bad_samples = BAD_SAMPLES

    train_dataset = PCQM4Mv2Subset(args.num_samples, args.root, bad_samples=bad_samples)
    degree = train_degree(train_dataset)

    sample = next(iter(train_dataset))

    info = models[args.model](
        args,
        sample=sample,
        degree=lambda: degree,
    )

    TRAIN_mean, TRAIN_std = (
        mean(train_dataset).item(),
        std(train_dataset).item(),
    )

    DataLoaderClass = DataLoader
    dataloader_kwargs = {}

    train_loader = DataLoaderClass(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        **dataloader_kwargs,
    )

    device = accelerator.fetch_device(0)
    model = info.model.to(device)

    criterion = nn.L1Loss()

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = [{"params": model.parameters(), "lr": args.lr}]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=0)

    lr_scheduler = None
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    num_batches = len(train_loader)
    loader = observer.loader(train_loader)

    model.train()  # No eval ever.
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(loader):
            batch = batch.to(device)

            if args.use3d:
                molecule_repr = model(z=batch.z, pos=batch.pos, batch=batch.batch)
            else:
                molecule_repr = model(
                    x=batch.x.type(torch.float),
                    batch=batch.batch,
                    edge_index=batch.edge_index,
                    batch_size=batch_size(batch),
                )
                molecule_repr = global_max_pool(molecule_repr, batch.batch)

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

        if loader.is_done():
            break


if __name__ == "__main__":
    main()
