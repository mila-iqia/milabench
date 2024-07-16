#!/usr/bin/env python

import argparse
import os

import torch
import torch.nn.functional as F
import lightning as L
import torchvision.models as torchvision_models

import torchcompat.core as accelerator
from benchmate.dataloader import imagenet_dataloader, dataloader_arguments


def criterion():
    return F.cross_entropy


class TorchvisionLightning(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = criterion()

    def training_step(self, batch, batch_idx):
        x, y = batch
        p = self.model(x)
        loss = self.criterion(p, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    dataloader_arguments(parser)
    args = parser.parse_args()
    model = getattr(torchvision_models, args.model)()

    rank = os.getenv("RANK", 0)
    world_size = os.getenv("WORLD_SIZE", 1)
    local_world_size = os.getenv("LOCAL_WORLD_SIZE", 1)

    n = accelerator.device_count()
    nnodes = world_size // local_world_size
    dataset = imagenet_dataloader(args, model, rank, world_size)

    model = TorchvisionLightning(model)

    # train model
    trainer = L.Trainer(
        accelerator="auto", 
        devices=n, 
        num_nodes=nnodes, 
        strategy="ddp",
        max_epochs=args.epochs,
        precision=16
    )
    trainer.fit(model=model, train_dataloaders=dataset)


if __name__ == "__main__":
    main()
