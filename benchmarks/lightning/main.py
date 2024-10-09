#!/usr/bin/env python


import argparse
import os

os.environ["PT_HPU_LAZY_MODE"] = str(int(int(os.getenv("WORLD_SIZE", -1)) <= 0))

from habana_frameworks.torch import hpu; hpu.init()

import torch
import torch.nn.functional as F
import lightning as L
import torchvision.models as torchvision_models

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



def prepare_voir():
    from benchmate.observer import BenchObserver
    from benchmate.monitor import bench_monitor
    import torchcompat.core as accelerator
    observer = BenchObserver(
        accelerator.Event, 
        earlystop=100,
        batch_size_fn=lambda x: len(x[0]),
        raise_stop_program=False,
        stdout=True,
    )

    return observer, bench_monitor

def main():
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", 1))
    
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--model", type=str, help="torchvision model name", required=True
    )
    dataloader_arguments(parser)
    args = parser.parse_args()
    model = getattr(torchvision_models, args.model)()

    import torchcompat.core as accelerator
  
    n = accelerator.device_count()
    n = local_world_size
    nnodes = world_size // local_world_size

    model = TorchvisionLightning(model)

    accelerator.set_enable_tf32(True)

    observer, monitor = prepare_voir()
    loader = observer.loader(imagenet_dataloader(args, model, rank, world_size))

    # train model
    trainer = L.Trainer(
        accelerator="auto", 
        devices=n, 
        num_nodes=nnodes, 
        strategy="auto",
        max_epochs=args.epochs,
        precision="bf16-mixed",
        enable_checkpointing=False,
        enable_progress_bar=False,
        reload_dataloaders_every_n_epochs=1,
        max_steps=120
    )

    with monitor(poll_interval=0.1):
        trainer.fit(model=model, train_dataloaders=loader)
    print("finished: ", rank)


if __name__ == "__main__":
    main()
