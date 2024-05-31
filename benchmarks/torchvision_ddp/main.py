#!/usr/bin/env python

import argparse
import os
import sys
import json
import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed import init_process_group, destroy_process_group

import torchvision.transforms as transforms
import torchvision.models as torchvision_models
import torchvision.datasets as datasets

import voir
from voir.wrapper import DataloaderWrapper, StopProgram
from voir.smuggle import SmuggleWriter
from giving import give, given
import torchcompat.core as accelerator


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["ID"] = str(rank)
    accelerator.init_process_group(backend=accelerator.ccl, rank=rank, world_size=world_size)
    accelerator.set_device(rank)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        world_size: int
    ) -> None:
        self.rank = gpu_id
        self.device = accelerator.fetch_device(gpu_id)
        self.model = model.to(self.device)
        self.train_data = DataloaderWrapper.with_sumggler(
            train_data, 
            accelerator.Event,
            rank=self.rank,
            device=self.device,
            earlystop=60,
            raise_stop_program=True,
            batch_size_fn=lambda x: len(x[0])
        )
        self.optimizer = optimizer
        # self.model = FSDP(model, device_id=self.device)
        self.model = DDP(model, device_ids=[self.device])
        self.world_size = world_size

    def print(self, *args, **kwargs):
        if self.rank == 0:
            print(*args, **kwargs)

    def _run_batch(self, source, targets):
        with accelerator.amp.autocast(dtype=torch.bfloat16):
            self.optimizer.zero_grad()
            output = self.model(source)
            loss = F.cross_entropy(output, targets)

            loss.backward()
            accelerator.mark_step()

            self.optimizer.step()
            accelerator.mark_step()

            return loss.detach()

    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.device)
            targets = targets.to(self.device)

            loss = self._run_batch(source, targets)
            self.train_data.add_loss(loss)

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)


def image_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return data_transforms

def prepare_dataloader(dataset: Dataset, args):
    dsampler = DistributedSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers if not args.noio else 0,
        pin_memory=not args.noio,
        shuffle=False,
        sampler=dsampler
    )

class FakeDataset:
    def __init__(self, args):
        self.data = [
            (torch.randn((3, 224, 224)), i % 1000) for i in range(60 * args.batch_size)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def dataset(args):
    if args.noio:
        return FakeDataset(args)
    else:
        data_directory = os.environ.get("MILABENCH_DIR_DATA", None)
        if args.data is None and data_directory:
            args.data = os.path.join(data_directory, "FakeImageNet")

        return datasets.ImageFolder(os.path.join(args.data, "train"), image_transforms())


def load_train_objs(args):
    train = dataset(args)

    model = getattr(torchvision_models, args.model)()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    return train, model, optimizer


def worker_main(rank: int, world_size: int, args):
    try:
        print(f">>> rank: {rank}")

        ddp_setup(rank, world_size)

        dataset, model, optimizer = load_train_objs(args)
        train_data = prepare_dataloader(dataset, args)

        trainer = Trainer(model, train_data, optimizer, rank, world_size)

        trainer.train(args.epochs)
        
        destroy_process_group()

        print(f"<<< rank: {rank}")
    except StopProgram:
        print("Early stopping")
    except Exception as err:
        print(err)


def main():
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--batch-size', default=512, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument(
        "--model", type=str, help="torchvision model name", default="resnet50"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="number of workers for data loading",
    )
    parser.add_argument(
        "--noio",
        action='store_true',
        default=False,
        help="Disable IO by providing an in memory dataset",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "fp32", "tf32", "tf32-fp16"],
        default="fp32",
        help="Precision configuration",
    )
    parser.add_argument("--data", type=str, help="data directory")
    args = parser.parse_args()
    
    world_size = accelerator.device_count()

    #
    # This is not voir friendly as it does not allow voir to hook itself
    # to the process
    mp.spawn(
        worker_main,
        args=(
            world_size, 
            args
        ), 
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
