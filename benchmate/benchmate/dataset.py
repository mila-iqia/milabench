import os
from collections import defaultdict
import math

import torch
from torch.utils.data.distributed import DistributedSampler


def no_transform(args):
    return args


def transform_images(transform_x, transform_y=no_transform):
    def _(args):
        print(args)
        return transform_x(args[0]), transform_y(args[1])

    return _


def transform_celebA(transform_x):
    def _(args):
        print(args)
        return transform_x(args["image"])

    return _


class TransformedDataset:
    def __init__(self, dataset, transforms=no_transform):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.transforms(self.dataset[item])


class ImageNetAsFrames:
    def __init__(self, folder) -> None:
        self.clip = defaultdict(list)
        for root, _, files in os.walk(folder):
            clip_id = root.split("/")[-1]
            video = self.clip[clip_id]
            for frame in files:
                video.append(frame)

    def __getitem__(self, item):
        return self.clip[item]

    def __len__(self):
        return len(self.clip)


class ExclusiveSetSampler(DistributedSampler):
    def __init__(self, dataset, num_sets: int, set_id: int, shuffle: bool = True, seed: int = 0, drop_last: bool = False) -> None:
        super().__init__(
            dataset, 
            num_replicas=num_sets,
            rank=set_id,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last        
        )

