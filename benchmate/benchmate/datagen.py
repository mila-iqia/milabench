#!/usr/bin/env python

import argparse
from collections import defaultdict
import multiprocessing
import os
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')

import torch

from tqdm import tqdm



class FakeInMemoryDataset:
    def __init__(self, producer, batch_size, batch_count):
        self.data = [producer(i) for i in range(batch_size * batch_count)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class FakeImageClassification(FakeInMemoryDataset):
    def __init__(self, shape, batch_size, batch_count):
        def producer(i):
            return (torch.randn(shape), i % 1000)

        super().__init__(producer, batch_size, batch_count)



def write(args):
    import torch 
    import torchvision.transforms as transforms

    offset, outdir, size  = args

    img = torch.randn(*size)
    target = torch.randint(0, 1000, size=(1,), dtype=torch.long)[0]
    img = transforms.ToPILImage()(img)

    class_val = int(target)
    image_name = f"{offset}.jpeg"

    path = os.path.join(outdir, str(class_val))
    os.makedirs(path, exist_ok=True)

    image_path = os.path.join(path, image_name)
    img.save(image_path)


def generate(image_size, n, outdir, start = 0):
    work_items = []
    for i in range(n):
        work_items.append([
            start + i,
            outdir,
            image_size,
        ])

    n_worker = min(multiprocessing.cpu_count(), 8)
    with multiprocessing.Pool(n_worker) as pool:
        for _ in tqdm(pool.imap_unordered(write, work_items), total=n):
            pass


def count_images(path):
    count = defaultdict(int)
    for root, _, files in tqdm(os.walk(path)):
        split = root.split('/')[-2]
        count[split] += len(files)

    return count


def generate_sets(root, sets, shape):
    """When experimenting we want that prepare check that the generated dataset is of the right size"""
    root = Path(root)
    sentinel = root / "done"

    total_images = count_images(root)

    for split, count in sets.items():
        current_count = total_images.get(split, 0)

        if current_count < count:
            print(f"Generating {split} (current {current_count}) (target: {count})")
            generate(shape, count - current_count, os.path.join(root, split), start=current_count)

    with open(sentinel, "w") as fp:
        json.dump(sets, fp)


def generate_fakeimagenet():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--batch-count", default=60, type=int)
    parser.add_argument("--image-size", default=[3, 384, 384], type=int, nargs="+")
    parser.add_argument("--val", default=0.1, type=float, nargs="+")
    parser.add_argument("--test", default=0.1, type=float, nargs="+")
    args, _ = parser.parse_known_args()

    data_directory = os.environ["MILABENCH_DIR_DATA"]
    dest = os.path.join(data_directory, "FakeImageNet")
    print(f"Generating fake data into {dest}...")

    total_images = args.batch_size * args.batch_count
    size_spec = {
        "train": total_images, 
        "val": int(total_images * args.val), 
        "test": int(total_images * args.test)
    }

    generate_sets(dest, size_spec, args.image_size)
    print("Done!")


if __name__ == "__main__":
    generate_fakeimagenet()