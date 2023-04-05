#!/usr/bin/env python

import multiprocessing
import os
from pathlib import Path

from tqdm import tqdm


def write(args):
    from torchvision.datasets import FakeData

    image_size, offset, count, outdir = args
    dataset = FakeData(
        size=count, image_size=image_size, num_classes=1000, random_offset=offset
    )

    image, y = next(iter(dataset))
    class_val = int(y)
    image_name = f"{offset}.jpeg"

    path = os.path.join(outdir, str(class_val))
    os.makedirs(path, exist_ok=True)

    image_path = os.path.join(path, image_name)
    image.save(image_path)


def generate(image_size, n, outdir):
    p_count = min(multiprocessing.cpu_count(), 8)
    pool = multiprocessing.Pool(p_count)
    for _ in tqdm(
        pool.imap_unordered(write, ((image_size, i, n, outdir) for i in range(n))),
        total=n,
    ):
        pass


def generate_sets(root, sets, shape):
    root = Path(root)
    sentinel = root / "done"
    if sentinel.exists():
        print(f"{root} was already generated")
        return
    if root.exists():
        print(f"{root} exists but is not marked complete; deleting")
        root.rm()
    for name, n in sets.items():
        print(f"Generating {name}")
        generate(shape, n, os.path.join(root, name))
    sentinel.touch()


if __name__ == "__main__":
    data_directory = os.environ["MILABENCH_DIR_DATA"]
    dest = os.path.join(data_directory, "FakeImageNet")
    print(f"Generating fake data into {dest}...")
    generate_sets(dest, {"train": 4096, "val": 16, "test": 16}, (3, 384, 384))
    print("Done!")
