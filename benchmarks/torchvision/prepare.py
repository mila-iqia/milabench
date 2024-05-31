#!/usr/bin/env python

import argparse
import multiprocessing
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm


def write(args):
    offset, (image, y), outdir = args

    class_val = int(y)
    image_name = f"{offset}.jpeg"

    path = os.path.join(outdir, str(class_val))
    os.makedirs(path, exist_ok=True)

    image_path = os.path.join(path, image_name)
    image.save(image_path)


def generate(image_size, n, outdir):
    from torchvision.datasets import FakeData

    dataset = FakeData(
        size=n, 
        image_size=image_size, 
        num_classes=1000, 
        random_offset=0
    )

    def expand(iterable):
        for i, item in enumerate(iterable):
            yield i, item, outdir

    n_worker = multiprocessing.cpu_count()
    with multiprocessing.Pool(n_worker) as pool:
        for _ in tqdm(pool.imap_unordered(write, expand(dataset)), total=n):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--batch-count", default=60, type=int)
    parser.add_argument("--image-size", default=[3, 384, 384], type=int, nargs="+")
    parser.add_argument("--val", default=0.1, type=float, nargs="+")
    parser.add_argument("--test", default=0.1, type=float, nargs="+")
    args = parser.parse_known_args()

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
    main()