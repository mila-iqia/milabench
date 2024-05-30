#!/usr/bin/env python

import argparse
import multiprocessing
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm


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


def generate(image_size, n, outdir):
    work_items = []
    for i in range(n):
        work_items.append([
            i,
            outdir,
            image_size,
        ])

    n_worker = min(multiprocessing.cpu_count(), 8)
    with multiprocessing.Pool(n_worker) as pool:
        for _ in tqdm(pool.imap_unordered(write, work_items), total=n):
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
    main()