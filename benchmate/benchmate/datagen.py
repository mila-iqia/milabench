#!/usr/bin/env python

import argparse
import json
import multiprocessing
import os
from collections import defaultdict
from pathlib import Path
import random
import sys

import torchcompat.core as acc
import torch
from benchmate.progress import tqdm


def write(args):
    import torchvision.transforms as transforms

    offset, outdir, prefix, size = args

    seed = int(0 + offset)
    torch.manual_seed(seed)
    random.seed(seed)
    
    img = torch.randint(0, 256, size, dtype=torch.uint8)
    # img = torch.randn(*size)
    
    img = transforms.ToPILImage()(img)
    target = offset % 1000  # torch.randint(0, 1000, size=(1,), dtype=torch.long)[0]
    class_val = int(target)
    
    # Some benches need filenames to match those of imagenet:
    # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/datasets/image_net.py#L40-L43
    if not prefix:  # train
        image_name = f"{class_val}_{offset}"
    else:           # val, test
        image_name = f"{prefix}{int(offset):08d}"

    path = os.path.join(outdir, str(class_val))
    os.makedirs(path, exist_ok=True)

    image_path = os.path.join(path, f"{image_name}.JPEG")
    img.save(image_path)


def generate(image_size, n, outdir, prefix="", start=0):
    work_items = []
    for i in range(n):
        work_items.append(
            [
                start + i,
                outdir,
                prefix,
                image_size,
            ]
        )

    n_worker = min(multiprocessing.cpu_count(), 8)
    with multiprocessing.Pool(n_worker) as pool:
        for _ in tqdm(pool.imap_unordered(write, work_items), total=n):
            pass


def count_images(path):
    count = defaultdict(int)
    for root, _, files in tqdm(os.walk(path)):
        split = root.split("/")[-2]
        count[split] += len(files)

    return count


def generate_sets(root, sets, shape):
    """When experimenting we want that prepare check that the generated dataset is of the right size"""
    root = Path(root)
    sentinel = root / "done"

    total_images = count_images(root)

    for split, count in sets.items():
        current_count = total_images.get(split, 0)

        # Some benches need filenames to match those of imagenet:
        # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/datasets/image_net.py#L40-L43
        if split == "train":
            prefix = ""
        else:  # split in (val, test):
            prefix = f"ILSVRC2012_{split}_"

        if current_count < count:
            print(f"Generating {split} (current {current_count}) (target: {count})")
            generate(
                shape,
                count - current_count,
                os.path.join(root, split),
                prefix=prefix,
                start=current_count,
            )

    with open(sentinel, "w") as fp:
        json.dump(sets, fp)


def device_count():
    try:
        return acc.device_count()
    except:
        return 1
    

def fakeimagenet_args(argv=None):

    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--batch-count", default=60, type=int)
    parser.add_argument("--device-count", default=device_count(), type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--image-size", default=[3, 384, 384], type=int, nargs="+")
    parser.add_argument("--val", default=0.1, type=float, nargs="+")
    parser.add_argument("--test", default=0.1, type=float, nargs="+")
    parser.add_argument("--output", default=os.getenv("MILABENCH_DIR_DATA", None), type=str)
    
    args, _ = parser.parse_known_args(argv)
    return args


def generate_fakeimagenet(args=None):
    multiprocessing.set_start_method("spawn", force=True)

    # config = json.loads(os.environ["MILABENCH_CONFIG"])

    if args is None:
        args = fakeimagenet_args()

    if overrides := os.getenv("MILABENCH_TESTING_PREPARE"):
        bs, bc = overrides.split(",")
        args.batch_size, args.batch_count = int(bs), int(bc)

    assert args.output is not None, "Output directory is required"
    data_directory = args.output
    
    dest = os.path.join(data_directory, f"FakeImageNet")
    print(f"Generating fake data into {dest}...")

    total_images = args.batch_size * args.batch_count * args.device_count
    size_spec = {
        f"train": total_images,
        f"val": int(total_images * args.val),
        f"test": int(total_images * args.test),
    }

    generate_sets(dest, size_spec, args.image_size)

    labels = set([int(entry.name) for entry in Path(dest).glob("*/*/")])
    with open(os.path.join(dest, "labels.txt"), "wt") as _f:
        # class_id,class_name
        _f.writelines([f"{l},{l}\n" for l in sorted(labels)])

    print("Done!")


if __name__ == "__main__":
    generate_fakeimagenet()
