import multiprocessing
import os

from torchvision.datasets import FakeData
from tqdm import tqdm

from ..fs import XPath


def write(args):
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
    count = n // p_count
    pool = multiprocessing.Pool(p_count)
    for _ in tqdm(pool.imap_unordered(write, ((image_size, i, n, outdir) for i in range(n))), total=n):
        pass


def generate_sets(root, sets, shape):
    root = XPath(root)
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
