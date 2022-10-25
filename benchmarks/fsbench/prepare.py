#!/usr/bin/env python
import os

from milabench.datasets.fake_images import generate_sets
from milabench.fs import XPath
import deeplake
import numpy as np


def make_deeplake_group(ds, folder, class_names):
    group_name = os.path.basename(folder)

    files_list = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(folder)):
        for filename in filenames:
            if filename == 'done':
                continue
            files_list.append(os.path.join(dirpath, filename))

    with ds:
        ds.create_group(group_name)
        ds[group_name].create_tensor('images', htype='image', sample_compression='jpeg')
        ds[group_name].create_tensor('labels', htype='class_label', class_names=class_names)
        for f in files_list:
            label_num = int(os.path.basename(os.path.dirname(f)))
            ds[group_name].append({'images': deeplake.read(f), 'labels': np.uint16(label_num)})


# adjust the size of the generated dataset (1 = ~2Gb)
scale = 100
if __name__ == "__main__":
    # If you need the whole configuration:
    # config = json.loads(os.environ["MILABENCH_CONFIG"])

    data_directory = os.environ["MILABENCH_DIR_DATA"]
    dest = os.path.join(data_directory, "LargeFakeUniform")

    generate_sets(dest, {"train": 14000 * scale, "val": 500 * scale, "test": 500 * scale}, (3, 512, 512))

    root = dest + '.lake'
    sentinel = XPath(root + '-done')
    if sentinel.exists():
        print(f"{root} was already generated")
    else:
        ds = deeplake.empty(dest + '.lake')
        class_names = [str(i) for i in range(1000)]
        make_deeplake_group(ds, os.path.join(dest, 'train'), class_names)
        make_deeplake_group(ds, os.path.join(dest, 'val'), class_names)
        make_deeplake_group(ds, os.path.join(dest, 'test'), class_names)
        sentinel.touch()
