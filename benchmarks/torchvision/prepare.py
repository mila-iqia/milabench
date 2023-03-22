#!/usr/bin/env python

import os

from milabench.datasets.fake_images import generate_sets

if __name__ == "__main__":
    data_directory = os.environ["MILABENCH_DIR_DATA"]
    dest = os.path.join(data_directory, "FakeImageNet")
    print(f"Generating fake data into {dest}...")
    generate_sets(dest, {"train": 10000, "val": 100, "test": 100}, (3, 512, 512))
    print("Done!")
