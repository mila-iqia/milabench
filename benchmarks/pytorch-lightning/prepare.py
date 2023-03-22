#!/usr/bin/env python

import json
import os

from utils import VISION_DATAMODULES, DataOptions

from milabench.datasets.fake_images import generate_sets


def generate_data():
    """Generate fake image net if the folder does not exist yet"""
    data_directory = os.environ["MILABENCH_DIR_DATA"]
    dest = os.path.join(data_directory, "FakeImageNet")
    
    if not os.path.exists(dest):
        print(f"Generating fake data into {dest}...")
        generate_sets(dest, {"train": 1000, "val": 10, "test": 10}, (3, 512, 512))
        print("Done!")


def main():
    config = json.loads(os.environ["MILABENCH_CONFIG"])
    data_directory = os.environ["MILABENCH_DIR_DATA"]
    argv_dict = config["argv"]

    datamodule_str = argv_dict.get("--datamodule")
    if datamodule_str is None:
        options = DataOptions()
        datamodule_cls = options.datamodule
    else:
        datamodule_cls = VISION_DATAMODULES[datamodule_str]

    datamodule = datamodule_cls(data_dir=data_directory)
    datamodule.prepare_data()
    
    generate_data()


if __name__ == "__main__":
    main()
