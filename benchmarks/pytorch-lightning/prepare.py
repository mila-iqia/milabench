#!/usr/bin/env python

import os
import json
from utils import VISION_DATAMODULES, DataOptions
from simple_parsing import ArgumentParser
import shlex


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


if __name__ == "__main__":
    main()
