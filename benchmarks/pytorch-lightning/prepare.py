#!/usr/bin/env python

import os
import json
from utils import DataOptions
from simple_parsing import ArgumentParser
import shlex


def main():
    config = json.loads(os.environ["MILABENCH_CONFIG"])
    data_directory = os.environ["MILABENCH_DIR_DATA"]
    argv_dict = config["argv"]
    argv_str = " ".join(f"--{k} {v}" for k, v in argv_dict.items())

    parser = ArgumentParser()
    parser.add_arguments(DataOptions, "options")
    args, _ = parser.parse_known_args(shlex.split(argv_str))
    options: DataOptions = args.options

    datamodule = options.datamodule(data_dir=data_directory)
    datamodule.prepare_data()


if __name__ == "__main__":
    main()
