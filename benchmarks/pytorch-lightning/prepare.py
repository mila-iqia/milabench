#!/usr/bin/env python

import os
import json
from utils import DataOptions
from simple_parsing import ArgumentParser
import shlex


def main():
    # If you need the whole configuration:
    config = json.loads(os.environ["MILABENCH_CONFIG"])
    data_directory = os.environ["MILABENCH_DIR_DATA"]
    argv_dict = config["argv"]
    argv_str = " ".join(f"--{k} {v}" for k, v in argv_dict.items())
    # Download (or generate) the needed dataset(s). You are responsible
    # to check if it has already been properly downloaded or not, and to
    # do nothing if it has been.

    # If there is nothing to download or generate, just delete this file.
    parser = ArgumentParser()
    parser.add_arguments(DataOptions, "options")
    args, _ = parser.parse_known_args(shlex.split(argv_str))
    options: DataOptions = args.options

    datamodule = options.datamodule(data_dir=data_directory)
    datamodule.prepare_data()


if __name__ == "__main__":
    main()
