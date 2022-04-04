#!/usr/bin/env python

import os

if __name__ == "__main__":
    # If you need the whole configuration:
    # config = json.loads(os.environ["MILABENCH_CONFIG"])

    data_directory = os.environ["MILABENCH_DIR_DATA"]

    # Download (or generate) the needed dataset(s). You are responsible
    # to check if it has already been properly downloaded or not, and to
    # do nothing if it has been.
    ...

    # If there is nothing to download or generate, just delete this file.
