#!/usr/bin/env python

import os

from pcqm4m_subset import PCQM4Mv2Subset

if __name__ == "__main__":
    # If you need the whole configuration:
    # config = json.loads(os.environ["MILABENCH_CONFIG"])

    data_directory = os.environ["MILABENCH_DIR_DATA"]

    # TODO: Handle argument for the number of samples
    dataset = PCQM4Mv2Subset(10000, root=data_directory)
    # If there is nothing to download or generate, just delete this file.
