#!/usr/bin/env python
import os

from milabench.datasets.fake_images import generate_sets


# adjust the size of the generated dataset (1 = ~2Gb)
scale = 100
if __name__ == "__main__":
    # If you need the whole configuration:
    # config = json.loads(os.environ["MILABENCH_CONFIG"])

    data_directory = os.environ["MILABENCH_DIR_DATA"]
    dest = os.path.join(data_directory, "LargeFakeUniform")

    generate_sets(dest, {"train": 14000 * scale, "val": 500 * scale, "test": 500 * scale}, (3, 512, 512))
