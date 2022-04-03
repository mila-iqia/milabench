#!/usr/bin/env python

import os

from torchvision.datasets import MNIST

if __name__ == "__main__":
    dest = os.environ["MILABENCH_DIR_DATA"]
    print(f"Downloading MNIST into {dest}/MNIST...")
    MNIST(dest, download=True)
    print("Done!")
