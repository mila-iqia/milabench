#!/usr/bin/env python

import os

def download_huggingface_celebA():
    #
    # Format is no good
    #
    from datasets import load_dataset
    dataset = load_dataset(
        "student/celebA", 
        revision="2d31e6555722815c74ea7c845b07c1063dd705e9",
        cache_dir="/tmp/milabench/cuda/results/data"
    )


def download_torchvision_celebA():
    #
    #   pip install gdown
    #
    # gdown.exceptions.FileURLRetrievalError: Failed to retrieve file url:
    #
    #         Too many users have viewed or downloaded this file recently. Please
    #         try accessing the file again later. If the file you are trying to
    #         access is particularly large or is shared with many people, it may
    #         take up to 24 hours to be able to view or download the file. If you
    #         still can't access a file after 24 hours, contact your domain
    #         administrator.

    from torchvision.datasets import CelebA
    dataset = CelebA(
        os.path.join(os.environ["MILABENCH_DIR_DATA"], "CelebA"), 
        split="train",
        download=True
    )

def main():
    """"""
    # download_torchvision_celebA()


if __name__ == "__main__":
    main()
