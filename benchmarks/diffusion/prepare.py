#!/usr/bin/env python
from dataclasses import dataclass
import os

from datasets import load_dataset


@dataclass
class TrainingConfig:
    dataset_name: str = "huggan/smithsonian_butterflies_subset"


def main():
    from argklass import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(TrainingConfig)
    config, _ = parser.parse_known_args()

    _ = load_dataset(config.dataset_name, split="train")


if __name__ == "__main__":
    main()
