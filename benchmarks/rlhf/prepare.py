#!/usr/bin/env python

from dataclasses import dataclass

from argklass import ArgumentParser
from benchmate.hugginface import download_hf_dataset, download_hf_model


@dataclass
class Arguments:
    dataset: str = "trl-internal-testing/descriptiveness-sentiment-trl-style"
    split: str = "descriptiveness"
    subset: str = None
    model_name_or_path: str = "EleutherAI/pythia-1b-deduped"
    per_device_train_batch_size: int = 16


def arguments() -> Arguments:
    parser = ArgumentParser()
    parser.add_arguments(Arguments)
    args, _ = parser.parse_known_args()
    return args


def new_prepare():
    args = arguments()

    download_hf_dataset(args.dataset, args.split, name=args.subset)

    # Download model
    download_hf_model(args.model_name_or_path)

    print("=" * 60)
    print("Prepare script completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    new_prepare()
