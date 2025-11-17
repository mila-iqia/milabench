#!/usr/bin/env python
from dataclasses import dataclass

from argklass import ArgumentParser
from argklass.arguments import argument
from benchmate.hugginface import download_hf_dataset, download_hf_model


@dataclass
class Arguments:
    mode: str = None
    dataset: str = None
    split: str = None
    subset: str = None
    model: str = None
    batch_size: int = 16
    dtype: str = "bfloat16"
    multi_gpu: bool = False
    prepare: bool = False

def arguments():
    parser = ArgumentParser()
    parser.add_arguments(Arguments)
    args, _ = parser.parse_known_args()

    return args


def main():
    args = arguments()

    # Download dataset
    if args.mode == "flux":
        from datasets import load_dataset

        base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
        num_shards = 10  # Number of webdataset tar files
        urls = [base_url.format(i=i) for i in range(num_shards)]
        _ = load_dataset(
            "webdataset", 
            data_files={"train": urls}, 
            split="train", 
            streaming=False
        )
    else:
        download_hf_dataset(args.dataset, args.split, name=args.subset)

    # Download model
    download_hf_model(args.model)

    print("=" * 60)
    print("Prepare script completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
