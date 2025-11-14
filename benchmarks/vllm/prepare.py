#!/usr/bin/env python

from argparse import ArgumentParser
from benchmate.hugginface import download_hf_dataset, download_hf_model
import os


def arguments():
    import sys

    # Remove '--' separator if present
    argv = [arg for arg in sys.argv[1:] if arg != '--']

    parser = ArgumentParser()
    parser.add_argument('server_model', type=str, help='Model to use for the server')
    parser.add_argument('--model', type=str, help='Model name (client-side)')
    parser.add_argument('--dataset-name', type=str, help='Dataset name (random, hf, etc.)')
    parser.add_argument('--dataset-path', type=str, help='Path to HuggingFace dataset')
    parser.add_argument('--hf-name', type=str, help='HuggingFace dataset name')
    parser.add_argument('--hf-split', type=str, default=None, help='Dataset split to use')

    args, _ = parser.parse_known_args(argv)
    return args


def main():
    args = arguments()

    # Download dataset
    download_hf_dataset(args.hf_name, args.hf_split)

    # Download model
    download_hf_model(args.model)

    print("=" * 60)
    print("Prepare script completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
