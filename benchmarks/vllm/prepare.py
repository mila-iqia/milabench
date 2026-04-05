#!/usr/bin/env python

from argparse import ArgumentParser
from benchmate.hugginface import download_hf_dataset, download_hf_model
import subprocess
import sys


def arguments():
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


def setup_flashinfer():
    commands = [
        # ["flashinfer", "clear-cache"],
        ["flashinfer", "show-config"],
        # ["flashinfer", "download-cubin"],
    ]
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print(f"Warning: '{' '.join(cmd)}' exited with code {result.returncode}")


def main():
    args = arguments()

    download_hf_dataset(args.hf_name, args.hf_split)
    download_hf_model(args.model)

    setup_flashinfer()

    print("=" * 60)
    print("Prepare script completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
