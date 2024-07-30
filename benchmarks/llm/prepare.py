#!/usr/bin/env python
import argparse
from dataclasses import dataclass


from torchtune._cli.tune import TuneCLIParser
import pyyaml

@dataclass
class Arguments:
    recipe: str
    config: str = None


class MyParser(TuneCLIParser):
    def parse_args(self, args=None) -> argparse.Namespace:
        """Parse CLI arguments"""
        return self._parser.parse_args(args)


def main():
    from argklass import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(Arguments)
    args, _ = parser.parse_known_args()

    print(args)

    with open(args.config, "r") as fp:
        config = pyyaml.safe_load(fp)

    repo_id = config["model"]["repo_id"]
    hf_token = ""
    output_dir = config["checkpointer"]["checkpoint_dir"]

    download_args = [
        "download",
        repo_id,
        "--output-dir",
        output_dir
    ]
    
    if hf_token:
        download_args.extend([
            "--hf-token",
            hf_token,
        ])
                
    parser = MyParser()
    args = parser.parse_args(download_args)
    parser.run(args)


if __name__ == "__main__":
    main()
