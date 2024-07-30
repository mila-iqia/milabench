#!/usr/bin/env python
import argparse
from dataclasses import dataclass

from argklass import ArgumentParser, argument
from torchtune._cli.tune import TuneCLIParser
import yaml

@dataclass
class Arguments:
    recipe: str
    config: str = None


class MyParser(TuneCLIParser):
    def parse_args(self, args=None) -> argparse.Namespace:
        """Parse CLI arguments"""
        return self._parser.parse_args(args)


def main():

    parser = ArgumentParser()
    parser.add_arguments(Arguments)
    args, _ = parser.parse_known_args()

    from omegaconf import OmegaConf
    cli = OmegaConf.from_cli()
    base = OmegaConf.load(args.config)
    config = OmegaConf.merge(base, cli)

    repo_id = config["repo_id"]
    hf_token = ""
    output_dir = config["checkpointer"]["output_dir"]

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
