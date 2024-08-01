#!/usr/bin/env python
import argparse
from dataclasses import dataclass
import os

from omegaconf import OmegaConf
from argklass import ArgumentParser
from torchtune._cli.tune import TuneCLIParser

from benchmate.ux import long_action


@dataclass
class Arguments:
    recipe: str
    config: str = None


class MyParser(TuneCLIParser):
    def parse_args(self, args=None) -> argparse.Namespace:
        """Parse CLI arguments"""
        return self._parser.parse_args(args)


def load_model(recipe, cfg):
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "recipes"))

    if recipe.endswith("full_finetune_distributed.py"):
        from full_finetune_distributed import FullFinetuneRecipeDistributed as Recipe

    if recipe.endswith("lora_finetune_distributed.py"):
        from lora_finetune_distributed import LoRAFinetuneRecipeDistributed as Recipe
        
    if recipe.endswith("lora_finetune_single_device.py"):
        from lora_finetune_single_device import LoRAFinetuneRecipeSingleDevice as Recipe

    with long_action("Still working", 30):
        recipe = Recipe(cfg=cfg)
        recipe.setup(cfg=cfg)
        recipe.save_checkpoint(0)


def main():
    parser = ArgumentParser()
    parser.add_arguments(Arguments)
    args, rest = parser.parse_known_args()

    cli = OmegaConf.from_cli()
    base = OmegaConf.load(args.config)
    config = OmegaConf.merge(base, cli)

    repo_id = config["repo_id"]
    hf_token = os.getenv("HUGGING_FACE_TOKEN", None)
    output_dir = config["checkpointer"]["output_dir"]
    ignore_pattern = "*.safetensors"

    if config.get("safetensors", False):
        ignore_pattern = "consolidated.*.pth"

    download_args = [
        "download",
        repo_id,
        "--output-dir",
        output_dir,
        "--ignore-patterns",
        ignore_pattern
    ]
    
    if hf_token:
        download_args.extend([
            "--hf-token",
            hf_token,
        ])
                
    parser = MyParser()
    args = parser.parse_args(download_args)
    parser.run(args)

    if "qlora" in config.get("model", {}).get("_component_", ""):
        load_model(args.recipe, config)


if __name__ == "__main__":
    main()
