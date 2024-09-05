#!/usr/bin/env python
import argparse
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
import time

import llama3.llama.model
import fairscale.nn.model_parallel
from omegaconf import OmegaConf
from argklass import ArgumentParser
import torch.distributed
from torchtune._cli.tune import TuneCLIParser

from benchmate.ux import long_action


@dataclass
class Arguments:
    recipe: str
    config: str = None


@dataclass
class ModelArgs(llama3.llama.model.ModelArgs):
    use_scaled_rope: bool = True


class MyParser(TuneCLIParser):
    def parse_args(self, args=None) -> argparse.Namespace:
        """Parse CLI arguments"""
        parsed_args = self._parser.parse_args(args)
        # Workaround to send a list to of ignore_patterns as self._parser does
        # not support a list in input
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--ignore-patterns",
            type=str,
            action='append',
        )
        ignore_patterns_args, _ = parser.parse_known_args(args)
        if ignore_patterns_args.ignore_patterns:
            parsed_args.ignore_patterns = ignore_patterns_args.ignore_patterns
        return parsed_args


def generate_model(
        conn:multiprocessing.connection.Connection,
        params_path:Path,
        rank=0,
        model_parallel_size=1
    ):
    try:
        torch.distributed.init_process_group(rank=rank, world_size=model_parallel_size)
        fairscale.nn.model_parallel.initialize.initialize_model_parallel(model_parallel_size)
        conn.send(os.getpid())
        while not conn.poll():
            time.sleep(0.1)
        conn.recv()
        params = json.loads(params_path.read_text())
        model = llama3.llama.model.Transformer(ModelArgs(**params))
        torch.save(model.state_dict(), params_path.with_name(f"consolidated.{rank:02}.pth"))
    except Exception as e:
        conn.send(e)
        raise
    finally:
        conn.close()


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

    ignore_patterns = ["*.safetensors", "original/consolidated.*.pth"]

    if config.get("safetensors", False):
        ignore_pattern = "*consolidated.*.pth"

    download_args = [
        "download",
        repo_id,
        "--output-dir",
        output_dir,
        *sum(
            [
                ["--ignore-patterns", ignore_pattern]
                for ignore_pattern in ignore_patterns
            ], 
            []
        )
    ]
    
    if hf_token is not None:
        download_args.extend([
            "--hf-token",
            hf_token,
        ])
    else:
        print("No HF token found...")

    parser = MyParser()
    args = parser.parse_args(download_args)
    parser.run(args)

    if not config.get("safetensors", False):
        params_path = next(args.output_dir.glob("**/params.json"))
        model_parallel_size = 8 if repo_id.split("-")[-1].lower() == "70b" else 1
        pipes = [multiprocessing.Pipe() for _ in range(model_parallel_size)]
        processes = [
            multiprocessing.Process(
                target=generate_model,
                args=[conn, params_path, rank, model_parallel_size]
            )
            for rank, (_, conn) in enumerate(pipes)
        ]
        # Init torch.distributed process_group and fairscale model parallel in
        # each fake workers
        [p.start() for p in processes]
        pids = set()
        for (conn, _) in pipes:
            while not conn.poll():
                time.sleep(0.1)
            pid = conn.recv()
            if isinstance(pid, Exception):
                raise pid
            pids.add(pid)
        assert len(pids) == model_parallel_size
        # Generate each chunk of the model one by one
        for p, (conn, _) in zip(processes, pipes):
            conn.send(True)
            p.join()

    if "qlora" in config.get("model", {}).get("_component_", ""):
        load_model(args.recipe, config)


if __name__ == "__main__":
    main()
