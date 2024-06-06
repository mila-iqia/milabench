#!/usr/bin/env python

import json
import os
import argparse
import time
import sys

import torch

from benchmate.monitor import setupvoir
import torchcompat.core as accelerator

root = os.path.dirname(__file__)


def available_models():
    models = dict()

    for size in ("7b", "13b", "70b"):
        models[f"llama2-{size}"] = {
            "name": f"meta-llama/Llama-2-{size}-chat-hf",
            "config": f"llama2_{size}_chat_hf.config",
        }

    return models


class WrappedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.count = 0

    def __call__(self, *args, **kwargs):
        input_ids = self.tokenizer(*args, **kwargs)

        self.count = 1
        for c in input_ids["input_ids"].shape:
            self.count *= c

        return input_ids

    def __getattr__(self, attr):
        if hasattr(self.tokenizer, attr):
            method = getattr(self.tokenizer, attr)
            return method
        else:
            raise AttributeError(
                f"'{type(self.tokenizer).__name__}' object has no attribute '{attr}'"
            )


def println(*args, **kwargs):
    print(*args, *kwargs, file=sys.stderr)


def huggingface_main(args, model, config):
    # Huggingface imported AFTER setup
    import transformers
    from transformers import LlamaForCausalLM, LlamaTokenizerFast
    from transformers.models.llama.configuration_llama import LlamaConfig
    from voir.wrapper import DataloaderWrapper, Wrapper
    from datasets import load_dataset
    import optimum.habana
    
    # Dataset here
    println("Dataset")
    dataset = load_dataset("wikitext", "wikitext-103-v1")

    println("Tokenizer")
    # LLAMA tokenizer official tokenizer is hidden behind a login
    tokenizer = WrappedTokenizer(
        LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    )

    # Prepare is done
    if args.prepare:
        return 0

    # We do not download LLAMA because it takes too long
    # we just instantiate an untrained one
    println("Model")
    device = accelerator.fetch_device(0)

    model = LlamaForCausalLM(LlamaConfig.from_dict(config)).to(device=device)

    println("Pipeline")
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        # device_map="cuda",
        tokenizer=tokenizer,
        device=device,
    )

    in_token_count = 0
    out_token_count = 0

    start = time.time()

    log, monitor = setupvoir()

    # loader = Wrapper(dataset["train"], accelerator.Event, earlystop=60)
    loader = dataset["train"]

    println("Starting")
    count = 0
    for entry in loader:
        text = entry["text"].strip()

        # Titles
        if text == "" or text.startswith(" = ") or len(text) < 10:
            continue

        count += 1
        sequences = pipeline(
            text,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=400,
        )

        for seq in sequences:
            out_token_count += len(seq["generated_text"])

        in_token_count += tokenizer.count
        total = out_token_count + in_token_count

        elapsed = time.time() - start
        println(
            f"{elapsed =}, {total / elapsed =} {in_token_count =} {out_token_count =}"
        )

        if total > 30:
            out_token_count = 0
            in_token_count = 0
            start = time.time()

            if log is not None:
                log({"task": "train", "rate": total / elapsed, "units": "Tok/s"})

        if count > 40:
            break

    monitor.stop()


def main():
    import torch

    models = available_models()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama2-7b", choices=models.keys())
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--cache", required=True, type=str)

    #
    args = parser.parse_args()
    os.environ["XDG_CACHE_HOME"] = str(args.cache)

    settings = models[args.model]
    model, config = settings["name"], settings["config"]

    with open(os.path.join(root, "config", config), "r") as file:
        config = json.load(file)

    with torch.no_grad():
        return huggingface_main(args, model, config)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        # Habana likes to eat exceptions
        print(err)