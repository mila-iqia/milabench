#!/usr/bin/env python
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text
file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation

Old bug (fixed with the solution in the last reply to the thread):
- Fix bug: fatal error: cusolverDn.h: No such file or directory
https://github.com/microsoft/DeepSpeed/issues/2684
"""


def arguments():
    from argparse import ArgumentParser
    import os

    parser = ArgumentParser()
    parser.add_argument("--per_gpu_batch_size", required=True, type=int)
    parser.add_argument("--max_train_steps", required=True, type=int)
    parser.add_argument("--cpus_per_gpu", required=True, type=int)
    parser.add_argument("--validation_split_percentage", required=True, type=int)
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument("--dataset_config_name", required=True, type=str)
    parser.add_argument("--dataset_rev", required=True, type=str)
    parser.add_argument("--cache", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--prepare_only", action="store_true", default=False)

    #
    #   Is this still needed for docker?
    #
    # overrides = os.getenv("MILABENCH_CONFIG")
    # if overrides:
    #     return json.loads(overrides)

    args = parser.parse_args()

    # FIXME: we could move this logic to the activator
    os.environ["XDG_CACHE_HOME"] = str(args.cache)
    return vars(args)


_ = arguments()


# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from itertools import chain
from typing import Optional

import datasets
import rich.logging
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler
from accelerate.utils.dataclasses import InitProcessGroupKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
import torchcompat.core as acc
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from voir.smuggle import SmuggleWriter
from voir.instruments.gpu import get_gpu_info
from voir.instruments.utils import Monitor

logger = get_logger(__name__)


def main():
    #
    #   Is this still needed for docker?
    #
    # is_prepare_phase = os.environ.get('MILABENCH_PREPARE_ONLY', None) == "1"
    config = arguments()
    is_prepare_phase = config["prepare_only"]

    per_gpu_batch_size = config["per_gpu_batch_size"]
    max_train_steps = config["max_train_steps"]

    @dataclass
    class CustomInitProcessGroupKwargs(InitProcessGroupKwargs):
        # IDEA: `InitProcessGroupKwargs` only has `init_method` and `timeout` entries. I'd add `store` too.
        init_method: Optional[str] = None
        timeout: timedelta = timedelta(seconds=1800)
        backend: str = acc.ccl

        # store: Optional[Store] = None

        rank: Optional[int] = None
        world_size: Optional[int] = None

    if not is_prepare_phase:
        # This branch is when we run for real
        MASTER_ADDR = os.environ["MASTER_ADDR"]
        MASTER_PORT = os.environ["MASTER_PORT"]

        init_process_group_kwargs = CustomInitProcessGroupKwargs(
            init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
            timeout=timedelta(seconds=60),
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
        )
        print(init_process_group_kwargs.backend)

        # Accelerator SUCK, it is impossible to make it use hccl
        # We can bypass Accelerator logic by initializing the group ourselves
        if acc.device_type == "hpu":
            acc.init_process_group(
                init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
                timeout=timedelta(seconds=60),
                rank=int(os.environ["RANK"]),
                world_size=int(os.environ["WORLD_SIZE"]),
            )

        accelerator = Accelerator(kwargs_handlers=[init_process_group_kwargs])
    else:
        accelerator = Accelerator()

    if not is_prepare_phase and accelerator.is_main_process:
        # Set up logging for milabench (only in the run phase, for the main process)

        data_file = SmuggleWriter(sys.stdout)
        def mblog(data):
            if data_file is not None:
                print(json.dumps(data), file=data_file)

        def monitor_fn():
            data = {
                gpu["device"]: {
                    "memory": [gpu["memory"]["used"], gpu["memory"]["total"]],
                    "load": gpu["utilization"]["compute"],
                    "temperature": gpu["temperature"],
                }
                for gpu in get_gpu_info()["gpus"].values()
            }
            mblog({"task": "main", "gpudata": data})

        monitor_fn()
        monitor = Monitor(3, monitor_fn)
        monitor.start()

    else:

        def mblog(data):
            pass

        monitor = None

    logging.basicConfig(
        level=logging.INFO,
        format=f"[{accelerator.process_index}/{accelerator.num_processes}] %(name)s - %(message)s ",
        handlers=[
            rich.logging.RichHandler(markup=True, tracebacks_width=120)
        ],  # Very pretty, uses the `rich` package.
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    validation_split_percentage = config["validation_split_percentage"]
    dataset_name = config["dataset_name"]
    dataset_config_name = config["dataset_config_name"]
    raw_datasets = load_dataset(dataset_name, dataset_config_name, revision=config["dataset_rev"])
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[:{validation_split_percentage}%]", 
            revision=config["dataset_rev"]
        )
        raw_datasets["train"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[{validation_split_percentage}%:]", 
            revision=config["dataset_rev"]
        )

    model_name = config["model_name"]
    model_config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=config["cpus_per_gpu"],
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

    block_size = tokenizer.model_max_length
    if block_size > 1024:
        logger.warning(
            f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
            "Picking 1024 instead. You can change that default value by passing --block_size xxx."
        )
    block_size = 1024

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.local_main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=config["cpus_per_gpu"],
            load_from_cache_file=True,
            # TODO: See if this works (i.e. makes things faster and doesn't invalidate the cache)
            # keep_in_memory=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    if is_prepare_phase:
        return

    model = AutoModelForCausalLM.from_config(model_config)

    model.resize_token_embeddings(len(tokenizer))

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=per_gpu_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=per_gpu_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(no_decay_str in n for no_decay_str in no_decay)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(no_decay_str in n for no_decay_str in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    # New Code #
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=5e-5)
    # Scheduler and math around the number of training steps.

    # Get gradient accumulation steps from deepspeed config if available
    if accelerator.state.deepspeed_plugin is not None:
        gradient_accumulation_steps = (
            accelerator.state.deepspeed_plugin.deepspeed_config[
                "gradient_accumulation_steps"
            ]
        )
    else:
        gradient_accumulation_steps = 1

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=SchedulerType.LINEAR,
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer,
            total_num_steps=max_train_steps,
            warmup_num_steps=0,
        )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps < len(train_dataloader):
        max_train_steps = max_train_steps * gradient_accumulation_steps
    else:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    # Train! Choo! Choo!
    total_batch_size = (
        per_gpu_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )
    print("HERE", per_gpu_batch_size, total_batch_size)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    completed_steps = 0
    starting_epoch = 0
    last_log_time = time.time()

    from voir.wrapper import Wrapper
    wrapper = Wrapper(
        event_fn=acc.Event, 
        earlystop=30, 
        rank=int(os.environ["RANK"]), 
        device=acc.fetch_device(int(os.environ["RANK"])),
        stdout=True,
        batch_size_fn=lambda batch: batch["labels"].shape[0] * gradient_accumulation_steps
    )
    loader = wrapper.loader(train_dataloader)

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        for step, batch in enumerate(loader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            if accelerator.is_main_process:
                loader.add_loss(loss)
                # mblog({"task": "train", "loss": loss.detach().item()})

            accelerator.backward(loss)

            if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                train_dataloader
            ) - 1:
                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                optimizer.zero_grad()
                # This will make timings inconsistent if there are
                # skipped steps after the start.
                if not accelerator.optimizer_step_was_skipped:
                    completed_steps += 1

            if completed_steps >= max_train_steps:
                break


if __name__ == "__main__":
    main()
