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
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from itertools import chain
from pathlib import Path
from typing import Optional

import datasets
import rich.logging
import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler
from accelerate.utils.dataclasses import InitProcessGroupKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import get_full_repo_name

logger = get_logger(__name__)


# New Code #
def evaluate(per_gpu_batch_size, model, eval_dataloader, accelerator, eval_dataset):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(per_gpu_batch_size)))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        eval_loss = float("inf")
        perplexity = float("inf")
    return perplexity, eval_loss


def main():
    config = json.loads(os.environ["MILABENCH_CONFIG"])
    per_gpu_batch_size = config["per_gpu_batch_size"]
    max_train_steps = config["max_train_steps"]

    @dataclass
    class CustomInitProcessGroupKwargs(InitProcessGroupKwargs):
        # IDEA: `InitProcessGroupKwargs` only has `init_method` and `timeout` entries. I'd add `store` too.
        init_method: Optional[str] = None
        timeout: timedelta = timedelta(seconds=1800)

        # store: Optional[Store] = None

        rank: Optional[int] = None
        world_size: Optional[int] = None

    MASTER_ADDR = os.environ["MASTER_ADDR"]
    MASTER_PORT = os.environ["MASTER_PORT"]

    init_process_group_kwargs = CustomInitProcessGroupKwargs(
        init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
        # Reduced the timeout here, so the job fails quicker if there's a communication problem between nodes.
        timeout=timedelta(seconds=60),
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )

    accelerator = Accelerator(
        log_with=None,
        kwargs_handlers=[init_process_group_kwargs],
    )
    # Make one log on every process with the configuration for debugging.

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

    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # Downloading and loading a dataset from the hub.
    validation_split_percentage = config["validation_split_percentage"]
    dataset_name = config["dataset_name"]
    dataset_config_name = config["dataset_config_name"]
    raw_datasets = load_dataset(dataset_name, dataset_config_name)
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[:{validation_split_percentage}%]",
        )
        raw_datasets["train"] = load_dataset(
            dataset_name,
            dataset_config_name,
            split=f"train[{validation_split_percentage}%:]",
        )

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that
    # only one local process can concurrently download model & vocab.
    model_name = config["model_name"]
    model_config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True,
    )

    model = AutoModelForCausalLM.from_config(model_config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=config['cpus_per_gpu'],
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

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.local_main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=config['cpus_per_gpu'],
            load_from_cache_file=True,
            # TODO: See if this works (i.e. makes things faster and doesn't invalidate the cache)
            # keep_in_memory=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    if os.environ.get('MILABENCH_PREPARE_ONLY', None) is not None:
        return

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

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

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Scheduler and math around the number of training steps.

    # New Code
    # Get gradient accumulation steps from deepspeed config if available
    if accelerator.state.deepspeed_plugin is not None:
        gradient_accumulation_steps = (
            accelerator.state.deepspeed_plugin.deepspeed_config[
                "gradient_accumulation_steps"
            ]
        )

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

    # Train!
    total_batch_size = (
        per_gpu_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0
    best_metric = None

    # NOTE (@lebrice): Added these for the throughput metrics below.
    start_time: Optional[float] = None
    last_log_time: Optional[float] = None
    n_updates_since_start_of_run: int = 0
    n_updates_since_last_log: int = 0
    throughput_samples_per_sec: float = 0.0  # instantaneous throughput
    throughput_samples_per_sec_since_start: float = (
        0.0  # Average throughput since start of run
    )

    total_loss = 0.0

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                train_dataloader
            ) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                # NOTE: Added (@lebrice) for throughput metrics
                n_updates_since_start_of_run += 1
                n_updates_since_last_log += 1

            if (
                accelerator.is_local_main_process
                and completed_steps % 10 == 0
            ):
                if start_time is None:
                    # The first time we get here (first logging call), we've never logged before,
                    # so we can't calculate the throughput. Only save the start time for the next
                    # call.
                    start_time = time.time()
                else:
                    seconds_since_start = time.time() - start_time
                    seconds_since_last_log = time.time() - (last_log_time or start_time)

                    # TODO: Not 100% sure, but seems like we're only logging values on the first node,
                    # so we assume that one update here = one update on all nodes = total_batch_size
                    # samples.
                    n_samples_since_start = (
                        n_updates_since_start_of_run * total_batch_size
                    )
                    n_samples_since_last_log = (
                        n_updates_since_last_log * total_batch_size
                    )

                    throughput_samples_per_sec = (
                        n_samples_since_last_log / seconds_since_last_log
                    )
                    throughput_samples_per_sec_since_start = (
                        n_samples_since_start / seconds_since_start
                    )

                    # TODO: If we want to use tensorboard, use `accelerator.log` instead.
                    # accelerator.log(
                    print(
                        {
                            "train_loss": loss.detach().item(),
                            "samples_per_sec": throughput_samples_per_sec,
                            "avg_samples_per_sec": throughput_samples_per_sec_since_start,
                            "epoch": epoch,
                            "step": completed_steps,
                            "n_samples": completed_steps * total_batch_size,
                        }
                    )

                last_log_time = time.time()
                n_updates_since_last_log = 0

            if completed_steps >= max_train_steps:
                break

        perplexity, eval_loss = evaluate(
            per_gpu_batch_size, model, eval_dataloader, accelerator, eval_dataset
        )
        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if accelerator.is_main_process:
            # accelerator.log(
            print(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_epoch_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                }
            )

        # Tracks the best metric
        if best_metric is None or best_metric > perplexity:
            best_metric = perplexity
            accelerator.print(f"New best metric: {best_metric} at epoch {epoch}")


if __name__ == "__main__":
    main()
