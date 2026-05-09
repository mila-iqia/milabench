#!/usr/bin/env python

# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import shutil

# Per-GPU launches (milabench's `per_gpu` plan) all share the same host
# and default to MASTER_PORT=29500. With vllm_mode=colocate, vllm calls
# torch.distributed.init_process_group inside each process, causing
# EADDRINUSE. Derive a unique port from the first visible GPU id.
def _setup_distributed_env():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES") or ""
    first = visible.split(",")[0].strip() if visible else ""
    try:
        offset = int(first)
    except ValueError:
        offset = 0
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + offset))

_setup_distributed_env()

import torch
import accelerate
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
import torchcompat.core as compat

SIMPLE_CHAT_TEMPLATE = "{% for message in messages %}{{message['role'].capitalize() + ': ' + message['content'] + '\n\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"


class GRPOTrainerInstrumented(GRPOTrainer):
    def __init__(self, args: GRPOConfig, *pargs, **kwargs):
        args.report_to = []

        # Same monkeypatch as the rlhf/PPO bench: swap in milabench's
        # compat Accelerator before the trainer constructs its own.
        accelerate.Accelerator = compat.accelerate.Accelerator
        super().__init__(args=args, *pargs, **kwargs)

        from benchmate.observer import BenchObserver

        def batch_size_fn(batch):
            # GRPO's dataloader yields raw rows (a `prompt` column of strings)
            # before generation; count prompts per step. Effective work per
            # step is roughly batch_size * num_generations * max_completion_length.
            if isinstance(batch, dict):
                if "prompt" in batch:
                    return len(batch["prompt"])
                if "input_ids" in batch:
                    shape = batch["input_ids"].shape
                    return shape[0] * shape[-1]
            return 1

        self._bench_observer = BenchObserver(
            batch_size_fn=batch_size_fn,
            earlystop=70,
            raise_stop_program=True,
            stdout=True,
        )

    def get_train_dataloader(self):
        return self._bench_observer.iterate(super().get_train_dataloader())

    def _save_checkpoint(self, *args, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass


def main():
    from trl.scripts.utils import ScriptArguments

    parser = HfArgumentParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # Parity with the rlhf/PPO bench: reuse the policy checkpoint as the
    # reward model (sequence-classification head with num_labels=1). GRPO
    # needs no value model — group-relative advantages replace the critic.
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
    )

    peft_config = get_peft_config(model_args)

    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
    )
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))

    # GRPO consumes a `prompt` text column directly; trainer tokenizes
    # internally per generation step.
    with PartialState().local_main_process_first():
        train_dataset = train_dataset.select_columns(["prompt"])
        eval_dataset = eval_dataset.select_columns(["prompt"])

    ################
    # Training
    ################
    trainer = GRPOTrainerInstrumented(
        args=training_args,
        model=model_args.model_name_or_path,
        reward_funcs=reward_model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    from voir.phase import StopProgram
    from benchmate.monitor import bench_monitor

    try:
        with bench_monitor():
            main()
    except StopProgram:
        pass
