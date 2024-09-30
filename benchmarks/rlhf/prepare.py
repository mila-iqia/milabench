#!/usr/bin/env python

import shutil

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from datasets import load_dataset
from trl import ModelConfig
from trl.trainer.ppov2_trainer import PPOv2Config
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE


if __name__ == "__main__":
    parser = HfArgumentParser((PPOv2Config, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    
    # remove output_dir if exists
    shutil.rmtree(config.output_dir, ignore_errors=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
    )

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    
    value_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path, 
        trust_remote_code=model_config.trust_remote_code, 
        num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path, 
        trust_remote_code=model_config.trust_remote_code, 
        num_labels=1
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path,
        trust_remote_code=model_config.trust_remote_code
    )
    policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, 
        trust_remote_code=model_config.trust_remote_code
    )

    raw_datasets = load_dataset("trl-internal-testing/descriptiveness-sentiment-trl-style", split="descriptiveness")
