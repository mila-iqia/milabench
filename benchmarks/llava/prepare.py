#!/usr/bin/env python

import torch
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration


def main():
    # Load LLaVA model and processor with device_map="auto"
    _ = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float32,  # Change to float32
        device_map="auto",
        revision="e2214c2851fadaf9241c9f9ac91dcdee51981021"
    )
    _ = AutoProcessor.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        revision="e2214c2851fadaf9241c9f9ac91dcdee51981021"
    )

    # Load dataset and create DataLoader
    _ = load_dataset("HuggingFaceM4/the_cauldron", "aokvqa")["train"]


if __name__ == "__main__":
    main()
