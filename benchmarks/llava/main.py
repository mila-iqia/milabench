# This is the script run by milabench run (by default)

import time

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import AutoProcessor, LlavaForConditionalGeneration

from benchmate.observer import BenchObserver


def apply_chat_template(texts):
    formatted_conversation = "<image>\n"
    for conversation in texts:
        formatted_conversation += f"Human: {conversation['user'][0]}\n"
        formatted_conversation += f"Assistant: {conversation['assistant'][0]}\n"
    return formatted_conversation.strip()


def custom_collate(batch):
    if isinstance(batch[0], dict):
        return {key: custom_collate([d[key] for d in batch]) for key in batch[0].keys()}
    elif isinstance(batch[0], (list, tuple)):
        return [custom_collate(samples) for samples in zip(*batch)]
    elif isinstance(batch[0], Image.Image):
        return batch  # Return PIL images as is
    else:
        return default_collate(batch)


def main():
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=4,
        log_with="all",
        project_dir="logs",
    )

    set_seed(42)
    batch_size = 1  # Set to 1 for now, but can be easily changed
    num_epochs = 1

    # Load LLaVA model and processor with device_map="auto"
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float32,  # Change to float32
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # Load dataset and create DataLoader
    dataset = load_dataset("HuggingFaceM4/the_cauldron", "aokvqa")["train"]
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate
    )

    def batch_size_fn(batch):
        return (
            len(batch[1]["images"])
            if isinstance(batch, tuple)
            else len(batch["images"])
        )

    observer = BenchObserver(
        batch_size_fn=batch_size_fn, earlystop=70, raise_stop_program=True
    )
    optimizer = observer.optimizer(torch.optim.AdamW(model.parameters(), lr=5e-5))
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for epoch in range(num_epochs):
        for batch in observer.iterate(dataloader):
            images = batch["images"][0]  # Access the first item in the list of images
            texts = batch["texts"]
            prompt = apply_chat_template(texts)

            image = images[0] if isinstance(images, (list, tuple)) else images
            if isinstance(image, (list, tuple)) and len(image) == 1:
                image = image[0]

            inputs = processor(
                text=prompt, images=image, return_tensors="pt", padding=True
            )

            labels = inputs["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            inputs["labels"] = labels

            inputs = {
                k: v.to(
                    accelerator.device,
                    dtype=torch.float32 if v.dtype == torch.float16 else v.dtype,
                )
                for k, v in inputs.items()
            }
            outputs = model(**inputs)
            loss = outputs.loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()
            observer.record_loss(loss)

    assert epoch < 2, "milabench stopped the train script before the end of training"
    assert (
        observer.step < 70
    ), "milabench stopped the train script before the end of training"


if __name__ == "__main__":
    main()
