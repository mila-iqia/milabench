# This is the script run by milabench run (by default)

import time
import torchcompat.core as accelerator
from benchmate.observer import BenchObserver
from transformers import LlavaForConditionalGeneration, AutoProcessor
from datasets import load_dataset
import torch


def apply_chat_template(conversation):
    formatted_conversation = ""
    for turn in conversation:
        formatted_conversation += "<image>\n"
        formatted_conversation += f"Human: {turn['user']}\n"
        formatted_conversation += f"Assistant: {turn['assistant']}\n"
    return formatted_conversation.strip()

def main():
    device = accelerator.fetch_device(0)  # <= This is your cuda device

    # Load LLaVA model and processor
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf").to(device)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # Load a small subset of the dataset for benchmarking
    dataset = load_dataset("HuggingFaceM4/the_cauldron", "aokvqa")["train"]


    observer = BenchObserver(
        batch_size_fn=lambda batch: 1
    )

    optimizer = observer.optimizer(torch.optim.AdamW(model.parameters(), lr=5e-5))
    
    for epoch in range(10000):
        for i in observer.iterate(dataset):
            # Prepare input
            images = i['images']
            conversation = i['texts']
            prompt = apply_chat_template(conversation)
            print(prompt)
            inputs = processor(text=prompt, images=images, return_tensors="pt").to(device)

            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            observer.record_loss(loss)

    assert epoch < 2, "milabench stopped the train script before the end of training"
    assert i < 72, "milabench stopped the train script before the end of training"

if __name__ == "__main__":
    main()
