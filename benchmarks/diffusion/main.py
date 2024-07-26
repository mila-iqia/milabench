from dataclasses import dataclass

from accelerate import Accelerator

import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler

@dataclass
class Arguments:
    model: str = "runwayml/stable-diffusion-v1-5"
    dataset: str = "lambdalabs/naruto-blip-captions"
    batch_size: int = 16
    num_workers: int = 8
    revision: str = None
    variant: str = None
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    scale_lr: bool = True
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = False
    variant: str = None
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    epochs: int = 10


def models(accelerator, args: Arguments):
    encoder = CLIPTextModel.from_pretrained(
        args.model, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.model, subfolder="vae", revision=args.revision, variant=args.variant
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.model, subfolder="unet", revision=args.revision, variant=args.variant
    )

    vae.requires_grad_(False)
    encoder.requires_grad_(False)
    unet.train()

    # Move text_encode and vae to gpu and cast to weight_dtype
    encoder.to(accelerator.device, dtype=torch.bfloat16)
    vae.to(accelerator.device, dtype=torch.bfloat16)

    return encoder, vae, unet


def dataset(accelerator, args: Arguments):
    dataset = load_dataset(args.dataset, None)

    image_column = "image"
    caption_column = "text"

    tokenizer = CLIPTokenizer.from_pretrained(
        args.model, subfolder="tokenizer", revision=args.revision
    )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    import os
    total_samples = args.batch_size * 70 * int(os.getenv("WORLD_SIZE", 1))

    # DataLoaders creation:
    return torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        # This should be a distributed sampler
        # but the dataset is a bit small so epochs are too small as well
        sampler=torch.utils.data.RandomSampler(
            train_dataset, 
            replacement=True, 
            num_samples=total_samples
        ),
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
    )


def train(observer, args: Arguments):
    weight_dtype = torch.bfloat16

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    loader = dataset(accelerator, args)

    encoder, vae, unet = models(accelerator, args)

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    max_train_steps = args.epochs * math.ceil(len(loader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")

    unet, optimizer, loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, loader, lr_scheduler
    )

    encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    for epoch in range(0, args.epochs):

        for step, batch in enumerate(observer.iterate(loader)):
            latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            noise = torch.randn_like(latents)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = encoder(batch["input_ids"], return_dict=False)[0]

            # Predict the noise residual and compute loss
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()



def prepare_voir():
    from benchmate.observer import BenchObserver
    from benchmate.monitor import bench_monitor
    def batch_size(x):
        return x["pixel_values"].shape[0]

    observer = BenchObserver(
        earlystop=60,
        raise_stop_program=True,
        batch_size_fn=batch_size,
        stdout=True
    )

    return observer, bench_monitor

def main():
    from benchmate.metrics import StopProgram

    observer, monitor = prepare_voir()

    with monitor():
        try:
            from argklass import ArgumentParser
            parser = ArgumentParser()
            parser.add_arguments(Arguments)
            config, _ = parser.parse_known_args()

            train(observer, config)
        except StopProgram:
            pass


if __name__ == "__main__":
    main()