from pathlib import Path
import os
from dataclasses import dataclass

import torch
from torchvision import transforms
import torch.nn.functional as F
from diffusers import DDPMScheduler

from diffusers import DDPMPipeline
from accelerate import Accelerator
from tqdm.auto import tqdm
from datasets import load_dataset
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

# from huggingface_hub import HfFolder, Repository, whoami

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    gradient_accumulation_steps = 1
    dataset_name: str = "huggan/smithsonian_butterflies_subset"
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


def build_dataset(config):
    dataset = load_dataset(config.dataset_name, split="train")

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)

    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.train_batch_size, 
        shuffle=True
    )
    return loader



def build_model(config):
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    return model



def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def build_loss():
    return F.mse_loss

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    from benchmate.observer import BenchObserver

    def batch_size(x):
        return x["images"].shape[0]

    observer = BenchObserver(
        earlystop=65,
        batch_size_fn=lambda x: batch_size(x),
        stdout=True,
        raise_stop_program=True
    )

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # log_with="tensorboard",
        # project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if False:
            if config.push_to_hub:
                repo_name = get_full_repo_name(Path(config.output_dir).name)
                repo = Repository(config.output_dir, clone_from=repo_name)
            elif config.output_dir is not None:
                os.makedirs(config.output_dir, exist_ok=True)
            accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    criterion = build_loss()

    # Now you train the model
    for epoch in range(config.num_epochs):
        # progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        # progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(observer.iterate(train_dataloader)):
            clean_images = batch["images"].to(model.device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = criterion(noise_pred, noise)
                accelerator.backward(loss)
                observer.record_loss(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # progress_bar.update(1)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if False:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    evaluate(config, epoch, pipeline)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    if config.push_to_hub:
                        repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                    else:
                        pipeline.save_pretrained(config.output_dir)




def build_optimizer(config, model):
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

def main():
    config = TrainingConfig()

    model = build_model(config)
    dataset = build_dataset(config)
    optimizer = build_optimizer(config, model)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

  
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(dataset) * config.num_epochs),
    )

    from benchmate.metrics import StopProgram

    try:
        train_loop(config, model, noise_scheduler, optimizer, dataset, lr_scheduler)

    except StopProgram:
        pass

if __name__ == "__main__":
    main()
