import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

from . import utils
from .transformer_net import TransformerNet
from .vgg import Vgg16

from coleo import Argument, default, auto_cli
from milarun.lib import coleo_main, init_torch, iteration_wrapper, dataloop, memory_size


here = os.path.dirname(os.path.realpath(__file__))
repo_base = os.path.join(here, "..", "..", "..")


@coleo_main
def main(exp):

    # dataset to use
    dataset: Argument & str

    # Number of examples per batch
    batch_size: Argument & int = default(64)

    # path to style-image
    style_image: Argument & str = default(
        os.path.join(repo_base, "neural-style-images/style-images/candy.jpg")
    )

    # size of training images, default is 256 X 256
    image_size: Argument & int = default(256)

    # size of style-image, default is the original size of style image
    style_size: Argument & int = default(None)

    # weight for content-loss, default is 1e5
    content_weight: Argument & float = default(1e5)

    # weight for style-loss, default is 1e10
    style_weight: Argument & float = default(1e10)

    # learning rate, default is 1e-3
    lr: Argument & float = default(1e-3)

    torch_settings = init_torch()
    device = torch_settings.device

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = exp.get_dataset(dataset, transform).train
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=torch_settings.workers
    )

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    print(memory_size(vgg, batch_size=batch_size, input_size=(3, image_size, image_size)) * 4)

    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(style_image, size=style_size)
    style = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    wrapper = iteration_wrapper(exp, sync=torch_settings.sync)

    transformer.train()

    for it, (x, _) in dataloop(train_loader, wrapper=wrapper):
        it.set_count(len(x))

        n_batch = len(x)

        x = x.to(device)
        y = transformer(x)

        y = utils.normalize_batch(y)
        x = utils.normalize_batch(x)

        optimizer.zero_grad()

        features_y = vgg(y)
        features_x = vgg(x)

        content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

        style_loss = 0.
        for ft_y, gm_s in zip(features_y, gram_style):
            gm_y = utils.gram_matrix(ft_y)
            style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
        style_loss *= style_weight

        total_loss = content_loss + style_loss
        total_loss.backward()

        it.log(loss=total_loss.item())
        optimizer.step()
