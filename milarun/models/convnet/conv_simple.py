import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import time

try:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
except:
    pass

from coleo import Argument, default, auto_cli
from milarun.lib import coleo_main, init_torch, iteration_wrapper, dataloop
from milarun.lib import OptimizerAdapter, ModelAdapter


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])


@coleo_main
def main(exp):
    # dataset to use
    dataset: Argument & str

    # Learning rate
    lr: Argument & float = default(0.1)

    # Architecture to use
    arch: Argument = default("resnet18")

    # Batch size
    batch_size: Argument & int = default(128)

    # Whether to use float16
    half: Argument & bool = default(False)

    torch_settings = init_torch()
    device = torch_settings.device

    model = models.__dict__[arch]()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = OptimizerAdapter(torch.optim.SGD(
        model.parameters(),
        lr),
        half=half,
        dynamic_loss_scale=True
    )

    # ----
    model = ModelAdapter(model, half=half)

    # ----
    train_dataset = exp.get_dataset(dataset, data_transforms).train
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=torch_settings.workers,
        pin_memory=True,
    )

    wrapper = iteration_wrapper(exp, sync=None)

    for it, (input, target) in dataloop(train_loader, wrapper=wrapper):
        it.set_count(batch_size)

        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = criterion(output, target)

        it.log(loss=loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()
