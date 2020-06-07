import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from coleo import Argument, default
from milarun.lib import init_torch, coleo_main, dataloop, iteration_wrapper


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

    # Number of examples per batch
    batch_size: Argument & int = default(256)

    # Dataset to load
    dataset: Argument

    torch_settings = init_torch()
    dataset = exp.get_dataset(dataset)

    loader = torch.utils.data.DataLoader(
        dataset.train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=torch_settings.workers,
        pin_memory=True
    )

    wrapper = iteration_wrapper(exp, sync=None)

    # Warm up a bit
    for _, batch in zip(range(10), loader):
        for item in batch:
            item.to(torch_settings.device)
        break

    for it, batch in dataloop(loader, wrapper=wrapper):
        it.set_count(batch_size)
        it.log(eta=True)
        batch = [item.to(torch_settings.device) for item in batch]
        if torch_settings.sync:
            torch_settings.sync()
