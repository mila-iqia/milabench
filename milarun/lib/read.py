
import os
import torchvision.datasets as datasets
import torch
from types import SimpleNamespace as NS
from PIL import Image


class _TensorLabelSeparate:
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, i):
        data = self.data[i]
        target = int(self.targets[i])
        data = Image.fromarray(data.numpy(), mode='L')
        return self.transform(data), target

    def __len__(self):
        return len(self.data)


def _load(env, spec, transform):
    if spec["organization"]["type"] == "classification_directory":
        if spec["format"]["type"].startswith("image/"):
            path = os.path.join(env["root"], spec["path"])
            return datasets.ImageFolder(path, transform)

    if spec["organization"]["type"] == "tensor_label_separate":
        if spec["format"]["type"] == "tensor/pt":
            path = os.path.join(env["root"], spec["path"])
            data, targets = torch.load(path)
            return _TensorLabelSeparate(data, targets, transform)

    raise Exception(f"Does not know how to read dataset: {spec}")


def pytorch_reader(dataset, transform):
    rval = NS()
    manifest = dataset["manifest"]
    assert all("partition" in entry for entry in manifest) or len(manifest) == 1
    for entry in manifest:
        if "partition" not in entry:
            return _load(dataset["environment"], entry, transform)
        else:
            setattr(rval, entry["partition"], _load(dataset["environment"], entry, transform))
    return rval
