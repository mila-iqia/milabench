import os
from torchvision import datasets, transforms


class MNIST:
    def __init__(self, path):
        self.path = os.path.join(path, "mnist")

    def _prepare(self, download):
        self.train = datasets.MNIST(
            self.path,
            train=True,
            download=download,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,))
                ]
            ),
        )
        self.test = datasets.MNIST(
            self.path,
            train=False,
            download=download,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.Normalize((0.1307,), (0.3081,))
                ]
            ),
        )
        return True

    def avail(self, download=True):
        self._prepare(download=download)

    def download(self):
        self._prepare(download=True)


def mnist(path):
    return MNIST(path)
