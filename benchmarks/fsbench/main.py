from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import os
from giving import give
#from pysquash import SquashCursor

shuffle = False
batch_size = 64
n_epoch_valid = 10
n_iters = 5
# number of process to use for loading (0 = don't use extra processes)
multiprocess_load = 0


class SqhDataset(Dataset):
    def __init__(self, root):
        self.root = root
        labels = list(root)
        self.files = []
        for label in labels:
            self.files.extend(map(lambda p: label + b'/' + p, root.cd(label)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = int(path.split(b'/')[0])
        return len(self.root.open(path, binary=True, buffering=0).readall()), label


def _ld(path):
    with open(path, 'rb') as f:
        return len(f.read())


def make_loader(path, sub, shuffle):
    load = DatasetFolder(os.path.join(path, sub), loader=_ld, extensions=('.jpeg',))
    #load = SqhDataset(SquashCursor(path + ".sqh").cd(sub.encode('utf-8')))
    #load = ImageFolder(path, transform=ToTensor())
    return DataLoader(load, batch_size=batch_size, shuffle=shuffle, num_workers=multiprocess_load)

def main():
    # Write code here
    data_directory = os.environ.get("MILABENCH_DIR_DATA", None)
    dataset_dir = os.path.join(data_directory, "LargeFakeUniform")

    train_loader = make_loader(dataset_dir, "train", shuffle=shuffle)
    valid_loader = make_loader(dataset_dir, "val", shuffle=False)
    test_loader = make_loader(dataset_dir, "test", shuffle=False)

    for iter in range(n_iters):
        for epoch in range(n_epoch_valid):
            for inp, target in train_loader:
                give(batch=inp, step=True)
        for inp, target in valid_loader:
            give(batch=inp, step=True)
    for inp, target in test_loader:
        give(batch=inp, step=True)


if __name__ == "__main__":
    # Note: The line `if __name__ == "__main__"` is necessary for milabench
    # to recognize the entry point (it does some funky stuff to it).
    main()
