import os

from .dataset import CFTrainDataset, load_test_ratings, load_test_negs
from .convert import (TEST_NEG_FILENAME, TEST_RATINGS_FILENAME,
                      TRAIN_RATINGS_FILENAME)

class ML20m:

    def __init__(self, path, nb_neg):
        self.path = os.path.join(path, "ml-20m")
        self.nb_neg = nb_neg

    def avail(self, download=True):
        if download:
            self.download()
        train_path = os.path.join(self.path, TRAIN_RATINGS_FILENAME)
        self.train = CFTrainDataset(train_path, self.nb_neg)

    def download(self):
        if os.path.exists(self.path):
            return
        raise Exception("TODO")


def ml20m(path, nb_neg):
    return ML20m(path, nb_neg)

