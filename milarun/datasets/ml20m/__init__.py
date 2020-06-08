import os
import subprocess
from types import SimpleNamespace as NS

from .dataset import CFTrainDataset, load_test_ratings, load_test_negs
from .convert import (convert, TEST_NEG_FILENAME, TEST_RATINGS_FILENAME,
                      TRAIN_RATINGS_FILENAME)

class ML20m:

    def __init__(self, path, nb_neg):
        self.dataroot = path
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

        os.makedirs(self.path, exist_ok=True)
        subprocess.run(
            f"""
            cd {self.dataroot}

            wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
            unzip -u ml-20m.zip
            rm ml-20m.zip
            """,
            shell=True
        )
        convert(
            NS(
                path=os.path.join(self.path, "ratings.csv"),
                negatives=999,
                output=self.path,
            )
        )

def ml20m(path, nb_neg=None):
    return ML20m(path, nb_neg)

