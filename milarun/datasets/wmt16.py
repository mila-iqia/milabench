
import os
from milarun.models.seq2seq.data.dataset import ParallelDataset
from milarun.models.seq2seq.data.tokenizer import Tokenizer


class WMT16:
    def __init__(self, path, config, min_length, max_length, max_size):
        self.path = os.path.join(path, "wmt16")
        self.config = config
        self.min_length = min_length
        self.max_length = max_length
        self.max_size = max_size

    def avail(self, download=True):
        if download:
            self.download()
        tokenizer = Tokenizer(os.path.join(self.path, self.config.VOCAB_FNAME))
        self.data = ParallelDataset(
            src_fname=os.path.join(self.path, self.config.SRC_TRAIN_FNAME),
            tgt_fname=os.path.join(self.path, self.config.TGT_TRAIN_FNAME),
            tokenizer=tokenizer,
            min_len=self.min_length,
            max_len=self.max_length,
            sort=False,
            max_size=self.max_size
        )
        self.tokenizer = tokenizer

    def download(self):
        if os.path.exists(self.path):
            return
        raise Exception("TODO")


def wmt16(path, config, min_length, max_length, max_size):
    return WMT16(path, config, min_length, max_length, max_size)
