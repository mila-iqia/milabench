import os
import subprocess
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, pad_to_multiple_of=1):
        # Synthetic elements used to pad the dictionary length.
        # It is assumed that these synthetic elements do not appear in the actual data files.
        self.synthetic = ["vvvvvvvv" + str(i) for i in range(pad_to_multiple_of-1)]

        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'wiki.train.tokens'))
        self.valid = self.tokenize(os.path.join(path, 'wiki.valid.tokens'))
        self.test = self.tokenize(os.path.join(path, 'wiki.test.tokens'))

        # Pad dictionary size to desired multiple.  For example, padding to a multiple of 8
        # is necessary to ensure Tensor Core usage for the decoder.
        pad_elem = pad_to_multiple_of - len(self.dictionary)%pad_to_multiple_of
        if pad_elem != pad_to_multiple_of:
            for i in range(pad_elem):
                self.dictionary.add_word(self.synthetic[i])

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class Wiki2:
    def __init__(self, path, pad_to_multiple_of):
        self.dataroot = path
        self.path = os.path.join(path, "wikitext-2")
        self.pad_to_multiple_of = pad_to_multiple_of

    def avail(self, download=True):
        if download:
            self.download()
        self.corpus = Corpus(self.path, self.pad_to_multiple_of)

    def download(self):
        if os.path.exists(self.path):
            return
        os.makedirs(self.path, exist_ok=True)
        subprocess.run(
            f"""
            cd {self.dataroot}
            wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
            unzip wikitext-2-v1.zip
            rm wikitext-2-v1.zip
            """,
            shell=True
        )


def wiki2(path, pad_to_multiple_of=1):
    return Wiki2(path, pad_to_multiple_of)
