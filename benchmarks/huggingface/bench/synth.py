import torch

generators = {}


def register_generator(fn):
    generators[fn.__name__.lstrip("gen_")] = fn
    return fn


class SyntheticData:
    def __init__(self, generators, n, repeat):
        self.n = n
        self.repeat = repeat
        self.generators = generators
        self.data = [self.gen() for _ in range(n)]

    def gen(self):
        return {name: gen() for name, gen in self.generators.items()}

    def __getitem__(self, i):
        return self.data[i % self.n]

    def __len__(self):
        return self.n * self.repeat


def vocabgen(info):
    def gen():
        return torch.randint(0, info.config.vocab_size, (info.train_length,))

    return gen


@register_generator
def gen_AutoModelForCausalLM(info):
    return {
        "input_ids": vocabgen(info),
        "labels": vocabgen(info),
    }


@register_generator
def gen_AutoModelForSeq2SeqLM(info):
    return gen_AutoModelForCausalLM(info)


@register_generator
def gen_AutoModelForMaskedLM(info):
    return gen_AutoModelForCausalLM(info)


@register_generator
def gen_AutoModelForAudioClassification(info):
    extractor = info.extractor_class()

    def igen():
        wav = list(torch.rand(10000) * 2 - 1)
        dat = extractor(wav, sampling_rate=info.sampling_rate, return_tensors="pt")[
            "input_features"
        ]
        return dat[0]

    def ogen():
        return torch.randint(0, info.config.num_labels, ())

    return {
        "input_features": igen,
        "labels": ogen,
    }
