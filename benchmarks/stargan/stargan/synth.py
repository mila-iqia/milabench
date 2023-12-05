

class SyntheticData:
    def __init__(self, generators, n, repeat):
        self.n = n
        self.repeat = repeat
        self.generators = generators
        self.data = [self.gen() for _ in range(n)]

    def gen(self):
        if isinstance(self.generators, dict):
            return {name: gen() for name, gen in self.generators.items()}
        else:
            return [gen() for gen in self.generators]

    def __getitem__(self, i):
        return self.data[i % self.n]

    def __len__(self):
        return self.n * self.repeat
