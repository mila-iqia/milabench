from milabench.pack import Package

BRANCH = "ff7114655294aa3ba57127a260dbd1ef5190f610"


class TorchBenchmarkPack(Package):
    def install(self):
        code = self.dirs.code
        code.clone_subtree("https://github.com/pytorch/benchmark", BRANCH)
        self.pip_install("-r", code / "requirements-bench.txt")
        self.python("install.py", "--models", self.config["model"])

    def run(self, args, voirargs, env):
        args.insert(0, self.config["model"])
        super().run
        return self.launch("run.py", args=args, voirargs=voirargs, env=env)


__pack__ = TorchBenchmarkPack
