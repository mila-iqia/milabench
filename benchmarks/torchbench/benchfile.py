from milabench.pack import Package

BRANCH = "main"


class TorchBenchmarkPack(Package):
    def setup(self):
        code = self.dirs.code
        code.clone_subtree("https://github.com/pytorch/benchmark", BRANCH)
        self.install("-r", self.dirs.code / "requirements-bench.txt")
        self.run(
            "python",
            "install.py",
            "--models",
            self.config["model"],
        )

    def prepare(self):
        pass

    def launch(self, args, voirargs, env):
        args.insert(0, self.config["model"])
        return self.launch_script(
            "run.py",
            args=args,
            voirargs=voirargs,
            env=env
        )


__pack__ = TorchBenchmarkPack
