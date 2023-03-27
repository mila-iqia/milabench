from milabench.pack import Package

BRANCH = "56b90317cd9db1038b42ebdfc5bd81b1a2275cc1"


class TimmBenchmarkPack(Package):
    base_requirements = "requirements.in"
    main_script = "pytorch-image-models/train.py"

    @property
    def argv(self):
        extra_args = [
            "--data-dir", self.dirs.data,
            "--dataset", "FakeImageNet",
            "--output", self.dirs.extra / "output",
            "--checkpoint-hist", 1,
        ]
        return [*super().argv, *extra_args]

    async def install(self):
        await super().install()

        timm = self.dirs.code / "pytorch-image-models"
        if not timm.exists():
            timm.clone_subtree("https://github.com/huggingface/pytorch-image-models", BRANCH)


__pack__ = TimmBenchmarkPack
