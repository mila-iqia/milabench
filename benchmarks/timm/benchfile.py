from milabench.commands import TorchRunCommand
from milabench.pack import Package


BRANCH = "cb0e4391beedcc5ac3ae4bce16561b95c326f32c"


class TimmBenchmarkPack(Package):
    base_requirements = "requirements.in"
    main_script = "pytorch-image-models/train.py"

    @property
    def working_directory(self):
        return self.dirs.code / "pytorch-image-models"

    @property
    def argv(self):
        return [
            *super().argv,
            "--output",
            self.dirs.data / "FakeImageNet",
            "--checkpoint-hist",
            1,
        ]

    async def install(self):
        timm = self.dirs.code / "pytorch-image-models"
        if not timm.exists():
            timm.clone_subtree(
                "https://github.com/huggingface/pytorch-image-models", BRANCH
            )

        # install the rest, which might override what TIMM specified
        await super().install()

    def build_run_plan(self):
        import milabench.commands as cmd
        main = self.dirs.code / self.main_script

        # torchrun ... -m voir ... train_script ...
        return TorchRunCommand(
            cmd.VoirCommand(cmd.PackCommand(self, *self.argv, lazy=True), cwd=main.parent, module=True),
            module=True
        )


__pack__ = TimmBenchmarkPack
