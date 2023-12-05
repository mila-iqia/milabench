
from milabench.executors import TorchRunCommand
from milabench.pack import Package


BRANCH = "56b90317cd9db1038b42ebdfc5bd81b1a2275cc1"


class TimmBenchmarkPack(Package):
    base_requirements = "requirements.in"
    main_script = "pytorch-image-models/train.py"

    def make_env(self):
        return {
            **super().make_env(),
            "OMP_NUM_THREADS": str(self.config.get("cpus_per_gpu", 8)),
        }

    @property
    def argv(self):
        return [
            *super().argv,
            "--data-dir",
            self.dirs.data,
            "--dataset",
            "FakeImageNet",
            "--output",
            self.dirs.extra / self.logdir.name / self.tag,
            "--checkpoint-hist",
            1,
        ]

    async def install(self):
        await super().install()

        timm = self.dirs.code / "pytorch-image-models"
        if not timm.exists():
            timm.clone_subtree(
                "https://github.com/huggingface/pytorch-image-models", BRANCH
            )

    def build_run_plan(self):
        # self.config is not the right config for this
        plan = super().build_run_plan()
        return TorchRunCommand(plan, use_stdout=True)


__pack__ = TimmBenchmarkPack
