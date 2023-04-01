from milabench.pack import Package

BRANCH = "56b90317cd9db1038b42ebdfc5bd81b1a2275cc1"


class TimmBenchmarkPack(Package):
    base_requirements = "requirements.in"
    main_script = "pytorch-image-models/train.py"

    def make_env(self):
        return {
            **super().make_env(),
            "OMP_NUM_THREADS": str(self.config.get("cpus_per_gpu", 8))
        }

    @property
    def argv(self):
        return [
            *super().argv,
            "--data-dir", self.dirs.data,
            "--dataset", "FakeImageNet",
            "--output", self.dirs.extra / self.logdir.name / self.tag,
            "--checkpoint-hist", 1,
        ]

    async def install(self):
        await super().install()

        timm = self.dirs.code / "pytorch-image-models"
        if not timm.exists():
            timm.clone_subtree("https://github.com/huggingface/pytorch-image-models", BRANCH)

    async def run(self):
        main = self.dirs.code / self.main_script
        assert main.exists()
        nproc = self.config.get("nproc", 1)
        if nproc > 1:
            return await self.voir(
                self.main_script,
                args=self.argv,
                cwd=main.parent,
                wrapper=["torchrun", f"--nproc_per_node={nproc}", "-m"],
                use_stdout=True,
            )
        else:
            return await self.voir(self.main_script, args=self.argv, cwd=main.parent)


__pack__ = TimmBenchmarkPack
