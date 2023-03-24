from milabench.pack import Package

BRANCH = "168c34e8f5c19a34c1b372004e3a840bd400a9f6"


class StableBenchmarkPack(Package):
    base_requirements = ["requirements-pre.in", "requirements.in"]
    main_script = "rlzoo/train.py"

    async def install(self):
        await super().install()

        zoo = self.dirs.code / "rlzoo"
        if not zoo.exists():
            zoo.clone_subtree("https://github.com/DLR-RM/rl-baselines3-zoo", BRANCH)


__pack__ = StableBenchmarkPack
