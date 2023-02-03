from milabench.pack import Package

BRANCH = "168c34e8f5c19a34c1b372004e3a840bd400a9f6"


class StableBenchmarkPack(Package):
    # Requirements file installed by install(). It can be empty or absent.
    requirements_file = "requirements.txt"

    def install(self):
        """Install requirements one by one, as baselines setup is non-standard"""
        # box2d-py requires swig
        self.conda_install("-c", "conda-forge", "swig")

        super().install()

        code = self.dirs.code
        code.clone_subtree("https://github.com/DLR-RM/rl-baselines3-zoo", BRANCH)

    def run(self, args, voirargs, env):
        return self.launch("train.py", args=args, voirargs=voirargs, env=env)


__pack__ = StableBenchmarkPack
