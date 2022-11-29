from milabench.pack import Package

BRANCH = "168c34e8f5c19a34c1b372004e3a840bd400a9f6"


class StableBenchmarkPack(Package):
    # Requirements file installed by install(). It can be empty or absent.
    requirements_file = "requirements.txt"

    def install(self):
        """Install requirements one by one, as baselines setup is non-standard"""
        super().install()

        code = self.dirs.code
        code.clone_subtree("https://github.com/DLR-RM/rl-baselines3-zoo", BRANCH)

        reqfile = code / "requirements.txt"
        reqfile.sub("box2d-py", "# box2d-py")

        # this dependency requires swig, use conda and forget about it
        self.conda_install("-c", "conda-forge", "pybox2d")

        self.pip_install("-r", reqfile)

    def run(self, args, voirargs, env):
        return self.launch("train.py", args=args, voirargs=voirargs, env=env)


__pack__ = StableBenchmarkPack
