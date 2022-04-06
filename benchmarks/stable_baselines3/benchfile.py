from milabench.pack import Package


BRANCH = "2509bca3d6cc8a84d3330b0699dd9dc34b06524a"


class StableBenchmarkPack(Package):
    # Requirements file installed by install(). It can be empty or absent.
    requirements_file = "requirements.txt"

    def install(self):
        """Install requirements one by one, as baselines setup is non-standard"""
        super().install()

        code = self.dirs.code
        code.clone_subtree("https://github.com/DLR-RM/rl-baselines3-zoo", BRANCH)

        self.conda_install('-c', 'conda-forge', 'pybox2d')
        self.pip_install("-r", code / "requirements.txt")

    def prepare(self):
        super().prepare()  # super() call executes prepare_script

    def run(self, args, voirargs, env):
        # arguments = self.config['args']
        # args.extend(arguments)
        super().run
        return self.launch("run.py", args=args, voirargs=voirargs, env=env)



__pack__ = StableBenchmarkPack
