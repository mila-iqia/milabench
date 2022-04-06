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

        # this dependency requires swig, use conda and forget about it
        self.conda_install('-c', 'conda-forge', 'pybox2d')

        with open(code / "requirements.txt", "r") as requirements:
            for requirement in requirements.readlines():

                # Already installed, this requires swig
                # but we already installed a binary built using conda
                if requirement.startswith('box2d-py'):
                    continue

                if '# tmp fix: until compatibility with panda-gym v2' in requirement:
                    continue

                self.pip_install(requirement)

    def prepare(self):
        super().prepare()  # super() call executes prepare_script

    def run(self, args, voirargs, env):
        # arguments = self.config['args']
        # args.extend(arguments)
        super().run
        return self.launch("train.py", args=args, voirargs=voirargs, env=env)


__pack__ = StableBenchmarkPack
