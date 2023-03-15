from milabench.pack import Package

BRANCH = "168c34e8f5c19a34c1b372004e3a840bd400a9f6"


class StableBenchmarkPack(Package):
    requirements_file = "requirements-bench.txt"

    def install(self):
        """Install requirements one by one, as baselines setup is non-standard"""
        code = self.dirs.code

        # Install swig first as it is required to compile box2d-py
        self.pip_install("-r", code / "requirements-pre.txt")

        super().install()

        code.clone_subtree("https://github.com/DLR-RM/rl-baselines3-zoo", BRANCH)

    def pin(self, *pip_compile_args, constraints:list=tuple()):
        super().pin(*pip_compile_args, requirements_file="requirements-pre.txt",
                    constraints=constraints, with_mb=False)
        super().pin(*pip_compile_args, input_files=("requirements-pre.txt",),
                    constraints=constraints)

    def run(self, args, voirargs, env):
        return self.launch("train.py", args=args, voirargs=voirargs, env=env)


__pack__ = StableBenchmarkPack
