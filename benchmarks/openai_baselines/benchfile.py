from milabench.pack import Package


class TheBenchmark(Package):
    # Requirements file installed by install(). It can be empty or absent.
    requirements_file = "requirements.txt"

    # The preparation script called by prepare(). It must be executable,
    # but it can be any type of script. It can be empty or absent.
    prepare_script = "prepare.py"

    # The main script called by run(). It must be a Python file. It has to
    # be present.
    main_script = "main.py"

    # You can remove the functions below if you don't need to modify them.

    def make_env(self):
        # Return a dict of environment variables for prepare_script and
        # main_script.
        return super().make_env()

    def install(self):
        """Install requirements one by one, as baselines setup is non-standard"""
        if self.requirements_file is None:
            return

        reqs = self.dirs.code / self.requirements_file

        if not reqs.exists():
            return

        with open(reqs, 'r') as requirements:
            for requirement in requirements.readlines():
                self.pip_install(requirement)

    def prepare(self):
        super().prepare()  # super() call executes prepare_script

    def run(self, args, voirargs, env):
        arguments = self.config['args']
        args.extend(arguments)
        return super().run(args, voirargs, env)



__pack__ = TheBenchmark
