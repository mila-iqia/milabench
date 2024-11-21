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
        super().install()  # super() call installs the requirements

    def prepare(self):
        super().prepare()  # super() call executes prepare_script

    def run(self, args, voirargs, env):
        # You can insert new arguments to args/voirargs or change the env,
        # although changing the env is a bit simpler if you modify make_env
        return super().run(args, voirargs, env)
        # Note: run() must return a running process, so make sure not to lose
        # the return value of super() here.


__pack__ = TheBenchmark
