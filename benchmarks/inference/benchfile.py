from milabench.pack import Package
import milabench.commands as cmd 
from milabench.utils import assemble_options


class Inference(Package):
    # Requirements file installed by install(). It can be empty or absent.
    base_requirements = "requirements.in"

    # The preparation script called by prepare(). It must be executable,
    # but it can be any type of script. It can be empty or absent.
    prepare_script = "prepare.py"

    # The main script called by run(). It must be a Python file. It has to
    # be present.
    main_script = f"main.py"

    # You can remove the functions below if you don't need to modify them.

    def make_env(self):
        # Return a dict of environment variables for prepare_script and
        # main_script.
        return super().make_env()

    async def install(self):
        await super().install()

    async def prepare(self):
        await super().prepare()  # super() call executes prepare_script

    def build_run_plan(self):
        main = self.dirs.code / self.main_script
        pack = cmd.PackCommand(self, *self.argv, lazy=True)
        return cmd.VoirCommand(pack, cwd=main.parent).use_stdout()

    @property
    def prepare_argv(self):
        return ["--prepare"] + self.argv

    def build_prepare_plan(self):
        # Run the same script but with fast arguments
        main = self.dirs.code / self.main_script
        pack = cmd.PackCommand(self, *self.prepare_argv, lazy=True)
        return cmd.VoirCommand(pack, cwd=main.parent).use_stdout()


__pack__ = Inference
