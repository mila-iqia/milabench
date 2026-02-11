import os
from milabench.pack import Package

from milabench.commands import AccelerateAllNodes


class Diffusion(Package):
    # Requirements file installed by install(). It can be empty or absent.
    base_requirements = "requirements.in"

    # The preparation script called by prepare(). It must be executable,
    # but it can be any type of script. It can be empty or absent.
    prepare_script = "prepare.py"

    # The main script called by run(). It must be a Python file. It has to
    # be present.
    main_script = "main.py"

    # You can remove the functions below if you don't need to modify them.

    async def install(self):
        await super().install()  # super() call installs the requirements
        
    async def prepare(self):
        await super().prepare()  # super() call executes prepare_script

    def build_run_plan(self):
        from milabench.commands import PackCommand

        if "HF_TOKEN" in os.environ or "MILABENCH_HF_TOKEN" in os.environ:
            os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", os.environ["MILABENCH_HF_TOKEN"])

        main = self.dirs.code / self.main_script
        plan = PackCommand(self, *self.argv, lazy=True)

        if False:
            plan = VoirCommand(plan, cwd=main.parent)

        return AccelerateAllNodes(plan).use_stdout()


__pack__ = Diffusion
