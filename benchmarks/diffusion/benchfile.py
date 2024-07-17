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

    def make_env(self):
        return {
            **super().make_env(),
            "OMP_NUM_THREADS": str(self.config.get("cpus_per_gpu", 8)),
        }

    async def install(self):
        await super().install()  # super() call installs the requirements
        
    async def prepare(self):
        await super().prepare()  # super() call executes prepare_script

    def build_run_plan(self):
        plan = super().build_run_plan()

        return AccelerateAllNodes(plan, use_stdout=True)


__pack__ = Diffusion
