from milabench.pack import Package
from milabench.commands import TorchrunAllNodes, TorchrunAllGPU, ListCommand


SOURCE_DIR = "src"
# Fix https://github.com/facebookresearch/dinov2/pull/281
REPO_URL = "https://github.com/pathologywatch/dinov2"
BRANCH = "733d285dccf407ebd2c0e97f899d67206ede44a5"


class Dinov2(Package):
    # Requirements file installed by install(). It can be empty or absent.
    base_requirements = "requirements.in"

    # The preparation script called by prepare(). It must be executable,
    # but it can be any type of script. It can be empty or absent.
    prepare_script = "prepare.py"

    # The main script called by run(). It must be a Python file. It has to
    # be present.
    main_script = "main.py"

    # You can remove the functions below if you don't need to modify them.

    @property
    def working_directory(self):
        return self.dirs.code / SOURCE_DIR

    def make_env(self):
        # Return a dict of environment variables for prepare_script and
        # main_script.
        env = super().make_env()
        return env

    async def install(self):
        await super().install()

        source_destination = self.dirs.code / SOURCE_DIR
        if not source_destination.exists():
            source_destination.clone_subtree(
                REPO_URL, BRANCH
            )

    async def prepare(self):
        await super().prepare()  # super() call executes prepare_script

    def build_run_plan(self):
        # self.config is not the right config for this
        plan = super().build_run_plan()

        return TorchrunAllNodes(plan).use_stdout()



__pack__ = Dinov2
