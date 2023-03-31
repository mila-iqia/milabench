from milabench.fs import XPath
from milabench.pack import Package

BRANCH = "69d22b99ec02ff868dbc1170e39686935f9d1274"


class DLRMBenchmarkPack(Package):
    base_requirements = ["requirements_repo.in", "requirements.in"]
    main_script = "dlrm/dlrm_s_pytorch.py"

    async def install(self):
        dlrm:XPath = self.dirs.code / "dlrm"
        if not dlrm.exists():
            dlrm.clone_subtree("https://github.com/facebookresearch/dlrm", BRANCH)
            (dlrm / "requirements.txt").rename(self.dirs.code / "requirements_repo.in")

        await super().install()


__pack__ = DLRMBenchmarkPack


class TheBenchmark(Package):
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
        # Return a dict of environment variables for prepare_script and
        # main_script.
        return super().make_env()

    async def install(self):
        await super().install()  # super() call installs the requirements

    async def prepare(self):
        await super().prepare()  # super() call executes prepare_script

    async def run(self):
        return await super().run()
