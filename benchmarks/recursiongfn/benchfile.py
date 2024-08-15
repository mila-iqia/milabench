from milabench.pack import Package


URL = "https://github.com/recursionpharma/gflownet"
BRANCH = "bengioe-mila-demo"

class Recursiongfn(Package):
    # Requirements file installed by install(). It can be empty or absent.
    base_requirements = "requirements.in"

    # The preparation script called by prepare(). It must be executable,
    # but it can be any type of script. It can be empty or absent.
    prepare_script = "prepare.py"

    # The main script called by run(). It must be a Python file. It has to
    # be present.
    main_script = "main.py"

    # You can remove the functions below if you don't need to modify them.
    def clone(self):
        gflownet = self.dirs.code / "gflownet"
        if not gflownet.exists():
            gflownet.clone_subtree(URL, BRANCH)

    async def install(self):
        # self.clone()
        await super().install()  # super() call installs the requirements

    async def prepare(self):
        await super().prepare()  # super() call executes prepare_script



__pack__ = Recursiongfn
