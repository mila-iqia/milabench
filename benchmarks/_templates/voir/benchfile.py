from milabench.pack import Package


SOURCE_DIR = "src"
REPO_URL = "https://github.com/Delaunay/extern_example.git"
BRANCH = "a3130b3251d3a909d2a7c59efa948a34e52cbbbb"


class Template(Package):
    # Requirements file installed by install(). It can be empty or absent.
    base_requirements = "requirements.in"

    # The preparation script called by prepare(). It must be executable,
    # but it can be any type of script. It can be empty or absent.
    prepare_script = "prepare.py"

    # The main script called by run(). It must be a Python file. It has to
    # be present.
    main_script = f"{SOURCE_DIR}/main.py"

    # You can remove the functions below if you don't need to modify them.

    def make_env(self):
        # Return a dict of environment variables for prepare_script and
        # main_script.
        return super().make_env()

    async def install(self):
        await super().install()

        source_destination = self.dirs.code / SOURCE_DIR
        if not source_destination.exists():
            source_destination.clone_subtree(
                REPO_URL, BRANCH
            )

    async def prepare(self):
        await super().prepare()  # super() call executes prepare_script



__pack__ = Template
