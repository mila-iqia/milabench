import os

from milabench.pack import Package


class Dimenet(Package):

    # Pytorch does this:
    #
    #   CUDA_HOME = _find_cuda_home() if (torch.cuda._is_compiled() and torch.version.cuda) else None
    #
    #   This means that if you install Pytorch CPU even with CUDA HOME set it will ignore CUDA
    #
    # When doing a unpinned install, torch CPU get installed instead of CUDA, but it should be CUDA
    #
    #
    #

    # Requirements file installed by install(). It can be empty or absent.
    base_requirements = ["requirements.in"]

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
        env = super().make_env()

        # In the case of compiling pytorch geometric
        # we want to compile for conda support even if no GPUs are availble
        env.update({
            "FORCE_CUDA": "1",
            "CUDA_HOME": os.getenv("CUDA_HOME", '/usr/local/cuda/')
        })
        return env

    async def install(self):
        await super().install()  # super() call installs the requirements

    async def prepare(self):
        await super().prepare()  # super() call executes prepare_script


__pack__ = Dimenet
