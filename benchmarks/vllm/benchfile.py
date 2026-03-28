import os

from milabench.pack import Package
import milabench.commands as cmd
from milabench.utils import assemble_options


class VLLMParallel(cmd.Command):
    """This is like a torchrun but it handles the tensor parallel as well"""
    def __init__(self, base_cmd, dataparallel_gpu, tensorparallel_gpu):
        # assert dataparallel_gpu * tensorparallel_gpu <= ngpu

        self.local_world = ngpu / tensorparallel_gpu
        self.world_size = self.local_world * self.num_machine

    def rank():
        os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(self.world_size)

        os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
        os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)


class VLLM(Package):
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
        env = super().make_env()
        env["XDG_CACHE_HOME"] = str(self.dirs.cache)
    
        env["FLASHINFER_CACHE_DIR "] = str(self.dirs.cache / "flashinfer")
        env["FLASHINFER_CUBIN_DIR "] = str(self.dirs.cache / "flashinfer" / "cubins")

        # flashinfer defaults its workspace to ~/.cache/flashinfer/<version>
        # which may not be writable on compute nodes; redirect to our cache dir
        env["FLASHINFER_WORKSPACE_DIR"] = os.path.join(
            str(self.dirs.cache), "flashinfer", "workspace"
        )
        return env

    async def install(self):
        await super().install()

    async def prepare(self):
        await super().prepare()  # super() call executes prepare_script

    def client_argv(self, prepare=False):
        args = self.config.get("client", {}).get("argv", [])

        if prepare and isinstance(args, dict):
            args['--num-prompts'] = 1
            # args['--prepare'] = True

        return assemble_options(args)

    def server_argv(self, prepare=False):
        return assemble_options(self.config.get("server", {}).get("argv", []))

    @property
    def argv(self):
        return self.server_argv() + ['--'] + self.client_argv()

    def build_run_plan(self):
        main = self.dirs.code / self.main_script
        pack = cmd.PackCommand(self, *self.argv, lazy=True)
        return cmd.VoirCommand(pack, cwd=main.parent).use_stdout()


__pack__ = VLLM
