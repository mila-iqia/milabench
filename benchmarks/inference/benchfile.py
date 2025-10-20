from milabench.pack import Package
import milabench.commands as cmd 
from milabench.utils import assemble_options


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
        # Return a dict of environment variables for prepare_script and
        # main_script.
        return super().make_env()

    async def install(self):
        await super().install()

    async def prepare(self):
        await super().prepare()  # super() call executes prepare_script

    @property
    def client_argv(self):
        return assemble_options(self.config.get("client", {}).get("argv", []))

    @property
    def server_argv(self):
        return assemble_options(self.config.get("server", {}).get("argv", []))
    
    @property
    def argv(self):
        return self.server_argv + ['--'] + self.client_argv

    def build_run_plan(self):
        main = self.dirs.code / self.main_script
        pack = cmd.PackCommand(self, *self.argv, lazy=True)
        return cmd.VoirCommand(pack, cwd=main.parent).use_stdout()
        # return super().build_run_plan().use_stdout()


        # we can send early stop events when we want to stop one
        # but what about the other ?
        # What will end the server in particular ?
        # The client might be able to send a stop server
        # but that is unlikely

        # client_pack = cmd.ClientServer.new_client_pack(self)

        # server_pack = cmd.ClientServer.new_server_pack(self)

        # client_cmd = cmd.PackCommand(
        #     client_pack, self.client_main, *self.client_argv
        # )

        # server_cmd = cmd.PackCommand(
        #     server_pack, self.server_main, *self.server_argv
        # )

        # return cmd.ClientServer(
        #     self,
        #     client_cmd,
        #     server_cmd,
        # )

#
# 
#

__pack__ = VLLM
