from __future__ import annotations

import asyncio
import json
import os
from copy import deepcopy
from hashlib import md5
from typing import Dict, Generator, List, Tuple

from voir.instruments.gpu import get_gpu_info

from .. import pack
from ..fs import XPath
from ..merge import merge
from ..utils import select_nodes
from .executors import execute_command


def clone_with(cfg, new_cfg):
    return merge(deepcopy(cfg), new_cfg)


class Command:
    """Base class for an execution plan

    Will recursively go through its embedded `Command` to build a command
    line's arguments list to be passed to the leaf `Command`'s
    `pack.BasePackage.execute()`

    Arguments:
        pack_or_exec: `Command` or `pack.BasePackage`. If a `pack.BasePackage`, the
                      instance's `execute()` will be used to perform the
                      commands calls
        **kwargs: kwargs to be passed to the `pack_or_exec.execute()`, if a
                  `pack.BasePackage`

    Notes:
        All dynamic operations need to happen inside the argv/_argv methods.
        Those methods are guaranteed to be called with the final configuration.
    """

    def __init__(self, pack_or_exec: Command | pack.BasePackage, **kwargs) -> None:
        self._pack = None
        self.exec = None
        # used to know if the command is executed through SSH or locally
        self.remote = False

        if isinstance(pack_or_exec, Command):
            self.exec = pack_or_exec
            self._pack = None
        elif isinstance(pack_or_exec, pack.BasePackage):
            self.exec = None
            self._pack = pack_or_exec
        elif pack_or_exec is not None:
            raise TypeError(f"Need to be pack or executor not `{pack_or_exec}`")

        self._kwargs = kwargs

    @property
    def pack(self) -> pack.BasePackage:
        if self._pack:
            return self._pack
        return self.exec.pack

    def _set_pack(self, pack):
        if self._pack:
            self._pack = pack
            return True

        elif self.exec:
            return self.exec._set_pack(pack)

        return False

    def packs(self):
        if self.pack:
            yield self.pack
        else:
            yield from self.exec.packs()

    def copy(self, pack):
        """Copy the execution plan but use a different pack"""
        copy = deepcopy(self)
        copy._set_pack(pack)
        return copy

    def kwargs(self) -> Dict:
        """Return the `Command`'s kwargs to send to `pack.BasePackage.execute()`
        merged with the embeded `Command`, if any

        The `Command`'s kwargs will take priority over the embeded `Command`'s
        kwargs
        """
        kwargs = self._kwargs
        if self.exec:
            kwargs = {**self.exec.kwargs(), **kwargs}
        return kwargs

    def commands(self) -> Generator[Tuple[pack.BasePackage, List, Dict], None, None]:
        """Return a tuple of the leaf's `pack.BasePackage`, the `Command`'s list of
        command line's arguments and the `Command`'s kwargs to send to
        `pack.BasePackage.execute()`
        """
        yield self.pack, [], self.kwargs()

    async def execute(self, phase="run", timeout=False, timeout_delay=600, **kwargs):
        """Execute all the commands and return the aggregated results"""
        return await execute_command(self, phase, timeout, timeout_delay, **kwargs)


class SingleCmdCommand(Command):
    def argv(self, **kwargs) -> List:
        """Return the list of command line's arguments for this `Command`
        followed by its embedded `Command`'s list of command line's arguments

        Arguments:
            **kwargs: some `Command` might need an argument to dynamically
                      generate the list of command line's arguments
        """
        if self.exec:
            return self._argv(**kwargs) + self.exec.argv(**kwargs)
        return self._argv(**kwargs)

    def commands(self) -> Generator[Tuple[pack.BasePackage, List, Dict], None, None]:
        yield self.pack, self.argv(), self.kwargs()

    def _argv(self, **kwargs) -> List:
        del kwargs
        return []


class ListCommand(Command):
    """Execute a list of `Command`s in parallel

    Arguments:
        executors: `Command`s to be executed
        **kwargs: kwargs to be passed to the `pack.execute()`
    """

    def __init__(self, *executors: Tuple[Command], **kwargs) -> None:
        super().__init__(None, **kwargs)
        self.executors = executors

    @property
    def pack(self):
        return self.executors[0].pack

    def commands(self) -> Generator[Tuple[pack.BasePackage, List, Dict], None, None]:
        for executor in self.executors:
            yield from executor.commands()

    def packs(self):
        for exec in self.executors:
            yield from exec.packs()


class CmdCommand(SingleCmdCommand):
    """Execute a command

    Arguments:
        pack: `pack.BasePackage`'s instance from which `execute()` will be used to
              perform the command
        *cmd_argv: command line arguments list to execute
        **kwargs: kwargs to be passed to the `pack.execute()`
    """

    def __init__(self, pack: pack.BasePackage, *cmd_argv, **kwargs) -> None:
        if isinstance(pack, Command):
            raise TypeError(
                f"{self.__class__.__name__} does not accept nested"
                f" {Command.__name__}"
            )
        super().__init__(pack, **kwargs)
        self.cmd_argv = cmd_argv

    def _argv(self, **_) -> List:
        return [*self.cmd_argv]


class PackCommand(CmdCommand):
    """Execute a `Package`'s script. If not specified, the `Package`'s
    main_script will be used

    Arguments:
        pack: `Package`'s instance from which `execute()` will be used to
              perform the command
        *script_argv: script's command line arguments list. If the first
                      argument is a file or directory that can be found, the
                      file will be used instead of the `pack`'s main_script
        lazy: if true calls pack.argv to get the latest updated version of the arguments
        **kwargs: kwargs to be passed to the `pack.execute()`
    """

    def __init__(self, pack: pack.Package, *script_argv, lazy=False, **kwargs) -> None:
        script = script_argv[:1]
        if script and XPath(script[0]).exists():
            script = script[0]
            script_argv = script_argv[1:]
        else:
            script = None

        super().__init__(pack, *script_argv, **kwargs)

        self.script = script
        self.lazy = lazy

    def command_arguments(self, **kwargs):
        if self.lazy:
            return self.pack.argv
        return super()._argv(**kwargs)

    def _argv(self, **kwargs) -> List:
        script = self.script or self.pack.main_script
        if not XPath(script).is_absolute():
            abs_main = self.pack.dirs.code / script
        else:
            abs_main = script

        if not abs_main.exists():
            raise FileNotFoundError(
                f"Cannot run script or directory because it does not exist: {script}"
            )

        if abs_main.is_dir():
            script = ["-m", str(script)]
        else:
            script = [str(abs_main)]
        return script + self.command_arguments()


class VoidCommand(CmdCommand):
    """Execute nothing"""

    def __init__(self, pack: pack.BasePackage, *argv, **kwargs) -> None:
        super().__init__(pack, "true", *argv, **kwargs)


class WrapperCommand(SingleCmdCommand):
    """Wrap an `Command` with any command

    Arguments:
        executor: `Command` to be executed
        *wrapper_argv: command line arguments list
        **kwargs: kwargs to be passed to the `pack.execute()`
    """

    def __init__(self, executor: SingleCmdCommand, *wrapper_argv, **kwargs) -> None:
        super().__init__(executor, **kwargs)
        self.wrapper_argv = wrapper_argv

    def _argv(self, **kwargs) -> List:
        del kwargs
        return [*self.wrapper_argv]


class DockerRunCommand(WrapperCommand):
    """Execute an `Command` through Docker

    Arguments:
        executor: `Command` to be executed through Docker
        image: the Docker image to use
        *docker_argv: Docker command line arguments list
        **kwargs: kwargs to be passed to the `pack.execute()`
    """

    def __init__(
        self, executor: SingleCmdCommand, image: str, *docker_argv, **kwargs
    ) -> None:
        super().__init__(
            executor,
            "docker",
            "run",
            "-i",
            "--rm",
            "--network",
            "host",
            "--privileged",
            "--gpus",
            "all",
            *docker_argv,
            **kwargs,
        )
        self.image = image

    def as_container_path(self, path):
        # replace local output path with docker path
        base = self.pack.dirs.base
        path = path.replace(str(base), "/milabench/envs")

        # Replace local installation path with docker path
        install_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path = path.replace(str(install_path), "/milabench/milabench")

        return path

    def argv(self, **kwargs) -> List:
        """Return the list of command line's arguments for this `Command`
        followed by its embedded `Command`'s list of command line's arguments

        Arguments:
            **kwargs: some `Command` might need an argument to dynamically
                      generate the list of command line's arguments
        """
        script_args = self.exec.argv(**kwargs)
        docker_args = self._argv(**kwargs)

        # we are already in docker the path are correct
        if len(docker_args) == 0:
            return script_args

        # we are outisde docker
        rewritten = []
        for arg in script_args:
            # rewrite path to be inside docker
            rewritten.append(self.as_container_path(arg))

        return docker_args + rewritten

    def is_inside_docker(self):
        return os.environ.get("MILABENCH_DOCKER", None)

    def _argv(self, **kwargs) -> List:
        # if the command is executed remotely it does not matter
        # if we are inside docker or not
        if (self.image is None) or (self.is_inside_docker() and not self.remote):
            # No-op when there's no docker image to run or inside a docker
            # container
            return []

        argv = super()._argv(**kwargs)

        env = self.pack.make_env()
        for var in ("XDG_CACHE_HOME", "OMP_NUM_THREADS"):
            argv.append("--env")
            argv.append(f"{var}='{self.as_container_path(env[var])}'")

        argv.append(self.image)
        return argv


class SSHCommand(WrapperCommand):
    """Execute an `Command` through ssh

    Arguments:
        executor: `Command` to be executed through ssh
        host: host's address
        *ssh_argv: ssh command line arguments list
        user: username to use to connect to the host. By default, `pack.config`
              will be used to find the username
        key: ssh key to use to connect to the host. By default, `pack.config`
             will be used to find the username
        port: ssh port to connect to. By default port 22 will be used
        **kwargs: kwargs to be passed to the `pack.execute()`
    """

    _BIN = "ssh"

    def __init__(
        self,
        executor: SingleCmdCommand,
        host: str,
        *ssh_argv,
        user: str = None,
        key: str = None,
        port: int = 22,
        **kwargs,
    ) -> None:
        super().__init__(
            executor,
            self._BIN,
            "-oCheckHostIP=no",
            "-oStrictHostKeyChecking=no",
            "-oPasswordAuthentication=no",
            *ssh_argv,
            **kwargs,
        )
        self.host = host
        self.user = user
        self.key = key
        self.port = port
        executor.remote = not self.is_local()

    def _find_node_config(self) -> Dict:
        for n in self.pack.config["system"]["nodes"]:
            if n.get("ip") == self.host:
                return n
        return {}

    def is_local(self):
        localnode = self.pack.config["system"]["self"]

        # self is none; the node we are currently
        # on is not part of the system; we are running
        # milabench remotely, sending remote commands to
        # the main node
        return (localnode is not None) and (
            self.host in localnode["ipaddrlist"]
            or self.host  # The ip belongs to the local node
            == localnode["hostname"]  # The hostname is the local node
        )

    def _load_env(self, node):
        if node.get("env", None):
            return node["env"]
        return []

    def _argv(self, **kwargs) -> List:
        # No-op when executing on a local node
        if self.is_local():
            return []

        node = self._find_node_config()
        user = self.user or node.get("user", None)
        key = self.key or node.get("key", None)
        host = f"{user}@{self.host}" if user else self.host

        argv = super()._argv(**kwargs)
        if key:
            # scp apparently needs `-i` to be first
            argv.insert(1, f"-i{key}")
        argv.append(f"-p{self.port}")
        argv.append(host)

        argv.extend(self._load_env(node))

        return argv


class SCPCommand(SSHCommand, CmdCommand):
    _BIN = "scp"

    def __init__(
        self,
        pack: pack.BasePackage,
        host: str,
        src: str,
        *scp_argv,
        dest: str = None,
        user: str = None,
        key: str = None,
        **kwargs,
    ) -> None:
        super().__init__(pack, host, "-r", *scp_argv, user=user, key=key, **kwargs)
        self.src = src
        self.dest = dest if dest is not None else self.src

    def _load_env(self, node):
        del node
        return []

    def _argv(self, **kwargs) -> List:
        argv = super()._argv(**kwargs)

        host = argv.pop()
        argv.append(self.src)
        argv.append(f"{host}:{self.dest}")

        return argv


class TorchRunCommand(WrapperCommand):
    def __init__(self, executor: SingleCmdCommand, *torchrun_argv, **kwargs) -> None:
        super().__init__(executor, "torchrun", *torchrun_argv, **kwargs)

    def _argv(self, **kwargs):
        devices = self.pack.config.get("devices", [])
        nproc = len(devices)
        if nproc > 1:
            argv = [*super()._argv(**kwargs), f"--nproc_per_node={nproc}", "--"]
            # Check if the sub-executor targets a module or not
            cmd = next(iter(self.exec.argv()), None)
            # if the command exists and it is not a path assume it is a module
            if cmd and not XPath(cmd).exists():
                argv.append("-m")
            return argv
        return []


class VoirCommand(WrapperCommand):
    """Execute an `Command` through voir

    Arguments:
        executor: `Command` to be executed
        *voir_argv: voir command line arguments list
        **kwargs: kwargs to be passed to the `pack.execute()`
    """

    def __init__(self, executor: SingleCmdCommand, *voir_argv, **kwargs) -> None:
        super().__init__(executor, "voir", **{"setsid": True, **kwargs})
        self.voir_argv = voir_argv

    def _argv(self, **kwargs) -> List:
        argv = super()._argv(**kwargs)

        if voirconf := self.pack.config.get("voir", None):
            hsh = md5(str(voirconf).encode("utf8"))
            voirconf_file = (
                self.pack.dirs.extra
                / f"voirconf-{self.pack.tag}-{hsh.hexdigest()}.json"
            )
            with open(voirconf_file, "w") as f:
                json.dump(fp=f, obj=voirconf, indent=4)
            voir_argv = ("--config", voirconf_file)
        else:
            voir_argv = ()

        return [
            *argv,
            *voir_argv,
            *self.voir_argv,
        ]


class NJobs(ListCommand):
    """Execute n instances of the same `Command` in parallel

    Arguments:
        executor: `Command` to be executed
        n: number of times `executor` should be executed
        **kwargs: kwargs to be passed to the `pack.execute()`
    """

    def __init__(self, executor: Command, n: int, gpus: list = None, **kwargs) -> None:
        self.n = n
        if gpus is None:
            gpus = []
        self.gpus = gpus

        executors = []
        for i in range(self.n):
            gcfg = {
                "tag": [*executor.pack.config["tag"], f"{i}"],
                "job-number": i,
                "devices": [gpu["device"] for gpu in self.gpus],
            }

            run = clone_with(executor.pack.config, gcfg)
            new_pack = executor.pack.copy(run)
            executors.append(executor.copy(new_pack))

        super().__init__(*executors, **kwargs)


class SequenceCommand(ListCommand):
    """Execute a list of `Command`s in sequence
    Arguments:
        executors: `Command`s to be executed
        **kwargs: kwargs to be passed to the `pack.execute()`
    """

    async def execute(self, **kwargs):
        error_count = 0

        def on_message(msg):
            nonlocal error_count

            if msg.event == "error":
                error_count += 1

            if msg.event == "end":
                error_count += int(msg.data.get("return_code", 0))

        loop = asyncio.get_running_loop()
        loop._callbacks.append(on_message)

        for executor in self.executors:
            await executor.execute(**{**self._kwargs, **kwargs})

            if error_count > 0:
                break

        loop._callbacks.remove(on_message)
        return error_count


class PerGPU(ListCommand):
    """Execute one instance of an `Command` on each gpu

    Arguments:
        executor: `Command` to be executed
        **kwargs: kwargs to be passed to the `pack.execute()`
    """

    def __init__(self, executor: Command, gpus: list = None, **kwargs) -> None:
        if gpus is None:
            gpus = [{"device": 0, "selection_variable": "CPU_VISIBLE_DEVICE"}]

        self.devices = gpus
        executors = []
        ngpus = len(self.devices)

        for gpu in self.devices:
            gid = gpu["device"]
            gcfg = {
                "tag": [*executor.pack.config["tag"], f"D{gid}"],
                "device": gid,
                "devices": [gid] if ngpus else [],
                "env": {gpu["selection_variable"]: str(gid)},
            }
            run = clone_with(executor.pack.config, gcfg)

            new_pack = executor.pack.copy(run)
            executors.append(executor.copy(new_pack))

        super().__init__(*executors, **kwargs)


#
# Check if we need this
#   I think if we use python script.py it will load
#   the right env and we do not need the activator
#
class ActivatorCommand(SingleCmdCommand):
    def __init__(self, pack: pack.BasePackage, **kwargs):
        super().__init__(pack, **kwargs)

    def _argv(self, **_) -> List:
        return [f"{self.pack.dirs.code / 'activator'}", f"{self.pack.dirs.venv}"]


# Accelerate
class AccelerateLaunchCommand(SingleCmdCommand):
    """Execute a `pack.BasePackage` with Accelerate

    Arguments:
        pack: `pack.BasePackage`'s instance from which `execute()` will be used to
              perform the command
        *accelerate_argv: Accelerate's command line arguments list
        **kwargs: kwargs to be passed to the `pack.execute()`
    """

    def __init__(
        self, pack: pack.BasePackage, rank, *accelerate_argv, **kwargs
    ) -> None:
        super().__init__(pack, **kwargs)
        self.accelerate_argv = accelerate_argv
        self.rank = rank

    def _get_main_and_workers(self):
        max_num = self.pack.config["num_machines"]
        nodes = select_nodes(self.pack.config["system"]["nodes"], max_num)
        return nodes[0], nodes[1:]

    def _argv(self, **_) -> List:
        manager, nodes = self._get_main_and_workers()

        num_machines = max(1, len(nodes) + 1)

        ngpu = len(get_gpu_info()["gpus"].values())
        nproc = ngpu * num_machines
        assert nproc > 0, f"nproc: {nproc} num_machines: {num_machines} ngpu: {ngpu}"

        deepspeed_argv = (
            [
                "--use_deepspeed",
                "--deepspeed_multinode_launcher=standard",
                "--zero_stage=2",
            ]
            if self.pack.config["use_deepspeed"]
            else ["--multi_gpu"]
        )

        return [
            # -- Run the command in the right venv
            # This could be inside the SSH Command
            # but it would need to be repeated for Docker
            # could be its own Command like VenvCommand that execute code
            # inside a specifc venv
            f"{self.pack.dirs.code / 'activator'}",
            f"{self.pack.dirs.venv}",
            # --
            "accelerate",
            "launch",
            "--mixed_precision=fp16",
            "--dynamo_backend=no",
            f"--machine_rank={self.rank}",
            f"--num_machines={num_machines}",
            *deepspeed_argv,
            f"--gradient_accumulation_steps={self.pack.config['gradient_accumulation_steps']}",
            f"--num_cpu_threads_per_process={self.pack.config['argv']['--cpus_per_gpu']}",
            f"--main_process_ip={manager['ip']}",
            f"--main_process_port={manager['port']}",
            f"--num_processes={nproc}",
            *self.accelerate_argv,
            str(self.pack.dirs.code / "main.py"),
            *self.pack.argv,
            "--cache",
            str(self.pack.dirs.cache),
        ]
