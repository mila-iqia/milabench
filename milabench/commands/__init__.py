from __future__ import annotations

import asyncio
import json
import os
from copy import deepcopy
from hashlib import md5
from typing import Dict, Generator, List, Tuple
from contextlib import contextmanager
import warnings

from voir.instruments.gpu import get_gpu_info

from .. import pack
from ..fs import XPath
from ..merge import merge
from ..utils import select_nodes
from .executors import execute_command
from ..system import option, DockerConfig


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

    def use_stdout(self):
        self.set_run_options(use_stdout=True)
        return self

    def set_run_options(self, **kwargs):
        self._kwargs.update(kwargs)
        return self

    @property
    def options(self):
        if self._pack:
            return self._kwargs

        if self.exec:
            # recursively retrieve options
            # this relies on dict insertion order
            opt = dict()
            opt.update(self.exec.options)
            opt.update(self._kwargs)
            return opt

        return self._kwargs

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
        yield self.pack, self.argv(), self.options

    def argv(self):
        raise NotImplementedError()

    async def execute(self, phase="run", timeout=False, timeout_delay=600, **kwargs):
        """Execute all the commands and return the aggregated results"""
        return await execute_command(self, phase, timeout, timeout_delay, **kwargs)

    def __repr__(self) -> str:
        typename = f"{type(self).__name__}"
        frags = []
        if self._pack:
            frags.append("pack")

        if self.exec:
            frags.append(repr(self.exec))

        return f"{typename}({', '.join(frags)})"


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
        yield self.pack, self.argv(), self.options

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
        # Note: it is better to use the function for it
        # since it will generate the executors when needed
        # allowing the environment to be modified/updated
        self._executors = executors

    @property
    def executors(self):
        return self._executors

    @property
    def pack(self):
        return self.executors[0].pack

    def commands(self) -> Generator[Tuple[pack.BasePackage, List, Dict], None, None]:
        for executor in self.executors:
            yield from executor.commands()

    def packs(self):
        for exec in self.executors:
            yield from exec.packs()

    def __repr__(self):
        typename = f"{type(self).__name__}"
        frags = []
        for e in self.executors:
            frags.append(repr(e))
        return f"{typename}([{', '.join(frags)}])"

    def set_run_options(self, **kwargs):
        for exec in self._executors:
            exec.set_run_options(**kwargs)
        return self

    def copy(self, pack):
        """Copy the execution plan but use a different pack"""
        copy = deepcopy(self)
        for e in copy._executors:
            e._set_pack(pack)
        return copy


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
            warnings.warn(f"Could not find `{script}` did you run `milabench install` ?")

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


class WorkingDir(WrapperCommand):
    """Wrap a command to change the working directory or force environment variables.
    
    This wrapper is usefull for commands that executes remotely or inside a container where
    the environment or the working directory might be changed by the SSH command or the container.
    """
    #  Maybe we should wrap ALL the commands with it so it can be invariant for how it is executed 
    #  and we don't have to worry about it

    def __init__(self, cmd: Command, **kwargs):
        args = [
            "env",
            "-C", str(cmd.pack.working_directory),
            "-",
            # We can also force environment variables
            f"XDG_CACHE_HOME={str(cmd.pack.dirs.cache)}",
        ]
        super().__init__(cmd, *args)


def is_inside_docker():
    return os.environ.get("MILABENCH_DOCKER", None)


class DockerRunCommand(WrapperCommand):
    """Execute an `Command` through Docker

    Arguments:
        executor: `Command` to be executed through Docker
        image: the Docker image to use
        *docker_argv: Docker command line arguments list
        **kwargs: kwargs to be passed to the `pack.execute()`
    """

    def __init__(
        self, executor: SingleCmdCommand, config: DockerConfig, *docker_argv, **kwargs
    ) -> None:
        self.config = config
        self.extra_args = docker_argv

        super().__init__(
            executor,
            **kwargs,
        )

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

    def _argv(self, **kwargs) -> List:
        # if the command is executed remotely it does not matter
        # if we are inside docker or not
        if (self.config.image is None) or (is_inside_docker() and not self.remote):
            # No-op when there's no docker image to run or inside a docker
            # container
            return []

        argv = super()._argv(**kwargs)

        env = self.pack.make_env()

        for var in ("XDG_CACHE_HOME", "OMP_NUM_THREADS"):
            if var in env:
                argv.append("--env")
                argv.append(f"{var}={self.as_container_path(env[var])}")

        return self.config.command(argv)


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
        local = self._is_local()

        # We try to detect when we need to SSH and avoid SSHing when we do not need to
        # BUT some setup we got access to had very weird config which made it easier
        # to just SSH all the time, even to itself
        if option("force_remote", int, 1):
            return False

        return local

    def _is_local(self):
        localnode = self.pack.config["system"]["self"]

        if localnode is not None:
            return (False
                # The ip belongs to the local node
                or self.host in localnode.get("ipaddrlist", [])
                # The hostname is the local node
                or self.host == localnode["hostname"]

                or self.host == localnode["ip"]
            )

        # self is none; the node we are currently
        # on is not part of the system; we are running
        # milabench remotely, sending remote commands to
        # the main node
        return False

    def _argv(self, **kwargs) -> List:
        # No-op when executing on a local node
        if self.is_local():
            return []

        node = self._find_node_config()
        user = self.user or node.get("user", None)
        key = self.key or node.get("key", None)
        host = f"{user}@{self.host}" if user else self.host

        argv = super()._argv(**kwargs)

        env = self.pack.make_env()

        # We need to set `XDG_CACHE_HOME` for datasets
        # This only works if server can `AcceptEnv`
        # for k in env.keys():
        #     argv.append(f"-oSendEnv={k}")

        # Those mean nothing inside docker
        # TODO: is the XDG_CACHE_HOME still needed or was it taken care somehwere else?
        envs = []
        # if not is_inside_docker():
        #     envs = [
        #         "env",
        #         "-C", str(self.pack.working_directory),
        #         "-",
        #         f"XDG_CACHE_HOME={str(self.pack.dirs.cache)}",
        #     ]

        argv.extend(["-oPasswordAuthentication=no"])
        argv.extend(["-p", str(self.port)])

        if key:
            argv.append(f"-i{key}")
        argv.append(host)

        # We need to set the working directory here because multinode
        # will not use the process cwd
        return (argv
            + envs
        )


class SCPCommand(SSHCommand, CmdCommand):
    _BIN = "scp"

    def __init__(
        self,
        pack: pack.BasePackage,
        host: str,
        directory: str,
        *scp_argv,
        user: str = None,
        key: str = None,
        **kwargs,
    ) -> None:
        super().__init__(pack, host, "-r", *scp_argv, user=user, key=key, **kwargs)
        self.dir = directory

    def _argv(self, **kwargs) -> List:
        argv = super()._argv(**kwargs)

        host = argv.pop()
        argv.append(self.dir)
        argv.append(f"{host}:{self.dir}")

        return argv



class TorchrunAllGPU(WrapperCommand):
    def __init__(self, executor: SingleCmdCommand, *torchrun_argv, module=False, **kwargs) -> None:
        # Some vendors force us to have weird venv that can resolve weirdly
        # use absolute paths to avoid issues

        self.binfolder = executor.pack.config["dirs"]["venv"]
        self.module = module

        # benchrun is a wrapper around torchrun
        # which insert voir file descritor
        super().__init__(
            executor, self.executable, *torchrun_argv, **kwargs
        )

    @property
    def executable(self):
        return f"{self.binfolder}/bin/benchrun"

    def get_cmd(self):
        return next(iter(self.exec.argv()), None)

    def device_count(self):
        devices = self.pack.config.get("devices", [])
        nproc = len(devices)
        return nproc

    def should_wrap(self):
        return self.device_count() > 1

    def _argv(self, **kwargs):
        devices = self.pack.config.get("devices", [])
        nproc = len(devices)

        if self.should_wrap():
            # spawn,fork,forkserver
            multi_gpu_args = (
                f"--nproc-per-node={nproc}",
            )

            if self.pack.config["plan"]["method"] == "per_gpu":
                multi_gpu_args = tuple()

            argv = [

                *super()._argv(**kwargs),
                *multi_gpu_args
                # "--start-method=forkserver"
            ]

            # Check if the sub-executor targets a module or not
            cmd = self.get_cmd()

            if self.module:
                argv.append("-m")

            else:
                if cmd:
                    # python or voir; tell it to not prepend python since we are doing it
                    if cmd in ("python", "voir"):
                        argv.append("--no-python")

                    # if the command exists and it is not a path assume it is a module
                    # script is not a file, maybe it is a module
                    elif not XPath(cmd).exists() and "/" not in cmd:
                        argv.append("-m")

                # everything after torchrun args are script args
                argv.append("--")

            return argv
        return []


def node_address(node):
    """Favour Hostname as it is the most consistent name across machines"""
    host = node.get("hostname")
    ip = node.get("ip")
    return ip or host


class ClientServer(ListCommand):
    # We have to pick which one is pushing metrics
    # and if one is not pushing anything

    #
    # TODO: Allow the client and server to be on different nodes
    #

    @staticmethod
    def new_pack(pack, tag, has_logs=True):
        config = pack.config
        tags = [*pack.config["tag"], tag]

        if not has_logs:
            tags.append("nolog")

        run = clone_with(config, {"tag": tags})
        return pack.copy(run)

    @staticmethod
    def new_client_pack(pack, has_logs=True):
        return ClientServer.new_pack(pack, 'client', has_logs)

    @staticmethod
    def new_server_pack(pack, has_logs=True):
        return ClientServer.new_pack(pack, 'server', has_logs)

    def __init__(self, pack, client, server, different_nodes=False, **kwargs):
        self.pack = pack
        self.client = client 
        self.server = server

        # TODO: Implement me
        # There is a problem if we are running the client or server on the milabench host
        # we do not want to run 2 docker container if one of the host is already running one
        #
        # if different_nodes:
        #     config = self.pack.config
        #     nodes = select_nodes(config["system"]["nodes"], 2)
            
        #     cn, sn = nodes
        #     key = config["system"].get("sshkey")
        #     config = DockerConfig(**config["system"].get("docker", {}))

        #     self.client = SSHCommand(
        #         host=node_address(cn),
        #         user=cn["user"],
        #         key=key,
        #         port=cn.get("sshport", 22)
        #         executor=DockerRunCommand(self.client, config)
        #     )

        #     self.server = SSHCommand(
        #         host=node_address(sn),
        #         user=sn["user"],
        #         key=key,
        #         port=sn.get("sshport", 22)

        #         executor=DockerRunCommand(self.server, config)
        #     )

        super().__init__(self.client, self.server)


class ForeachNode(ListCommand):
    def __init__(self, executor: Command, use_docker=True, **kwargs) -> None:
        super().__init__(None, **kwargs)
        self.options.update(kwargs)
        self.executor = executor
        self.base_tags = self.executor.pack.config["tag"]
        self.use_docker = use_docker

    def make_new_node_pack(self, rank, node, base) -> "BasePackage":
        """Make a new environment/config for the run"""
        config = base.pack.config
        tags = [*self.base_tags, node["name"]]

        # Workers do not send training data
        # tag it as such so validation can ignore this pack
        if rank != 0:
            tags.append("nolog")

        run = clone_with(config, {"tag": tags})
        return base.pack.copy(run)

    def make_new_node_executor(self, rank, node, base):
        """Make a new environment and create a new executor for the node"""
        pack = self.make_new_node_pack(rank, node, base)
        return base.copy(pack)

    def single_node(self):
        return self.executor

    @property
    def executors(self):
        """Build the executor lazyly when necessary so we get the latest config"""
        executors = []

        config = self.executor.pack.config

        max_num = config.get("num_machines", 1)
        self.nodes = select_nodes(config["system"]["nodes"], max_num)
        key = config["system"].get("sshkey")

        # useless in single node setups
        if len(self.nodes) == 1 or max_num == 1:
            return [self.single_node()]

        for rank, node in enumerate(self.nodes):
            options = dict()

            # Hummm...
            if rank == 0:
                options = dict(
                    setsid=True,
                    **self.options
                )

            bench_cmd = self.make_new_node_executor(rank, node, self.executor)

            # Hum, I think the docker wrapping could be done somewhere else
            # so we do not need that use_docker flag
            if self.use_docker:
                docker_cmd = DockerRunCommand(bench_cmd, DockerConfig(**config["system"].get("docker", {})))
            else:
                docker_cmd = bench_cmd

            worker = SSHCommand(
                host=node_address(node),
                user=node["user"],
                key=key,
                port=node.get("sshport", 22),
                executor=docker_cmd,
                **options
            )
            executors.append(worker)
        return executors

    def set_run_options(self, **kwargs):
        self.executor.set_run_options(**kwargs)
        return self

    def copy(self, pack):
        """Copy the execution plan but use a different pack"""
        copy = deepcopy(self)
        copy.executor._set_pack(pack)
        return copy

class TorchrunAllNodes(ForeachNode):
    """executes torchrun on multiple machines"""

    @staticmethod
    def make_base_executor(cls, executor, *args, **kwargs):
        config = executor.pack.config
        max_num = config.get("num_machines", 1)
        nodes = select_nodes(config["system"]["nodes"], max_num)

        main = nodes[0]

        # node[port] is for SSH
        main_host = node_address(main)
        # add them as option so we could tweak them if necessary
        main_port = option("torchrun.port", int, default=29400)
        backend = option("torchrun.backend", str, default="static")
        filters = option("torchrun.local_ranks_filder", str, default="0")

        if backend == "c10d":
            print("Warning: c10d can select the wrong node for RANK=0")

        main_addr = f"{main_host}:{main_port}"

        config = executor.pack.config

        multi_gpu_args = (
            f"--nnodes={len(nodes)}",
            f"--rdzv-backend={backend}",
            f"--rdzv-endpoint={main_addr}",
            f"--master-addr={main_host}",
            f"--master-port={main_port}",
            f"--local-ranks-filter={filters}",
        )

        if config["plan"]["method"] == "per_gpu":
            multi_gpu_args = tuple()

        return cls(
            executor,
            *multi_gpu_args,
            *args,
            **kwargs
        )

    def make_new_node_executor(self, rank, node, base):
        """Make a new environment and create a new executor for the node"""
        executor: TorchrunAllGPU = super().make_new_node_executor(rank, node, base)

        # Specify the node rank so rank 0 is consistently on the local node
        new_args = list(executor.wrapper_argv) +  [
            f"--node-rank={rank}",
            f"--local-addr={node['ip']}",
            f"--rdzv-conf=rank={rank}",
        ]
        executor.wrapper_argv = new_args

        return executor

    def __init__(self, executor: Command, *args, **kwargs) -> None:
        base_exec = TorchrunAllNodes.make_base_executor(
            TorchrunAllGPU,
            executor,
            *args,
            **kwargs
        )
        super().__init__(base_exec)


TorchRunCommand = TorchrunAllGPU

use_voir = True


@contextmanager
def disable_voir(enabled):
    global use_voir
    old = use_voir
    use_voir = enabled
    yield
    use_voir = old


class VoirCommand(WrapperCommand):
    """Execute an `Command` through voir

    Arguments:
        executor: `Command` to be executed
        *voir_argv: voir command line arguments list
        **kwargs: kwargs to be passed to the `pack.execute()`
        module: bool use voir module instead of voir wrapper.
            this is useful for torchrun since when a module is used
            the main torchrun process can be reused for rank=0 enabling
            voir to work using file descriptor.
    """

    def __init__(self, executor: SingleCmdCommand, *voir_argv, module=False, **kwargs) -> None:
        # Some vendors force us to have weird venv that can resolve weirdly
        # use absolute paths to avoid issues
        binfolder = executor.pack.config["dirs"]["venv"]
        voir = f"{binfolder}/bin/voir"

        if module:
            voir = "voir"

        super().__init__(
            executor, voir, **{"setsid": True, **kwargs}
        )
        self.voir_argv = voir_argv

    def _argv(self, **kwargs) -> List:
        argv = super()._argv(**kwargs)

        if not use_voir:
            # voir replace python
            return ["python"]

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
        if gpus is None or len(gpus) == 0:
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
        return [activator_script(), f"{self.pack.dirs.venv}", f"{self.pack.dirs.cache}"]



class AccelerateAllNodes(ForeachNode):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def single_node(self):
        ngpu = len(self.executor.pack.config.get("devices", []))

        # Multi GPU
        if ngpu > 1:
            return AccelerateLaunchCommand(self.executor, rank=0, **self.options)

        # Single GPU
        return self.executor

    def make_new_node_executor(self, rank, node, base):
        config = base.pack.config

        pack = self.make_new_node_pack(rank, node, base)
        executor = base.copy(pack)

        return DockerRunCommand(
            AccelerateLaunchCommand(executor, rank=rank, **self.options),
            DockerConfig(**config["system"].get("docker", {})),
        )


def activator_script():
    """Scripts that activate the venv just before executing a script

    Useful for commands that SSH somewhere and need to execute a command in a particular venv
    """

    path = XPath(__file__).parent.parent / "scripts" / "activator"
    assert path.exists()
    return str(path)


class SimpleCommand(SingleCmdCommand):
    def __init__(self, pack_or_exec: Command | pack.BasePackage, *args, **options) -> None:
        super().__init__(pack_or_exec, **options)
        self.args = args

    def argv(self):
        return list(self.pack.argv) + list(self.args)


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
        max_num = self.pack.config.get("num_machines", 1)
        nodes = select_nodes(self.pack.config["system"]["nodes"], max_num)
        return nodes[0], nodes[1:]

    def _argv(self, **_) -> List:
        manager, nodes = self._get_main_and_workers()

        num_machines = max(1, len(nodes) + 1)

        # Cant do that maybe this run is constrained
        # ngpu = len(get_gpu_info()["gpus"].values())

        ngpu = len(self.pack.config["devices"])
        nproc = ngpu * num_machines
        assert nproc > 0, f"nproc: {nproc} num_machines: {num_machines} ngpu: {ngpu}"

        if self.pack.config.get("use_deepspeed", False):
            deepspeed_argv = [
                "--use_deepspeed",
                "--deepspeed_multinode_launcher=standard",
                "--zero_stage=2",
            ]
        elif ngpu > 1:
            deepspeed_argv = ["--multi_gpu"]
        else:
            deepspeed_argv = []

        cpu_per_process = self.pack.resolve_argument('--cpus_per_gpu', 4)
        main_port = option("torchrun.port", int, default=29400)

        return [
            # -- Run the command in the right venv
            # This could be inside the SSH Command
            # but it would need to be repeated for Docker
            # could be its own Command like VenvCommand that execute code
            # inside a specifc venv
            activator_script(),
            f"{self.pack.dirs.venv}",
            f"{self.pack.dirs.cache}",
            # --
            "accelerate",
            "launch",
            "--mixed_precision=bf16",
            "--dynamo_backend=no",
            f"--machine_rank={self.rank}",
            f"--num_machines={num_machines}",
            *deepspeed_argv,
            f"--gradient_accumulation_steps={self.pack.config.get('gradient_accumulation_steps', 1)}",
            f"--num_cpu_threads_per_process={cpu_per_process}",
            f"--main_process_ip={manager['ip']}",
            f"--main_process_port={main_port}",
            f"--num_processes={nproc}",
            *self.accelerate_argv,
        ]
