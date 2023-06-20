from __future__ import annotations

import asyncio
import os
import json
import socket
from copy import deepcopy
from hashlib import md5
from typing import Dict, Generator, List, Tuple

from . import pack
from .alt_async import destroy
from .fs import XPath
from .merge import merge
from .metadata import machine_metadata


def clone_with(cfg, new_cfg):
    return merge(deepcopy(cfg), new_cfg)


async def force_terminate(pack, delay):
    await asyncio.sleep(delay)
    for proc in pack.processes:
        ret = proc.poll()
        if ret is None:
            await pack.message(
                f"Terminating process because it ran for longer than {delay} seconds."
            )
            destroy(proc)


class Executor():
    """Base class for an execution plan

    Will recursively go through its embedded `Executor` to build a command
    line's arguments list to be passed to the leaf `Executor`'s
    `BasePackage.execute()`

    Arguments:
        pack_or_exec: `Executor` or `BasePackage`. If a `BasePackage`, the
                      instance's `execute()` will be used to perform the
                      commands calls
        **kwargs: kwargs to be passed to the `pack_or_exec.execute()`, if a
                  `BasePackage`
                  
    Notes:
        All dynamic operations need to happen inside the argv/_argv methods.
        Those methods are guaranteed to be called with the final configuration.
    """
    def __init__(self, pack_or_exec: Executor | pack.BasePackage, **kwargs) -> None:
        if isinstance(pack_or_exec, Executor):
            self.exec = pack_or_exec
            self._pack = None
        elif isinstance(pack_or_exec, pack.BasePackage):
            self.exec = None
            self._pack = pack_or_exec
        else:
            raise TypeError(f"Need to be pack or executor {pack_or_exec}")

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

    def copy(self, pack):
        """Copy the execution plan but use a different pack"""
        copy = deepcopy(self)
        copy._set_pack(pack)
        return copy

    def commands(self) -> Generator[Tuple[pack.BasePackage, List, Dict], None, None]:
        """Return a tuple of the leaf's `BasePackage`, the `Executor`'s list of
        command line's arguments and the `Executor`'s kwargs to send to
        `BasePackage.execute()`
        """
        raise NotImplemented()

    async def execute(self, timeout=False, timeout_delay=600, **kwargs):
        """Execute all the commands and return the aggregated results
        """
        coro = []

        for pack, argv, _kwargs in self.commands():
            await pack.send(event="config", data=pack.config)
            await pack.send(event="meta", data=machine_metadata())

            if hasattr(pack, 'override_run'):
                # TODO: This is only to hack the CI and have the test passing
                # using the old system. Remove override_run() and use `Executor`
                # instead 
                fut = pack.override_run()
            else:
                pack.phase = "run"
                fut = pack.execute(*argv, **{**_kwargs, **kwargs})

            coro.append(fut)

            if timeout:
                delay = pack.config.get("max_duration", timeout_delay)
                asyncio.create_task(force_terminate(pack, delay))

        return await asyncio.gather(*coro)


class SingleCmdExecutor(Executor):
    def argv(self, **kwargs) -> List:
        """Return the list of command line's arguments for this `Executor`
        followed by its embedded `Executor`'s list of command line's arguments

        Arguments:
            **kwargs: some `Executor` might need an argument to dynamically
                      generate the list of command line's arguments
        """
        if self.exec:
            return self._argv(**kwargs) + self.exec.argv(**kwargs)
        return self._argv(**kwargs)

    def kwargs(self) -> Dict:
        """Return the `Executor`'s kwargs to send to `BasePackage.execute()`
        merged with the embeded `Executor`, if any

        The `Executor`'s kwargs will take priority over the embeded `Executor`'s
        kwargs
        """
        kwargs = self._kwargs
        if self.exec:
            kwargs = {**self.exec.kwargs(), **kwargs}
        return kwargs

    def commands(self) -> Generator[Tuple[pack.BasePackage, List, Dict], None, None]:
        yield self.pack, self.argv(), self.kwargs()

    def _argv(self, **kwargs) -> List:
        del kwargs
        return []


class ListExecutor(Executor):
    """Execute a list of `Executor`s in parallel

    Arguments:
        executors: `Executor`s to be executed
        **kwargs: kwargs to be passed to the `pack.execute()`
    """
    def __init__(
            self,
            *executors: Tuple[Executor],
            **kwargs
    ) -> None:
        super().__init__(
            executors[0],
            **kwargs
        )
        self.executors = executors

    def commands(self) -> Generator[Tuple[pack.BasePackage, List, Dict], None, None]:
        for executor in self.executors:
            yield from executor.commands()


class CmdExecutor(SingleCmdExecutor):
    """Execute a command

    Arguments:
        pack: `BasePackage`'s instance from which `execute()` will be used to
              perform the command
        *cmd_argv: command line arguments list to execute
        **kwargs: kwargs to be passed to the `pack.execute()`
    """
    def __init__(
            self,
            pack: pack.BasePackage,
            *cmd_argv,
            **kwargs
    ) -> None:
        if isinstance(pack, Executor):
            raise TypeError(f"{self.__class__.__name__} does not accept nested"
                            f" {Executor.__name__}")
        super().__init__(
            pack,
            **kwargs
        )
        self.cmd_argv = cmd_argv

    def _argv(self, **_) -> List:
        return [*self.cmd_argv]


class PackExecutor(CmdExecutor):
    """Execute a `Package`'s script. If not specified, the `Package`'s
    main_script will be used

    Arguments:
        pack: `Package`'s instance from which `execute()` will be used to
              perform the command
        *script_argv: script's command line arguments list. If the first
                      argument is a file or directory that can be found, the
                      file will be used instead of the `pack`'s main_script
        **kwargs: kwargs to be passed to the `pack.execute()`
    """
    def __init__(
            self,
            pack: pack.Package,
            *script_argv,
            **kwargs
    ) -> None:
        script = script_argv[:1]
        if script and XPath(script[0]).exists():
            script = script[0]
            script_argv = script_argv[1:]
        else:
            script = None

        super().__init__(pack, *script_argv, **kwargs)

        self.script = script

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
        return script + super()._argv(**kwargs)


class VoidExecutor(CmdExecutor):
    """Execute nothing"""
    def __init__(
            self,
            pack: pack.BasePackage,
            *argv,
            **kwargs
    ) -> None:
        super().__init__(pack, "true", *argv, **kwargs)


class WrapperExecutor(SingleCmdExecutor):
    """Wrap an `Executor` with any command

    Arguments:
        executor: `Executor` to be executed
        *wrapper_argv: command line arguments list
        **kwargs: kwargs to be passed to the `pack.execute()`
    """
    def __init__(
            self,
            executor: SingleCmdExecutor,
            *wrapper_argv,
            **kwargs
    ) -> None:
        super().__init__(
            executor,
            **kwargs
        )
        self.wrapper_argv = wrapper_argv

    def _argv(self, **kwargs) -> List:
        del kwargs
        return [*self.wrapper_argv]


class DockerRunExecutor(WrapperExecutor):
    """Execute an `Executor` through Docker

    Arguments:
        executor: `Executor` to be executed through Docker
        image: the Docker image to use
        *docker_argv: Docker command line arguments list
        **kwargs: kwargs to be passed to the `pack.execute()`
    """
    def __init__(
            self,
            executor: SingleCmdExecutor,
            image: str,
            *docker_argv,
            **kwargs
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
            **kwargs
        )
        self.image = image

    def _argv(self, **kwargs) -> List:
        if self.image is None or os.environ.get("MILABENCH_DOCKER", None):
            # No-op when there's no docker image to run or inside a docker
            # container
            return []

        argv = super().argv(**kwargs)

        env = self.pack.make_env()
        for var in ("MILABENCH_CONFIG", "XDG_CACHE_HOME", "OMP_NUM_THREADS"):
            argv.append("--env")
            argv.append(f"{var}='{env[var]}'")

        argv.append(self.image)
        argv.append(f"{self.pack.dirs.code / 'activator'}")
        argv.append(f"{self.pack.dirs.venv}")

        return argv


class SSHExecutor(WrapperExecutor):
    """Execute an `Executor` through ssh

    Arguments:
        executor: `Executor` to be executed through ssh
        host: host's address
        *ssh_argv: ssh command line arguments list
        user: username to use to connect to the host. By default, `pack.config`
              will be used to find the username
        key: ssh key to use to connect to the host. By default, `pack.config`
             will be used to find the username
        **kwargs: kwargs to be passed to the `pack.execute()`
    """
    def __init__(
        self,
        executor: SingleCmdExecutor,
        host: str,
        *ssh_argv,
        user: str = None,
        key: str = None,
        **kwargs,
    ) -> None:
        super().__init__(
            executor,
            "ssh",
            "-oCheckHostIP=no",
            "-oStrictHostKeyChecking=no",
            *ssh_argv,
            **kwargs)
        self.host = host
        self.user = user
        self.key = key

    def _find_node_config(self) -> Dict:
        for n in self.pack.config["system"]["nodes"]:
            if n["ip"] == self.host:
                return n
        return {}

    def _argv(self, **kwargs) -> List:
        if socket.gethostname() == self.host:
            # No-op when executing on the main node
            return []

        node = self._find_node_config()
        user = self.user or node.get("user", None)
        key = self.key or node.get("key", None)
        host = f"{user}@{self.host}" if user else self.host

        argv = super().argv(**kwargs)

        if key:
            argv.append(f"-i{key}")
        argv.append(host)

        return argv


class TorchRunExecutor(WrapperExecutor):
    def __init__(self, executor: Executor, *torchrun_argv, **kwargs) -> None:
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


class VoirExecutor(WrapperExecutor):
    """Execute an `Executor` through voir

    Arguments:
        executor: `Executor` to be executed
        *voir_argv: voir command line arguments list
        **kwargs: kwargs to be passed to the `pack.execute()`
    """
    def __init__(
            self,
            executor: SingleCmdExecutor,
            *voir_argv,
            **kwargs
    ) -> None:
        super().__init__(
            executor,
            "voir",
            **{"setsid":True, **kwargs}
        )
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


class NJobs(ListExecutor):
    """Execute n instances of the same `Executor` in parallel

    Arguments:
        executor: `Executor` to be executed
        n: number of times `executor` should be executed 
        **kwargs: kwargs to be passed to the `pack.execute()`
    """
    def __init__(self, executor: Executor, n: int, gpus: list = None, **kwargs) -> None:
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


class SequenceExecutor(ListExecutor):
    """Execute a list of `Executor`s in sequence
    Arguments:
        executors: `Executor`s to be executed
        **kwargs: kwargs to be passed to the `pack.execute()`
    """
    def __init__(
            self,
            *executors: Tuple[Executor],
            **kwargs
    ) -> None:
        super().__init__(
            None,
            **kwargs
        )
        self.executors = executors

    async def execute(self, **kwargs):
        results = [] 
        for executor in self.executors:
            results.append(await executor.execute(**{**self._kwargs, **kwargs}))

        return results


class PerGPU(ListExecutor):
    """Execute one instance of an `Executor` on each gpu

    Arguments:
        executor: `Executor` to be executed
        **kwargs: kwargs to be passed to the `pack.execute()`
    """
    def __init__(self, executor: Executor, gpus: list = None, **kwargs) -> None:
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


# Accelerate
class AccelerateLaunchExecutor(SingleCmdExecutor):
    """Execute a `BasePackage` with Accelerate

    Arguments:
        pack: `BasePackage`'s instance from which `execute()` will be used to
              perform the command
        *accelerate_argv: Accelerate's command line arguments list
        **kwargs: kwargs to be passed to the `pack.execute()`
    """
    def __init__(
            self,
            pack: pack.BasePackage,
            *accelerate_argv,
            **kwargs
    ) -> None:
        super().__init__(
            pack,
            **kwargs
        )
        self.accelerate_argv = accelerate_argv

    def _argv(self, rank, **_) -> List:
        nproc = (
            len(self.pack.config.get("devices", [])) * self.pack.config["num_machines"]
        )
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
            "accelerate",
            "launch",
            "--mixed_precision=fp16",
            "--dynamo_backend=no",
            f"--machine_rank={rank}",
            f"--num_machines={self.pack.config['num_machines']}",
            *deepspeed_argv,
            f"--gradient_accumulation_steps={self.pack.config['gradient_accumulation_steps']}",
            f"--num_cpu_threads_per_process={self.pack.config['cpus_per_gpu']}",
            f"--main_process_ip={self.pack.config['manager_addr']}",
            f"--main_process_port={self.pack.config['manager_port']}",
            f"--num_processes={nproc}",
            *self.accelerate_argv,
            str(self.pack.dirs.code / "main.py"),
        ]


class AccelerateLoopExecutor(Executor):
    """Execute an `AccelerateLaunchExecutor`

    Arguments:
        executor: `AccelerateLaunchExecutor` to be executed
        ssh_exec: `SSHExecutor` to be used. It must embed, directly or
                   indirectly, a `AccelerateLoopExecutor.PLACEHOLDER` which will
                   be replaced by `accelerate_exec`
        **kwargs: kwargs to be passed to the `pack.execute()`
    """

    class _Placeholder(Executor):
        def __init__(self) -> None:
            pass

    PLACEHOLDER = _Placeholder()

    def __init__(
            self,
            executor: AccelerateLaunchExecutor,
            ssh_exec: SSHExecutor = None,
            **kwargs
    ) -> None:
        if not isinstance(ssh_exec, SSHExecutor):
            raise ValueError(f"{self.__class__.__name__} only accepts"
                             f" {SSHExecutor.__name__} as nested"
                             f" {Executor.__name__}")
        super().__init__(
            ssh_exec,
            **kwargs
        )
        self.accelerate_exec = executor
        _exec = self
        while _exec:
            if _exec.exec is AccelerateLoopExecutor.PLACEHOLDER:
                _exec.exec = self.accelerate_exec
            _exec = _exec.exec

    def commands(self) -> Generator[Tuple[pack.BasePackage, List, Dict], None, None]:
        yield (
            self.pack,
            self.accelerate_exec.argv(rank=0),
            {"setsid": True, "use_stdout": True, **self.kwargs()},
        )
        for i, worker in enumerate(self.pack.config.get("worker_addrs", [])):
            self.exec.host = worker
            run_pack = self.pack.copy({"tag": [*self.pack.config["tag"], worker]})
            yield run_pack, self.argv(rank=i + 1), self.kwargs()
