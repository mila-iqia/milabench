#!/usr/bin/env python3

import os
import subprocess

from torch.distributed.run import main as torchrun
import torch.distributed.elastic.multiprocessing.api as elastic
import torch.distributed.elastic.multiprocessing.subprocess_handler as sub


class NewSubprocessHandler(sub.SubprocessHandler):
    def _popen(self, args, env) -> subprocess.Popen:
        kwargs = {}

        if fd := os.getenv("DATA_FD"):
            kwargs["pass_fds"] = [int(fd)]

        return subprocess.Popen(
            args=args,
            env=env,
            stdout=self._stdout,
            stderr=self._stderr,
            **kwargs,
        )

def get_subprocess_handler(
    entrypoint: str,
    args,
    env,
    stdout: str,
    stderr: str,
    local_rank_id: int,
):
    return NewSubprocessHandler(
        entrypoint=entrypoint,
        args=args,
        env=env,
        stdout=stdout,
        stderr=stderr,
        local_rank_id=local_rank_id,
    )


def main(args=None):
    elastic.get_subprocess_handler = get_subprocess_handler
    elastic.SubprocessHandler = NewSubprocessHandler

    torchrun(args)



if __name__ == "__main__":
    main()
