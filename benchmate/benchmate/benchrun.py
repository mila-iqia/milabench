#!/usr/bin/env python3

import os
import subprocess
from packaging import version

import torch
import torch.distributed.run as distrun
import torch.distributed.elastic.multiprocessing.api as elastic
import torch.distributed.elastic.multiprocessing.subprocess_handler as sub

from contextlib import contextmanager


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

def get_subprocess_handler(*args, **kwargs):
    return NewSubprocessHandler(*args, **kwargs)


@contextmanager
def forward_voir_file():
    """Overrides torchruns way of creating a new process so we can forward our file desctriptor"""
    old_handle = elastic.get_subprocess_handler
    old_handler = elastic.SubprocessHandler

    elastic.get_subprocess_handler = get_subprocess_handler
    elastic.SubprocessHandler = NewSubprocessHandler

    yield

    elastic.get_subprocess_handler = old_handle
    elastic.SubprocessHandler = old_handler


def run(args):
    with forward_voir_file():
        distrun.run(args)


def main(args=None):
    args = distrun.parse_args(args)
    run(args)


if __name__ == "__main__":
    main()
