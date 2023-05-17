import asyncio
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from itertools import count
from typing import Dict, List

from voir.proc import LogEntry

from milabench.alt_async import send

_count = count()


class BenchLogEntry(LogEntry):
    def __init__(self, pack, **rest):
        super().__init__(**rest)
        self.pack = pack

    @property
    def tag(self):
        return self.pack.tag


@dataclass
class Job:
    argv: List[str]
    info: dict = None
    env: Dict[str, object] = None
    cwd: str = None
    preexec_fn: object = None
    properties: dict = None


class TaskLogger:
    def __init__(self, **common_args):
        self._common = common_args

    def __getattr__(self, attr):
        @asynccontextmanager
        async def make(message, **args):
            token = next(_count)
            await send(
                BenchLogEntry(
                    event="start-task",
                    data={"task": attr, "token": token, "message": message, **args},
                    **self._common,
                )
            )
            yield
            await send(
                BenchLogEntry(
                    event="end-task",
                    data={"task": attr, "token": token},
                    **self._common,
                )
            )

        return make
