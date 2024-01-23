import asyncio
import os

from ..alt_async import destroy, run
from ..metadata import machine_metadata
from ..structs import BenchLogEntry


async def execute(pack, *args, cwd=None, env={}, external=False, **kwargs):
    """Run a command in the virtual environment.

    Unless specified otherwise, the command is run with
    ``self.dirs.code`` as the cwd.

    Arguments:
        args: The arguments to the command
        cwd: The cwd to use (defaults to ``self.dirs.code``)
    """
    args = [str(x) for x in args]
    if cwd is None:
        cwd = pack.dirs.code

    exec_env = pack.full_env(env) if not external else {**os.environ, **env}

    return await run(
        args,
        **kwargs,
        info={"pack": pack},
        env=exec_env,
        constructor=BenchLogEntry,
        cwd=cwd,
        process_accumulator=pack.processes,
    )


async def force_terminate(pack, delay):
    await asyncio.sleep(delay)
    for proc in pack.processes:
        ret = proc.poll()
        if ret is None:
            await pack.message(
                f"Terminating process because it ran for longer than {delay} seconds."
            )
            destroy(proc)


async def execute_command(
    command, phase="run", timeout=False, timeout_delay=600, **kwargs
):
    """Execute all the commands and return the aggregated results"""
    coro = []

    for pack in command.packs():
        pack.phase = phase

    timeout_tasks = []
    for pack, argv, _kwargs in command.commands():
        await pack.send(event="config", data=pack.config)
        await pack.send(event="meta", data=machine_metadata(pack))

        fut = execute(pack, *argv, **{**_kwargs, **kwargs})
        coro.append(fut)

        if timeout:
            delay = pack.config.get("max_duration", timeout_delay)
            timeout_task = asyncio.create_task(force_terminate(pack, delay))
            timeout_tasks.append(timeout_task)

    results = await asyncio.gather(*coro)

    if timeout:
        for task in timeout_tasks:
            task.cancel()
    return results
