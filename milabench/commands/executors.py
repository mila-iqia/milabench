import asyncio
import os
import traceback

from benchmate.warden import process_cleaner

from ..alt_async import destroy, run
from ..metadata import machine_metadata
from ..structs import BenchLogEntry
from ..syslog import syslog


async def execute(pack, *args, cwd=None, env={}, external=False, use_stdout=False, **kwargs):
    """Run a command in the virtual environment.

    Unless specified otherwise, the command is run with
    ``self.dirs.code`` as the cwd.

    Arguments:
        args: The arguments to the command
        cwd: The cwd to use (defaults to ``self.dirs.code``)
    """
    from ..sizer import resolve_argv, scale_argv

    if cwd is None:
        cwd = pack.working_directory

    exec_env = pack.full_env(env) if not external else {**os.environ, **env}

    # Final argument transformation,
    # everything is resolved right now
    # sized_args = scale_argv(pack, args)
    final_args = resolve_argv(pack, args)

    if use_stdout:
        exec_env["MILABENCH_USE_STDOUT"] = "1"

    return await run(
        final_args,
        **kwargs,
        use_stdout=use_stdout,
        info={"pack": pack},
        env=exec_env,
        constructor=BenchLogEntry,
        cwd=cwd,
        process_accumulator=pack.processes,
    )


async def force_terminate_now(pack, delay):
    for proc in pack.processes:
        ret = proc.poll()

        if ret is None:
            await pack.message(
                f"Terminating process because it ran for longer than {delay} seconds."
            )
            destroy(proc)


async def force_terminate(pack, delay):
    await asyncio.sleep(delay)
    force_terminate_now(pack, delay)


async def trigger_exceptions(futures, packs):
    for task in futures:
        pack = packs[task]
        if exc := task.exception():
            # Send the traceback
            for line in traceback.format_exception(exc):
                for l in line.split("\n"):
                    await pack.send(event="line", data=l, pipe="stderr")

            # also send the error
            await pack.send(event="error", data={
                "type": type(exc).__name__, 
                "message": str(exc) + ": did you run milabench install/prepare?",
            })


async def execute_command(
    command, phase="run", timeout=False, timeout_delay=600, with_gpu_warden=True, **kwargs
):
    """Execute all the commands and return the aggregated results"""
    packs = {}
    coro = []
    for pack in command.packs():
        pack.phase = phase

    max_delay = timeout_delay

    with process_cleaner(with_gpu_warden=with_gpu_warden) as warden:
        for pack, argv, _kwargs in command.commands():
            await pack.send(event="config", data=pack.config)
            await pack.send(event="meta", data=machine_metadata(pack))
            
            delay = None
            if timeout:
                delay = pack.config.get("max_duration", timeout_delay)
                max_delay = max(max_delay, delay)

            fut = asyncio.create_task(execute(pack, *argv, **{**_kwargs, **kwargs}))
            packs[fut] = pack
            coro.append(fut)
            warden.extend(pack.processes)

        if timeout:
            all_done = []

            for wait_count in range(3):
                # wait_count == 0: wait for the initial timeout
                # wait_count == 1: wait for the grace period for the process to end
                # wait_count == 2: unacceptable
                #                   1. Cancel the coroutine as the process is not responding
                #                   2. Send an error to the pack
                done, coro = await asyncio.wait(coro, timeout=max_delay)
                all_done.extend(done)

                if len(coro) == 0:
                    # all done exit
                    break

                if wait_count == 2:
                    syslog("{1} process(es) are still alive after grace period", len(coro))
                    syslog("Canceling the coroutines")
                    for timedout in coro:
                        timedout.cancel()
                        pack = packs[timedout]
                        pack.send(event="error", data={
                            "type": "TimeoutError", 
                            "message": "Survived after term & kill signal"
                        }) 
                    all_done.extend(coro)
                    break
            
                # Tasks timeout
                for timedout in coro:
                    # kill the underlying process which should force the coro to 
                    # return on next wait
                    pack = packs[timedout]
                    await force_terminate_now(pack, max_delay)

                # Grace period
                max_delay = 10

            await trigger_exceptions(all_done, packs)

            return all_done
        else:
            return await asyncio.gather(*coro)
