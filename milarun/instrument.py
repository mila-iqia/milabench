import json
import time
from contextlib import contextmanager

from giving import give

from .runner import StopProgram
from .utils import REAL_STDOUT, give_std


class Plain:
    def __init__(self, x):
        self._object = x

    def __rich__(self):
        return self._object


def dash(runner, gv, *, arg=None):
    """Create a simple terminal dashboard using rich.
    This displays a live table of the last value for everything given,
    with a progress bar for the current task under it.
    """

    from rich.console import Console, Group
    from rich.live import Live
    from rich.pretty import Pretty
    from rich.progress import Progress
    from rich.table import Table

    # Current rows are stored here
    rows = {}

    # First, a table with the latest value of everything that was given
    table = Table.grid(padding=(0, 3, 0, 0))
    table.add_column("key", style="bold green")
    table.add_column("value")

    # Below, a progress bar for the current task (train or test)
    progress_bar = Progress(auto_refresh=False)
    current_task = progress_bar.add_task("----")

    # Group them
    grp = Group(table, progress_bar)

    console = Console(color_system="standard", file=REAL_STDOUT)

    # This will wrap Live around the run block (the whole main function)
    gv.wrap(
        "run",
        Live(
            grp,
            refresh_per_second=4,
            console=console,
            # redirect_stdout=False,
            # redirect_stderr=False,
        ),
    )

    # This sets the progress bar's completion meter
    @gv.where("progress").ksubscribe
    def _(progress, total, descr):
        progress_bar.reset(current_task, total=total, description=descr)
        progress_bar.update(current_task, completed=progress)

    @ (gv["?#stdout"].roll(10) | gv["?#stderr"].roll(10)).subscribe
    def _(txt):
        give(stdout="".join(txt))

    # This updates the table every time we get new values
    @gv.where("!silent").subscribe
    def _(values):
        for k, v in values.items():
            if k.startswith("$") or k.startswith("#"):
                continue
            if k in rows:
                rows[k]._object = v
            else:
                if isinstance(v, str):
                    rows[k] = Plain(v)
                else:
                    rows[k] = Pretty(v)
                table.add_row(k, rows[k])


def stop(runner, gv, *, arg):
    assert arg is not None
    n = arg
    gv["?metric"].skip(n).subscribe(runner.stop)
    gv["?metric"].count(scan=True).map(
        lambda x: dict(progress=x, total=n, descr="Progress", silent=True)
    ).give()


def to_accuracy(runner, gv, *, arg):
    gv["?metric"].average(scan=10).filter(lambda m: m < arg).subscribe(runner.stop)


def timings(runner, gv, *, arg=None):
    times = (
        gv["?metric"]
        .map(lambda _: time.time())
        .pairwise()
        .starmap(lambda x, y: 1000 * (y - x))
    )

    reports = {
        "min duration": times.min(scan=True),
        "avg duration": times.average(scan=10),
        "max duration": times.max(scan=True),
    }

    for name, stream in reports.items():
        stream.format("{:3.2f}ms").give(name)


def dump(runner, gv):
    gv.display()


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if not isinstance(obj, float) and hasattr(obj, "__float__"):
            return float(obj)
        elif isinstance(obj, Exception):
            return {
                "type": type(obj).__name__,
                "message": str(obj),
            }
        else:
            return super().default(obj)


@contextmanager
def forward(runner, gv, arg):
    keys = arg.split(",") if isinstance(arg, str) else arg

    @gv.keep("#stdout", "#stderr", "#error", "#end", *keys).subscribe
    def _(data):
        REAL_STDOUT.write(json.dumps(data, cls=_Encoder) + "\n")
        REAL_STDOUT.flush()

    with give_std():
        try:
            yield
        except StopProgram:
            give(**{"#aborted": True})
        except Exception as exc:
            give(**{"#error": exc})


def wandb(runner, gv, *, arg):
    import wandb

    entity, project = arg.split(":")

    @gv["?args"].subscribe
    def _(args):
        wandb.init(project=project, entity=entity, config=vars(args))
        gv["?model"].first() >> wandb.watch
        gv.keep("metric") >> wandb.log


def monitor(runner, gv):
    from giving import give

    from .monitor import GPUMonitor

    monitor = GPUMonitor(runner, 2)
    monitor.start()

    # @gv["?metric"].subscribe
    # def _(_):
    #     give(gpu_load=monitor.data[0]["load"][-1])
    #     give(gpu_memory=monitor.data[0]["memory"][-1])
    #     give(gpu_temperature=monitor.data[0]["temperature"][-1])


def toast(runner, gv, *, arg):
    gv["?metric"].subscribe(print)


def flops(runner, gv, *, arg):
    from deepspeed.profiling.flops_profiler import get_model_profile

    @gv["?model"].subscribe
    def _(model):
        get_model_profile(
            model=model,
            input_res=tuple(arg),
            input_constructor=None,
            print_profile=True,
            detailed=True,
            warm_up=10,
            as_string=False,
            output_file=None,
            ignore_modules=None,
        )
        raise StopProgram()


def persec(runner, gv, *, arg=None):
    sync = None

    @(gv["?use_cuda"].first(lambda x: x).subscribe)
    def _(use_cuda):
        nonlocal sync
        import torch
        sync = torch.cuda.synchronize

    times = (
        gv.where("metric", "batch")
        .augment(time=lambda: time.time())
        .pairwise()
        .buffer_with_time(1.0)
    )

    @times.subscribe
    def _(elems):
        t = 0
        if sync is not None:
            t0 = time.time()
            sync()
            t1 = time.time()
            t += t1 - t0

        t += sum(e2["time"] - e1["time"] for e1, e2 in elems)
        n = sum(e1["batch"].shape[0] for e1, e2 in elems)

        runner.queue(rate = n/t)


    # times = (
    #     gv["?metric"]
    #     .map(lambda _: time.time())
    #     .pairwise()
    #     .starmap(lambda x, y: 1000 * (y - x))
    # )

    # @(gv["?metric"]).subscribe
