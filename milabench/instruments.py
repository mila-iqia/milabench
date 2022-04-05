import time
from threading import Thread

from hrepr import trepr
from voir.tools import gated, parametrized

from .utils import REAL_STDOUT


@gated("--display", "Display given")
def display(ov):
    ov.given.display()


class Plain:
    def __init__(self, x, fmt="{}"):
        self._object = x
        self.fmt = fmt

    def __rich__(self):
        return self.fmt.format(str(trepr(self._object, max_depth=2, sequence_max=10)))


@gated("--dash", "Display dash")
def dash(ov):
    """Create a simple terminal dashboard using rich.

    This displays a live table of the last value for everything given.
    """
    yield ov.phases.init

    from rich.console import Console, Group
    from rich.live import Live
    from rich.pretty import Pretty
    from rich.progress import ProgressBar
    from rich.table import Table

    gv = ov.given

    # Current rows are stored here
    rows = {}

    # First, a table with the latest value of everything that was given
    table = Table.grid(padding=(0, 3, 0, 0))
    table.add_column("key", style="bold green")
    table.add_column("value")

    console = Console(color_system="standard", file=REAL_STDOUT)

    out_err = gv["?#stdout"].roll(10) | gv["?#stderr"].roll(10)

    @out_err.subscribe
    def _(txt):
        ov.give(stdout="".join(txt))

    # This updates the table every time we get new values
    def update_with(values):
        if {"total", "progress", "descr"}.issubset(values.keys()):
            k = values["descr"]
            k = f"\\[{k}]"
            if k not in rows:
                progress_bar = ProgressBar(finished_style="blue", width=50)
                table.add_row(k, progress_bar)
                rows[k] = progress_bar
            progress_bar = rows[k]
            progress_bar.update(total=values["total"], completed=values["progress"])
            return

        units = values.get("units", None)

        for k, v in values.items():
            if k.startswith("$") or k.startswith("#") or k == "units":
                continue
            if k in rows:
                rows[k]._object = v
            else:
                if units:
                    rows[k] = Plain(v, f"{{}} {units}")
                else:
                    rows[k] = Plain(v)
                table.add_row(k, rows[k])

    gv.where("!silent").subscribe(update_with)
    ov.send = update_with

    with Live(table, refresh_per_second=4, console=console):
        yield ov.phases.finalize(-1000)


@parametrized("--stop", type=int, default=0, help="Number of train rates to sample")
def stop(ov):
    called = False

    def _stop(value):
        # The stop signal on train_rate has the unfortunate effect of creating
        # another train_rate, so this gets called twice.
        nonlocal called
        if not called:
            called = True
            ov.stop(value)

    yield ov.phases.load_script(priority=-100)
    stop = ov.options.stop
    if stop:
        steps = ov.given.where("train_rate")
        steps.map_indexed(
            lambda _, idx: {"progress": idx, "total": stop, "descr": "train"}
        ).give()
        steps.skip(stop) >> _stop


@gated("--train-rate")
def train_rate(ov):
    yield ov.phases.load_script

    sync = None

    def setsync(use_cuda):
        if use_cuda:
            nonlocal sync
            import torch

            sync = torch.cuda.synchronize

    ov.given["?use_cuda"].first_or_default(False) >> setsync

    times = (
        ov.given.where("step", "batch")
        .kmap(batch_size=lambda batch: len(batch))
        .augment(time=lambda: time.time_ns())
        .keep("time", "batch_size")
        .pairwise()
        .buffer_with_time(1.0)
    )

    @times.subscribe
    def _(elems):
        t = 0
        if sync is not None:
            t0 = time.time_ns()
            sync()
            t1 = time.time_ns()
            t += t1 - t0

        t += sum(e2["time"] - e1["time"] for e1, e2 in elems)
        n = sum(e1["batch_size"] for e1, e2 in elems)
        t /= 1_000_000_000

        if t:
            ov.give(train_rate=n / t, units="items/s")


@gated("--loading-rate")
def loading_rate(ov):
    yield ov.phases.load_script

    def _timing():
        t0 = time.time_ns()
        results = yield
        t1 = time.time_ns()
        if "batch" in results:
            seconds = (t1 - t0) / 1000000000
            data = results["batch"]
            if isinstance(data, list):
                data = data[0]
            return len(data) / seconds
        else:
            return None

    @ov.given.where("loader").ksubscribe
    def _(loader):
        typ = type(iter(loader))
        (
            ov.probe("typ.next(!$x:@enter, #value as batch, !!$y:@exit)")
            .wmap(_timing)
            .average(scan=5)
            .throttle(1)
            .map(lambda x: {"loading_rate": x, "units": "items/s"})
            .give()
        )


@gated("--compute-rate")
def compute_rate(ov):
    yield ov.phases.load_script

    def _timing():
        t0 = time.time_ns()
        results = yield
        t1 = time.time_ns()
        if "batch" in results:
            seconds = (t1 - t0) / 1000000000
            data = results["batch"]
            if isinstance(data, list):
                data = data[0]
            return len(data) / seconds
        else:
            return None

    (
        ov.given.wmap("compute_start", _timing)
        .average(scan=5)
        .throttle(1)
        .map(lambda x: {"compute_rate": x, "units": "items/s"})
        .give()
    )


class GPUMonitor(Thread):
    def __init__(self, ov, delay):
        super().__init__(daemon=True)
        self.ov = ov
        self.stopped = False
        self.delay = delay

    def run(self):
        import GPUtil

        while not self.stopped:
            self.ov.give(gpudata=GPUtil.getGPUs())
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


@gated("--gpu", "Profile GPU usage.")
def profile_gpu(ov):
    yield ov.phases.load_script
    monitor = GPUMonitor(ov, 100)
    monitor.start()
    yield ov.phases.run_script
    monitor.stop()


@gated("--verify", "Verify the benchmark")
def verify(ov):
    yield ov.phases.parse_args
    ov.require(dash.instrument)
    ov.require(train_rate.instrument)
    ov.require(loading_rate.instrument)
    ov.require(compute_rate.instrument)

    losses = ov.given["?loss"]

    # Verify that the loss decreases
    first_loss = losses.first()
    first_loss.give("initial_loss")
    last_loss = losses.last()
    (first_loss | last_loss).pairwise().starmap(lambda fl, ll: ll < fl).as_(
        "verify.loss_decreases"
    ) >> ov.send

    # Verify that the loss is below threshold
    last_loss.map(lambda x: x < 1).as_("verify.loss_below_threshold") >> ov.send

    # Verify the presence of certain fields
    for field in ["train_rate", "loading_rate", "compute_rate"]:
        (
            ov.given.getitem(field, strict=False)
            .is_empty()
            .map(lambda x: not x)
            .as_(f"verify.has_{field}")
            >> ov.send
        )

    yield ov.phases.finalize
