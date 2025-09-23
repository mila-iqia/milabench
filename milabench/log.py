import json
import os
import pprint
import shlex
import time
from collections import defaultdict
from datetime import datetime

from blessed import Terminal
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

from .config import get_run_count
from .fs import XPath

T = Terminal()
color_wheel = [T.cyan, T.magenta, T.yellow, T.red, T.green, T.blue]


class BaseLogger:
    def start(self):
        pass

    def end(self):
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self.end()


class TagConsole(BaseLogger):
    def __init__(self, tag, i):
        self.header = color_wheel[i % len(color_wheel)](T.bold(tag))

    def _ensure_line(self, x):
        if not x.endswith("\n"):
            x += "\n"
        return x

    def sprint(self, *parts):
        parts = [self.header, *parts]
        return self._ensure_line(" ".join(map(str, parts)))

    def spretty(self, *parts):
        *parts, obj = parts
        parts = [
            self.header,
            *parts,
            obj if isinstance(obj, str) else pprint.pformat(obj, width=120),
        ]
        return self._ensure_line(" ".join(map(str, parts)))

    def print(self, *parts):
        print(self.sprint(*parts), end="")

    def pretty(self, *parts):
        print(self.spretty(*parts), end="")

    def close(self):
        pass


class TerminalFormatter(BaseLogger):
    def __init__(self, dump_config=True):
        self.consoles = {}
        self.error_happened = set()
        self.early_stop = False
        self.dump_config = dump_config

    def console(self, tag):
        if tag not in self.consoles:
            self.consoles[tag] = TagConsole(tag, len(self.consoles))

        return self.consoles[tag]

    def __call__(self, entry):
        event = entry.event
        data = entry.data
        pipe = entry.pipe
        tag = entry.tag

        console = self.console(tag)

        if event == "line":
            data = (data or "").rstrip()
            if pipe == "stderr":
                console.print(T.bold_yellow("[stderr]"), T.yellow(data))
            else:
                console.print(T.bold("[stdout]"), data)

        elif event == "data":
            data = dict(data)
            if "progress" in data:
                return
            console.pretty(T.bold_magenta("[data]"), data)

        elif event == "start":
            console.print(
                T.bold_green("[start]"),
                T.bold_green(shlex.join(data.get("command", []))),
                T.gray(f'[at {datetime.fromtimestamp(data["time"])}]'),
            )

        elif event == "stop":
            self.early_stop = True

        elif event == "error":
            self.error_happened.add(tag)
            if data["message"]:
                console.print(
                    T.bold_red(data["type"] + ":"),
                    T.red(data["message"]),
                )
            else:
                console.print(T.bold_red(data["type"]))

        elif event == "end":
            rc = data.get(
                "return_code",
            )
            wrong = not self.early_stop and (
                (tag in self.error_happened) or data["return_code"] != 0
            )
            if wrong:
                rc = data["return_code"] or "ERROR"
                console.print(
                    T.bold_red(f"[{event} ({rc})]"),
                    T.bold_red(shlex.join(data.get("command", []))),
                    T.gray(f'[at {datetime.fromtimestamp(data["time"])}]'),
                )
            else:
                console.print(
                    T.bold_green(f"[{event}]"),
                    T.bold_green(shlex.join(data.get("command", []))),
                    T.gray(f'[at {datetime.fromtimestamp(data["time"])}]'),
                )

        elif event == "phase":
            pass

        elif event == "config":
            def _show(k, entry):
                if k.startswith("config.system"):
                    return

                if isinstance(entry, dict):
                    for k2, v in entry.items():
                        _show(f"{k}.{k2}", v)
                else:
                    console.pretty(T.bold(f"[{k}]"), entry)

            if self.dump_config:
                _show("config", data)

        elif event == "meta":
            pass
        
        elif event == "message":
            console.pretty(T.bold(f"[{event}]"), data["message"])

        else:
            console.pretty(T.bold(f"[{event}]"), data)


class BaseReporter(BaseLogger):
    def __init__(self, pipe):
        self.pipe = pipe
        self.files = {}

    def file(self, entry):
        if entry.tag not in self.files:
            file = entry.pack.logfile(self.pipe)
            os.makedirs(XPath(file).parent, exist_ok=True)
            self.files[entry.tag] = open(file, "a").__enter__()
        return self.files[entry.tag]

    def log(self, entry):
        pass

    def cleanup(self, entry):
        if entry.event == "end":
            if entry.tag in self.files:
                self.files[entry.tag].__exit__(None, None, None)
                del self.files[entry.tag]

    def __call__(self, entry):
        self.log(entry)
        self.cleanup(entry)


class TextReporter(BaseReporter):
    def log(self, entry):
        if entry.event == "line" and entry.pipe == self.pipe:
            self.file(entry).write(entry.data)

    def close(self):
        assert not self.files
        for open_file in self.files.values():
            open_file.__exit__(None, None, None)


class DataReporter(BaseReporter):
    def __init__(self):
        super().__init__(pipe="data")

    def log(self, entry):
        d = entry.dict()
        d.pop("pack")
        try:
            j = json.dumps(d)
        except TypeError:
            j = {"#unrepresentable": str(d)}
        self.file(entry).write(f"{j}\n")


def new_progress_bar():
    progress = Progress(
        BarColumn(),
        TextColumn("({task.completed}/{task.total})"),
    )
    progress._task = progress.add_task("progress")
    return progress


class DashFormatter(BaseLogger):
    def __init__(self):
        self.panel = Panel("")
        self.console = Console()
        self.live = Live(self.panel, refresh_per_second=4, console=self.console)
        self.rows = defaultdict(dict)
        self.benchcount = defaultdict(int)
        self.endtimes = {}
        self.early_stop = {}
        # Limit the number of rows to avoid too much clutering
        # This is a soft limit, it only prunes finished runs
        self.max_rows = 8
        self.prune_delay = 60
        self.current = 0
        self.created_time = time.time()

    def _get_global_progress_bar(self):
        progress = self.rows.get("GLOBAL")
        if progress is not None:
            return progress["progress"]
        progress = new_progress_bar()
        progress.update(progress._task, completed=self.current, total=get_run_count())
        self.rows["GLOBAL"] = {"progress": progress}
        return progress

    def _update_global(self, inc):
        self.current += inc
        if total := get_run_count():
            progress = self._get_global_progress_bar()
            progress.update(progress._task, completed=self.current, total=total)

    def should_prune(self, tag, elasped):
        # Old run, remove it
        if elasped > self.prune_delay:
            return True
        
        # Bench is running
        bench = tag.split(".")[0]
        if self.benchcount.get(bench, 0) > 0:
            return False

        # We have too many rows
        if self.max_rows:
            return len(self.rows) > self.max_rows
        
        return False
    
    def prune(self):
        now = time.time()
        for tag, endtime in list(self.endtimes.items()):
            if self.should_prune(tag, now - endtime):
                del self.endtimes[tag]
                del self.rows[tag]

    def refresh(self):
        self.prune()
        self.live.update(self.make_table())

    def start(self):
        self._update_global(0)
        self.live.__enter__()

    def end(self):
        self.live.__exit__(None, None, None)

    def __call__(self, entry):
        if get_run_count():
            self._get_global_progress_bar()
    
        event = entry.event
        data = entry.data
        tag = entry.tag
        row = self.rows[tag]

        method = getattr(self, f"on_{event}", None)
        if method:
            method(entry, data, row)

    def on_start(self, entry, data, row):
        self.benchcount[entry.tag.split('.')[0]] += 1

    def on_stop(self, entry, data, row):
        self.early_stop[entry.tag] = True

    def on_end(self, entry, data, row):
        self._update_global(1)
        self.endtimes[entry.tag] = time.time()
        self.benchcount[entry.tag.split('.')[0]] -= 1


class ShortDashFormatter(DashFormatter):
    def make_table(self):
        table = Table(padding=(0, 3, 0, 0))
        table.add_column("bench", style="bold white")
        table.add_column("status")
        table.add_column("progress", style="bold white")
        table.add_column("rate", style="bold green")
        table.add_column("loss", style="bold cyan")
        table.add_column("gpu_load", style="bold magenta")
        table.add_column("gpu_mem", style="bold magenta")
        table.add_column("gpu_temp", style="bold magenta")

        for bench, values in self.rows.items():
            if bench == "GLOBAL":
                table.add_row(
                    bench,
                    values.get("progress", "??%"),
                )
            else:
                table.add_row(
                    bench,
                    values.get("status", "?"),
                    values.get("progress", "??%"),
                    values.get("rate", "?"),
                    values.get("loss", "?"),
                    values.get("gpu_load", "?"),
                    values.get("gpu_mem", "?"),
                    values.get("gpu_temp", "?"),
                )

        return table

    def on_data(self, entry, data, row):
        data = dict(data)
        task = data.get("task", None)
        if prog := data.get("progress", None):
            if task == "early_stop":
                current, total = prog
                if total > 0:
                    perc = int(100 * (current / total))
                    if perc >= 100:
                        perc = "DONE"
                    else:
                        perc = f"{perc}%"
                    row["progress"] = perc
        elif gpudata := data.get("gpudata", None):
            for gpuid, data in gpudata.items():
                load = int(data.get("load", 0) * 100)
                currm, totalm = data.get("memory", [0, 0])
                temp = int(data.get("temperature", 0))
                row[f"gpu:{gpuid}"] = (
                    f"{load}% load | {currm:.0f}/{totalm:.0f} MB | {temp}C"
                )
                row["gpu_load"] = f"{load}%"
                row["gpu_mem"] = f"{currm:.0f}/{totalm:.0f} MB"
                row["gpu_temp"] = f"{temp}C"
                break
        elif (rate := data.get("rate", None)) is not None:
            if task == "train":
                row["rate"] = f"{rate:.2f}"
        elif (loss := data.get("loss", None)) is not None:
            if task == "train":
                row["loss"] = f"{loss:.2f}"
        self.refresh()

    def on_start(self, entry, data, row):
        super().on_start(entry, data, row)

        row["status"] = Text("RUNNING", style="bold yellow")
        self.refresh()

    def on_error(self, entry, data, row):
        row["status"] = Text("ERROR", style="bold red")
        self.refresh()

    def on_end(self, entry, data, row):
        super().on_end(entry, data, row)
        rc = data["return_code"]
        if rc == 0 or self.early_stop.get(entry.tag, False):
            row["status"] = Text("COMPLETED", style="bold green")
        else:
            row["status"] = Text(f"FAIL:{rc}", style="bold red")
        self.refresh()



octet_units = [
    (" o", 1024 ** 0),
    ("Ko", 1024 ** 1),
    ("Mo", 1024 ** 2),
    ("Go", 1024 ** 3)
]

def find_byte_exponent(value):
    for i in reversed(range(len(octet_units))):
        if value > octet_units[i][1]:
            return octet_units[i]
    return octet_units[0]

def formatbyte(value):
    name, exp = find_byte_exponent(value)
    return f"{value // exp:4d} {name}"
    

class LongDashFormatter(DashFormatter):
    def make_table(self):
        table = Table.grid(padding=(0, 3, 0, 0))
        table.add_column("bench", style="bold yellow")
        table.add_column("key", style="bold green")
        table.add_column("value")

        for bench, values in self.rows.items():
            values = dict(values)
            progress = values.pop("progress", None)
            if progress is not None:
                table.add_row(bench, "", progress)
                bench = ""
            for key, value in values.items():
                table.add_row(bench, key, value)
                bench = ""  # Avoid displaying the bench for the other rows

        return Panel(table)

    def on_data(self, entry, data, row):
        data = dict(data)
        if prog := data.get("progress", None):
            task = data.get("task", None)
            if task == "early_stop":
                current, total = prog
                if "progress" not in row:
                    progress_bar = Progress(
                        BarColumn(),
                        TimeRemainingColumn(),
                        TextColumn("({task.completed}/{task.total})"),
                    )
                    progress_bar._task = progress_bar.add_task("progress")
                    row["progress"] = progress_bar
                else:
                    progress_bar = row["progress"]
                    progress_bar.update(
                        progress_bar._task, completed=current, total=total
                    )
        elif gpudata := data.get("gpudata", None):
            for gpuid, data in gpudata.items():
                load = int(data.get("load", 0) * 100)
                currm, totalm = data.get("memory", [0, 0])
                temp = int(data.get("temperature", 0))
                row[f"gpu:{gpuid}"] = (
                    f"{load:3d}% load | {currm:.0f}/{totalm:.0f} MB | {temp}C"
                )
        elif iodata := data.get("iodata", None):
            read = int(iodata.get("read_time", 0))
            write = int(iodata.get("write_time", 0))
            readc = int(iodata.get("read_count", 0))
            writec = int(iodata.get("write_count", 0))
            busy = int(iodata.get("busy_time", 0))
            row["iodata"] = (
                f"(rt={read} wt={write} bt={busy}) ms | (rc={readc} wc={writec}) bytes"
            )
            
        elif process := data.get("process", None):
            pid = process.get("pid", "0")
            currm, totalm = process.get("memory", [0, 0])
            load = process.get("load", 0)
            read_bytes = int(process.get("read_bytes", 0))
            write_bytes = int(process.get("write_bytes", 0))

            read_chars = int(process.get("read_chars", 0))
            write_chars = int(process.get("write_chars", 0))

            row[f"cpu.{pid}"] = f"{load:3.0f}% load | {currm/1e9:.0f}/{totalm/1e9:.0f} GB"
            row[f"physical.io.{pid}"] = f"{formatbyte(read_bytes)} reads {formatbyte(write_bytes)} writes"
            row[f"virtual.io.{pid}"] = f"{formatbyte(read_chars)} reads {formatbyte(write_chars)} writes"

        elif cpudata := data.get("cpudata", None):
            currm, totalm = cpudata.get("memory", [0, 0])
            load = cpudata.get("load", 0)
            row["cpudata"] = (
                f"{load:3.0f}% load | {currm/1e9:.0f}/{totalm/1e9:.0f} GB"
            )
        elif netdata := data.get("netdata", None):
            bytes_sent = netdata.get("bytes_sent", 0)
            bytes_recv = netdata.get("bytes_recv", 0)
            row["netdata"] = (
                f"s={bytes_sent} | r={bytes_recv}"
            )
        else:
            task = data.pop("task", "")
            units = data.pop("units", "")
            time = data.pop("time", "")
            if time:
                time = f"time {self.time(time)} s"
            row.update(self.newlines(task, units, time, data))
        
        self.refresh()

    def newlines(self, task, units, time, data):
        lines = {}
        for k, v in data.items():
            key = f"{task} {k}".strip()
            value = f"{self.format(k, v)} {self.unit(k, units)}"

            if time:
                value = f"{value:<{max(80 - len(value), 1)}} {time}"
                time = ""
            lines[key] = value
        return lines

    def time(self, t):
        return int(t - self.created_time)

    def unit(self, k, unit):
        return unit

    def format(self, key, value):
        try:
            return f"{value:0.2f}"
        except:
            return str(value)