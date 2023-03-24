import json
import os
import pprint
import shlex
from datetime import datetime

from blessed import Terminal

from .fs import XPath

T = Terminal()
color_wheel = [T.cyan, T.magenta, T.yellow, T.red, T.green, T.blue]


class TagConsole:
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
            obj if isinstance(obj, str) else pprint.pformat(obj),
        ]
        return self._ensure_line(" ".join(map(str, parts)))

    def print(self, *parts):
        print(self.sprint(*parts), end="")

    def pretty(self, *parts):
        print(self.spretty(*parts), end="")

    def close(self):
        pass


class TerminalFormatter:
    def __init__(self):
        self.consoles = {}
        self.error_happened = set()

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
            wrong = (tag in self.error_happened) or data["return_code"] != 0
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
                if isinstance(entry, dict):
                    for k2, v in entry.items():
                        _show(f"{k}.{k2}", v)
                else:
                    console.pretty(T.bold(f"[{k}]"), entry)

            _show("config", data)

        elif event == "message":
            console.pretty(T.bold(f"[{event}]"), data["message"])

        else:
            console.pretty(T.bold(f"[{event}]"), data)


class BaseReporter:
    def __init__(self, pipe):
        self.pipe = pipe
        self.files = {}

    def file(self, entry):
        if entry.tag not in self.files:
            file = entry.pack.logfile(self.pipe)
            os.makedirs(XPath(file).parent, exist_ok=True)
            self.files[entry.tag] = open(file, "w").__enter__()
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
        j = json.dumps(d)
        self.file(entry).write(f"{j}\n")
