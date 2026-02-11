from contextlib import contextmanager
from dataclasses import dataclass, field

from ..structs import BenchLogEntry


class ValidationLayer:
    """Validation layer interface, captures events, makes report"""

    ignore_tag = "nolog"

    def __init__(self, **kwargs) -> None:
        # early stop means voir requested milabench to stop the benchmark
        # this means we can ignore the process return code because it got SIGTERM'ed
        self.early_stop = False

        # return code of the validation layer
        # this is used to make milabench fail on critical errors
        self._return_code = None

    def __call__(self, entry):
        return self.on_event(entry)

    def on_event(self, entry: BenchLogEntry):
        if self.ignore_tag in entry.tag:
            return

        method = getattr(self, f"on_{entry.event}", None)

        if method is not None:
            method(entry)

    def on_stop(self, entry):
        self.early_stop = True

    def on_config(self, entry):
        pass

    def on_start(self, entry):
        pass

    def on_error(self, entry):
        pass

    def on_line(self, entry):
        pass

    def on_data(self, entry):
        pass

    def on_end(self, entry):
        pass

    def on_phase(self, entry):
        pass

    @property
    def error_code(self):
        return self._return_code

    def set_error_code(self, code):
        self._return_code = code

    def end(self):
        return self._return_code

    def report(self, summary, **kwargs):
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


class Summary:
    """Simple utility to generate report with subsections"""

    @dataclass
    class _Section:
        name: str
        body: list = field(default_factory=list)

        @property
        def is_empty(self):
            return len(self.body) == 0

    def __init__(self) -> None:
        self.root = Summary._Section("")
        self.stack = [self.root]
        self.sections_lookup = dict()
        self.indent = "  "
        self.has_content = False
        self.sections = ["=", "-", "^", "*", "`"]

    def _line_char(self, depth):
        return self.sections[min(depth, len(self.sections))]

    def is_empty(self):
        return not self.has_content

    @contextmanager
    def section(self, title):
        self.newsection(title)
        yield
        self.endsection()

    def newsection(self, title):
        s = self.sections_lookup.get(title)
        if s is None:
            s = Summary._Section(title)
            self.stack[-1].body.append(s)

        self.stack.append(s)
        self.sections_lookup[title] = s

    def endsection(self):
        self.stack.pop()

    def newline(self):
        self.stack[-1].body.append("")

    def underline(self, size, char=None, depth=None):
        if char is None:
            char = self._line_char(depth)
        return char * size

    def add(self, txt):
        self.has_content = True
        self.stack[-1].body.append(txt)

    def show(self, printfun=print):
        if self.has_content:
            output = []
            self._show(self.root.body, 0, output)
            output.append("")
            printfun("\n".join(output))

    def _show(self, body, depth, output):
        def newline(text):
            output.append(self.indent * depth + text)

        for line in body:
            if isinstance(line, Summary._Section):
                if line.is_empty:
                    continue

                newline(line.name)
                newline(self.underline(len(line.name), depth=depth))
                self._show(line.body, depth + 1, output)
                continue

            newline(line)
        
        newline("")