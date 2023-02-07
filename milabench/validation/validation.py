from contextlib import contextmanager


class ValidationLayer:
    """Validation layer interface, capture event, and makes a report"""
    def __init__(self, gv) -> None:
        gv.subscribe(self._on_event)

    def _on_event(self, data):
        data = dict(data)
        run = data.pop("#run", None)
        pack = data.pop("#pack", None)
        tg = ".".join(run["tag"]) if run else pack.config["name"]
        ks = set(data.keys())
        self.on_event(pack, run, tg, ks, **data)    
    
    def on_event(self, pack, run, tag, keys, **data):
        raise NotImplementedError()
    
    def report(self):
        raise NotImplementedError()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


class Summary:
    """Simple utility to generate report with subsections"""
    def __init__(self) -> None:
        self.body = []
        self.stack = []
        self.indent = '  '
        self.sections = [
            '=',
            '-',
            '^',
            '*',
            '`'
        ]
        
    def _line_char(self):
        depth = len(self.stack)
        char = self.sections[min(depth, len(self.sections))]
        return char
        
    @contextmanager
    def section(self, title):
        self.newsection(title)
        yield
        self.endsection()
        
    def newsection(self, title):
        self.stack[-1].append(title)
        self.underline()
        self.stack.push([])
        
    def endsection(self):
        self.body.append(self.stack.pop())
        
    def newline(self):
        self.stack[-1].append('')
        
    def underline(self, char=None):
        if char is None:
            char = self._line_char()
        
        last_line = self.stack[-1][-1]
        self.stack.append(char * len(last_line))
        
    def add(self, txt):
        self.stack[-1].append(txt)
        
    def show(self):
        output = []
        self._show(self.body, 0, output)
        
        print('\n'.join(output))
        
    def _show(self, body, depth, output):
    
        for line in body:
            if isinstance(line, list):
                self._show(line, depth + 1)
        
            output.append(self.indent * depth + line)
        
    