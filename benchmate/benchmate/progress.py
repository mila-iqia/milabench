"""THIS FILE CANNOT IMPORT ANYTHING FROM benchmate"""

import os
import sys
import time
import threading
import atexit
import signal
import multiprocessing

import tqdm as _tqdm


MIN_INTERVAL = 30
MAX_INTERVAL = 60

tqdm_cls = _tqdm.tqdm
if hasattr(_tqdm, "_original_tqdm"):
    tqdm_cls = _tqdm._original_tqdm 


class PatchedTQDM(tqdm_cls):
    def __init__(self, *args, **kwargs):
        disable = kwargs.pop("disable", False) or int(os.getenv("MILABENCH_NO_PROGRESS", 0))

        mininterval = max(MIN_INTERVAL, kwargs.pop("mininterval", MIN_INTERVAL))
        maxinterval = max(MAX_INTERVAL, kwargs.pop("maxinterval", MAX_INTERVAL))

        super().__init__(*args, mininterval=mininterval, maxinterval=maxinterval, disable=disable, **kwargs)
    

tqdm = PatchedTQDM


def patch_tqdm():
    if multiprocessing.current_process().name != "MainProcess":
        return

    if not hasattr(_tqdm, "_original_tqdm"):
        print("Installing SlowTQDM", file=sys.stderr, flush=True)

        _tqdm._original_tqdm = _tqdm.tqdm
        _tqdm.tqdm = PatchedTQDM

        # also patch tqdm.auto
        try:
            import tqdm.auto as _tqdm_auto
            _tqdm_auto._original_tqdm = _tqdm_auto.tqdm
            _tqdm_auto.tqdm = PatchedTQDM

        except ImportError:
            pass


class TimedFlushBuffer:
    """We don't want unbuffered but we want some periodic updates"""
    def __init__(self, stream, flush_interval=30):
        self.stream = stream
        self.buffer = []
        self.size = 0
        self.flush_interval = flush_interval
        self.lock = threading.Lock()
        self.last_flush = time.time()

    def should_flush(self):
        elapsed = time.time() - self.last_flush
        return elapsed >= self.flush_interval or self.size > 8 * 1024 * 1024

    def write(self, s):
        with self.lock:
            self.buffer.append(s)
            self.size += len(s)
            
            if self.should_flush():
                self._flush()

    def flush(self):
        with self.lock:
            self._flush()

    def _flush(self):
        if self.buffer:
            self.stream.write("".join(self.buffer))
            self.stream.flush()
            self.buffer = []
            self.size = 0
            self.last_flush = time.time()


def force_flush():
    if multiprocessing.current_process().name != "MainProcess":
        return

    # We do not want to use unbuffered python 
    # but we still want all the logs when the job crashes
    handlers = {}

    def flush(name):
        def flush_streams():
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                print(f"{multiprocessing.current_process().name} Flushing {name}", file=sys.stderr)
            except ValueError:
                print("Couldn't do last flush")
        return flush_streams


    def handle_signal(signum, frame):
        flush(f"signal {signum}")()

        if (handler := handlers[signum]) not in (None, signal.SIG_IGN, signal.SIG_DFL):
            handler(signum, frame)

        signal.signal(signum, signal.SIG_DFL)

        try:
            os.kill(os.getpid(), signum)
        except Exception:
            os._exit(1)

    handle_signal._is_forceflush = True

    updated = False
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        current_handler = signal.getsignal(sig)
        if getattr(current_handler, "_is_forceflush", False):            # Already handled, skip
            continue

        handlers[sig] = current_handler
        signal.signal(sig, handle_signal)
        updated = True
    
    if updated:
        atexit.register(flush("atexit"))
        print("Installing Force flush", file=sys.stderr, flush=True)


def timed_flush():
    if multiprocessing.current_process().name != "MainProcess":
        return

    if not isinstance(sys.stdout, TimedFlushBuffer):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

        print("Installing TimedFlush", file=sys.stderr, flush=True)
        # Replace sys.stdout/stderr
        sys.stdout = TimedFlushBuffer(sys.stdout, flush_interval=30)
        sys.stderr = TimedFlushBuffer(sys.stderr, flush_interval=30)


if int(os.getenv("MILABENCH_GLOBAL_PATCH", 0)) == 1:
    patch_tqdm()
    force_flush()
