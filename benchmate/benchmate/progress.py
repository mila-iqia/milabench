"""THIS FILE CANNOT IMPORT ANYTHING FROM benchmate"""

import os
import sys
import time
import threading
import atexit
import signal

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
        self.flush_interval = flush_interval
        self.lock = threading.Lock()
        self.last_flush = time.time()
        # Start background flusher
        self._stop_thread = False
        self.thread = threading.Thread(target=self._flusher, daemon=True)
        self.thread.start()

    def should_flush(self):
        return time.time() - self.last_flush >= self.flush_interval or len(self.buffer) > 8 * 1024 * 1024

    def write(self, s):
        with self.lock:
            self.buffer.append(s)
            
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
            self.last_flush = time.time()

    def _flusher(self):
        while not self._stop_thread:
            time.sleep(1)
            with self.lock:
                if self.should_flush():
                    self._flush()

    def stop(self):
        self._stop_thread = True
        self.thread.join()
        self.flush()


def force_flush():
    # We do not want to use unbuffered python 
    # but we still want all the logs when the job crashes
    handlers = {}

    def flush(name):
        def flush_streams():
            try:
                sys.stdout.flush()
                sys.stderr.flush()
                print(f"Flushing {name}", file=sys.stderr)
            except ValueError:
                print("Couldn't do last flush")
        return flush_streams


    def handle_signal(signum, frame):
        flush(f"signal {signum}")()
        if (handler := handlers[signum]) not in (None, signal.SIG_IGN, signal.SIG_DFL):
            handler(signum, frame)

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
    if not isinstance(sys.stdout, TimedFlushBuffer):
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)

        print("Installing TimedFlush", file=sys.stderr, flush=True)
        # Replace sys.stdout/stderr
        sys.stdout = TimedFlushBuffer(sys.stdout, flush_interval=30)
        sys.stderr = TimedFlushBuffer(sys.stderr, flush_interval=30)


patch_tqdm()
force_flush()