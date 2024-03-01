import json
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import ParseResult, urlparse

from coleo import Option, tooled

SLEEP = 0.01
_INIT = 0
_READY = 1
_STOP = 2


def _process_kill(process):
    process.terminate()
    process.kill()


AUTHENTICATION = re.compile(r"Authenticated to (.*) \(via proxy\) using \"publickey\"")
LOCAL_FORWARDING = re.compile(
    r"debug1: Local forwarding listening on 127\.0\.0\.1 port \d{4,}"
)


class _WaitForGo:
    def __init__(self) -> None:
        self.authenticated = False
        self.forwarded = False

    @property
    def ready(self):
        return self.authenticated and self.forwarded

    def match(self, line):
        if AUTHENTICATION.match(line):
            self.authenticated = True
            print("SSH", line, end="")

        if LOCAL_FORWARDING.match(line):
            self.forwarded = True
            print("SSH", line, end="")


def _mp_worker(cmd, status, states, timeout=60):
    shell = True
    fmt = _WaitForGo()
    start = time.time()

    def read_output(process):
        while status.value != _STOP:
            try:
                line = process.stdout.readline()
                if len(line) > 0:
                    fmt.match(line)
            except ValueError:
                if process.poll() is not None:
                    return
                raise

    def wait_ready(process, timeout):
        # Wait for UE to finish loading its stuff
        while (process.poll() is None) and (not fmt.ready):
            if time.time() - start > timeout:
                raise TimeoutError("")

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # This is needed because without lines might not be recognized as such
        text=True,
        shell=shell,
    ) as process:
        try:
            states["pid"] = process.pid
            states["status"] = "starting"

            stdout_reader = threading.Thread(target=read_output, args=(process,))
            stdout_reader.start()

            wait_ready(process, timeout=timeout)

            states["status"] = "running"
            status.value = _READY

            while process.poll() is None:
                time.sleep(SLEEP)

                if status.value == _STOP:
                    states["status"] = "stopping"
                    _process_kill(process)
                    break

        except KeyboardInterrupt:
            states["status"] = "interrupted"

        return_code = process.poll() or 0
        states["status"] = "stopped"
        states["return_code"] = return_code
        return return_code or 0 + int(not fmt.ready)


class ReverseProxyProcess:
    def __init__(self, cmd, manager: multiprocessing.Manager, close) -> None:
        self.status = manager.Value("i", _INIT)
        self.states = manager.dict()
        self.states["status"] = "None"
        self.close = close
        self.proc = multiprocessing.Process(
            target=_mp_worker, args=(cmd, self.status, self.states)
        )
        self.proc.start()

    def wait(self):
        while self.states["status"] != "running":
            time.sleep(SLEEP)

    def is_alive(self):
        return self.proc.is_alive()

    def interrupt(self):
        self.stop()

    def stop(self):
        if not self.proc.is_alive():
            return

        if self.states["status"] == "running":
            self.close()

        signum = signal.SIGINT
        pid = self.states["pid"]
        start = time.time()
        self.status.value = _STOP

        while self.proc.is_alive() and self.states["status"] == "running":
            try:
                os.kill(pid, signum)
            except SystemError:
                pass

            time.sleep(SLEEP)

            if time.time() - start > 30:
                raise RuntimeError("Could not shutdown ssh")

        while self.proc.is_alive():
            time.sleep(SLEEP)

            if time.time() - start > 30:
                raise RuntimeError("Could not shutdown ssh")

        print(f"Shutdown after {time.time() - start}")

    def join(self):
        self.proc.join()


def get_open_port():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


@contextmanager
def reverse_proxy(uri, enabled):
    if not enabled:
        yield uri

    localport = get_open_port()
    localhost = "localhost"

    parsed: ParseResult = urlparse(uri)
    hostname = parsed.hostname
    port = parsed.port or 5432  # default port

    if parsed.port is None:
        newuri = uri.replace(hostname, f"{localhost}:{localport}")
    else:
        newuri = uri.replace(hostname, localhost).replace(str(port), str(localport))

    local = f"{localhost}:{localport}"
    remote = f"localhost:{port}"

    command = [
        "ssh",
        hostname,
        "-CNL",
        f"{local}:{remote}",
        "-vvv",
    ]
    p = ReverseProxyProcess(" ".join(command), multiprocessing.Manager(), lambda: None)
    p.wait()
    print("Reverse proxy ready")

    yield newuri

    print("Stopping")
    p.stop()


# fmt: off
@dataclass
class Arguments:
    uri: str
    folder: str
    meta: str = None
    testing: bool = True
# fmt: on


@tooled
def arguments():
    # URI to the database
    #   ex:
    #       - postgresql://user:password@hostname:27017/database
    #       - postgresql://milabench_write:1234@milabenchdb:5432/milabench
    #       - sqlite:///sqlite.db
    uri: Option & str

    # Run folder to save
    folder: Option & str

    # Json string of file to append to the meta dictionary
    meta: Option & str = None

    testing: Option & bool = True

    return Arguments(uri, folder, meta, testing)


@tooled
def cli_publish(args=None):
    """Publish an archived run to a database"""

    from ..metrics.archive import publish_archived_run
    from ..metrics.sqlalchemy import SQLAlchemy

    if args is None:
        args = arguments()

    if args.meta is not None:
        with open(args.meta, "r") as file:
            args.meta = json.load(file)

    with reverse_proxy(args.uri, enabled=args.testing) as uri:
        backend = SQLAlchemy(uri, meta_override=args.meta)
        publish_archived_run(backend, args.folder)

    sys.exit(0)
