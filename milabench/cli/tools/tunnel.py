"""Open a SSH tunnel to forward a local port to a remote port."""

import atexit
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

from argklass.command import Command


HOSTNAME = "milabench_sql"
DB_PORT = 5432


def start_tunnel(local_port=DB_PORT, hostname=HOSTNAME, db_port=None, stop_event=None):
    if db_port is None:
        db_port = local_port

    while not (stop_event and stop_event.is_set()):
        try:
            proc = subprocess.Popen([
                "ssh",
                "-v",
                "-N",
                "-L", f"127.0.0.1:{local_port}:127.0.0.1:{db_port}",
                "-o", "ExitOnForwardFailure=yes",
                "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=3",
                hostname,
            ])
            print("Tunnel established, PID:", proc.pid, file=sys.stderr)

            with open(".tunnel", "w") as fp:
                fp.write(f"{proc.pid}")

            def cleanup():
                if os.path.exists(".tunnel"):
                    os.remove(".tunnel")
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
                    print("Tunnel terminated", file=sys.stderr)

            atexit.register(cleanup)

            while proc.poll() is None and not (stop_event and stop_event.is_set()):
                time.sleep(0.5)

            if stop_event and stop_event.is_set():
                cleanup()
                break

            print("Tunnel dropped, retrying in 5s...", file=sys.stderr)
            if os.path.exists(".tunnel"):
                os.remove(".tunnel")
            time.sleep(5)

        except Exception as e:
            print("Tunnel error:", e, file=sys.stderr)
            time.sleep(5)


class Tunnel(Command):
    """Open a SSH tunnel to forward a local port to a remote port."""

    name = "tunnel"

    # fmt: off
    @dataclass
    class Arguments:
        """Open a SSH tunnel to forward a local port to a remote port."""
        local_port  : int = DB_PORT    # Local port to forward
        remote_port : int = DB_PORT    # Remote port to connect to
        hostname    : str = HOSTNAME   # SSH hostname
    # fmt: on

    @staticmethod
    def execute(args):
        stop_event = threading.Event()
        kwargs = {
            "stop_event": stop_event,
            "local_port": args.local_port,
            "db_port": args.remote_port,
            "hostname": args.hostname,
        }
        tunnel_thread = threading.Thread(target=start_tunnel, kwargs=kwargs, daemon=True)
        tunnel_thread.start()

        try:
            while tunnel_thread.is_alive():
                tunnel_thread.join(timeout=1)
        except KeyboardInterrupt:
            print("Keyboard interrupt received, shutting down tunnel...", file=sys.stderr)
            stop_event.set()
            tunnel_thread.join()
            print("Tunnel closed, exiting.", file=sys.stderr)


COMMANDS = Tunnel
