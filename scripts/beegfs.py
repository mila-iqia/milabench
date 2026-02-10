#!/usr/bin/env python3
import os
import io
import sys
import re
import json
import time
import selectors
import subprocess
import argparse
from contextlib import contextmanager

#
# TODO: make this a voir monitor ?
#

# -----------------------------
# Parser for lines
# -----------------------------
def make_parser(source):
    """Return a closure that parses lines for a given source."""
    elapsed = None
    started = False

    def parse_line(line):
        nonlocal elapsed, started
        line = line.strip()
        if not line:
            return None

        # header
        m = re.match(r"=+\s+(\d+)\s+s\s+=+", line)
        if m:
            elapsed = int(m.group(1))
            started = True
            return None

        if not started:
            return None

        parts = line.split()
        if len(parts) < 2:
            return None

        try:
            uid = int(parts[0])
        except ValueError:
            uid = parts[0]

        data = {"time": time.time(), "elapsed": elapsed, "uid": uid, "source": source}

        i = 1
        while i < len(parts):
            if parts[i].isdigit() and i + 1 < len(parts) and parts[i + 1].startswith("["):
                key = parts[i + 1].strip("[]")
                data[key] = int(parts[i])
                i += 2
            else:
                i += 1

        return data

    return parse_line

# -----------------------------
# Live display
# -----------------------------
@contextmanager
def live_display(args):
    if args.pipe:
        yield None
    else:
        from rich.live import Live
        from rich.table import Table

        table = Table()
        table.add_column("stats")
        table.add_column("Description")
        table.add_column("Level")

        with Live(table, refresh_per_second=1/5):
            yield table

# -----------------------------
# Spawn subprocess with PTY
# -----------------------------
def spawn_with_pty(cmd):
    """Spawn a subprocess with a pseudo-TTY and return a file-like object for reading output."""
    master_fd, slave_fd = os.openpty()  # create a pty pair
    proc = subprocess.Popen(
        cmd,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        text=True
    )
    os.close(slave_fd)  # parent closes slave
    # wrap master_fd into a text stream (line-buffered)
    return proc, io.open(master_fd, "r", encoding="utf-8", errors="replace", buffering=1)

# -----------------------------
# Strip ANSI/control codes
# -----------------------------
def clean_line(line):
    # remove ANSI escape sequences
    line = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line)
    # convert carriage return line updates (\r) to \n
    line = line.replace('\r', '\n')
    return line

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipe", action="store_true", default=False)
    args = parser.parse_args()

    user_uid = subprocess.getoutput("id -u $USER")

    CMD_METADATA = [
        "beegfs-ctl",
        "--userstats",
        f"--filter={user_uid}",
        "--nodetype=metadata",
        "--cfgFile=/etc/beegfs/scratch.d/beegfs-client.conf",
        "--allstats",
    ]

    CMD_STORAGE = [
        "beegfs-ctl",
        "--userstats",
        f"--filter={user_uid}",
        "--nodetype=storage",
        "--cfgFile=/etc/beegfs/scratch.d/beegfs-client.conf",
        "--allstats",
    ]

    # spawn both processes with pseudo-tty
    proc_meta, stream_meta = spawn_with_pty(CMD_METADATA)
    proc_storage, stream_storage = spawn_with_pty(CMD_STORAGE)

    sel = selectors.DefaultSelector()
    sel.register(stream_meta, selectors.EVENT_READ, make_parser("metadata"))
    sel.register(stream_storage, selectors.EVENT_READ, make_parser("storage"))

    with live_display(args) as table:
        while True:
            events = sel.select(timeout=1.0)
            # exit if both processes have finished
            if not events and proc_meta.poll() is not None and proc_storage.poll() is not None:
                break

            for key, _ in events:
                line = key.fileobj.readline()
                if not line:  # EOF
                    sel.unregister(key.fileobj)
                    continue

                line = clean_line(line)
                parser = key.data
                data = parser(line)

                if data:
                    if args.pipe:
                        sys.stdout.write(json.dumps(data) + "\n")
                        sys.stdout.flush()
                    elif table is not None:
                        pass  # optionally update rich table

    # wait for both processes to exit
    proc_meta.wait()
    proc_storage.wait()

if __name__ == "__main__":
    main()