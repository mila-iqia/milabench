"""
For resuming we could have a status.jsonl file that will store the run status throughout the run.
and then reconcile between the expected and what we got and resume the missing things.

Or we can have a special --resume flags, (we need to fix the name of the run though)
Then we check for a `bench.D0.data` file and check if it is valid. if not re run that 

"""
import json
import time
import os
import traceback

from filelock import FileLock


def serialize_exception(exc_type, exc_value, exc_tb):
    return {
        "type": exc_type.__name__ if exc_type else None,
        "message": str(exc_value) if exc_value else None,
        "traceback": traceback.format_exception(
            exc_type, exc_value, exc_tb
        ) if exc_tb else None,
    }


class StatusTracker:
    """This status tracker works by appending events to the status file"""
    def __init__(self, packs, repeat):
        self.repeat = repeat
        self.packs = packs

        assert len(self.packs) > 0, "No packs available"
        first_pack = next(self.packs.values())

        logdir = first_pack.logdir()
        self.sentinel = logdir / "status.jsonl"

    def __enter__(self):
        self.append(phase="run", status="started", time=time.time())
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is None:
            self.append(phase="run", status="ended", time=time.time())
        else:
            self.append(phase="run", status="error", time=time.time(), error=serialize_exception(exc_type, exc_value, exc_tb))
    
    def append(self, obj):
        with FileLock(self.sentinel + ".lock"):
            with open(self.sentinel, "a") as fp:
                json.dump(obj, fp)
                fp.write("\n")

    def __iter__(self):
        for index in range(self.repeat):
            for pack in self.packs.values():
                self.append(bench=pack.config["name"], status="started", time=time.time(), repeat=index)
                yield index, pack
                self.append(bench=pack.config["name"], status="ended", time=time.time(), repeat=index)




def resume_from_status(packs, runfolder, repeat):
    if repeat > 1:
        print("repeat > 1 is not supported")


    assert len(packs) > 0, "No packs available"
    first_pack = next(packs.values())

    logdir = first_pack.logdir()
    sentinel = logdir / "status.jsonl"

    status_line = []
    with FileLock(sentinel + ".lock"):
        with open(sentinel, "r") as fp:
            for line in fp.readlines():
                status_line.append(json.loads(line))
    
    # Need to reoncile the list of what we have
    # and the list of what we want
    want = {}
    have = {}

    return missing_packs