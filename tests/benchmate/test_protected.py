

import time
import multiprocessing 
import signal
import os
import subprocess

import pytest

from benchmate.warden import process_cleaner


def _worker(delay):
    time.sleep(delay)
    print('done')



def spawn(delay, warden):
    procs = []
    for _ in range(10):
        proc = multiprocessing.Process(target=_worker, args=(delay,))
        proc.start()
        procs.append(proc)
        warden.add_process(proc)
    
    return procs

def wait(procs):
    for proc in procs:
        proc.join()


def _protected_process(delay):
    with process_cleaner() as warden:
        procs = spawn(delay, warden)

        wait(procs)


def test_process_cleaner_process_ended_already():
    start = time.time()

    proc = multiprocessing.Process(target=_protected_process, args=(1,))
    proc.start()

    time.sleep(2)
    os.kill(proc.pid, signal.SIGINT)

    elapsed = time.time() - start
    assert elapsed < 30


def test_process_cleaner_process():
    start = time.time()

    proc = multiprocessing.Process(target=_protected_process, args=(60,))
    proc.start()

    time.sleep(1)
    os.kill(proc.pid, signal.SIGINT)

    elapsed = time.time() - start
    assert elapsed < 30


def test_keyboard_cleaner_process():
    start = time.time()

    with pytest.raises(KeyboardInterrupt):
        with process_cleaner() as warden:
            procs = spawn(60, warden)

            time.sleep(1)
            os.kill(os.getpid(), signal.SIGINT)

            wait(procs)
    
    elapsed = time.time() - start
    assert elapsed < 30


def test_keyboard_cleaner_process_ended():
    start = time.time()

    with pytest.raises(KeyboardInterrupt):
        with process_cleaner() as warden:
            procs = spawn(1, warden)

            time.sleep(2)
            os.kill(os.getpid(), signal.SIGINT)

            wait(procs)
    
    elapsed = time.time() - start
    assert elapsed < 30


def test_protected_multiplexer():
    from voir.proc import Multiplexer

    start = time.time()

    def ctor(*args, **kwargs):
        return kwargs

    with pytest.raises(KeyboardInterrupt):
        with process_cleaner() as warden:
            mx = Multiplexer(timeout=0, constructor=ctor)
            proc = mx.start(["sleep", "60"], info={}, env={}, **{})
            warden.add_process(proc)

            time.sleep(2)
            os.kill(os.getpid(), signal.SIGINT)

            for entry in mx:
                if entry:
                    print(entry)
    
    elapsed = time.time() - start
    assert elapsed < 30


def test_protected_multiplexer_ended():
    from voir.proc import Multiplexer

    start = time.time()

    with pytest.raises(KeyboardInterrupt):
        with process_cleaner() as warden:
            mx = Multiplexer(timeout=0, constructor=lambda **kwargs: kwargs)
            proc = mx.start(["sleep", "1"], info={}, env={}, **{})
            warden.add_process(proc)

            time.sleep(2)
            os.kill(os.getpid(), signal.SIGINT)

            for entry in mx:
                if entry:
                    print(entry)
    
    elapsed = time.time() - start
    assert elapsed < 30