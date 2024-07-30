import time

import pytest

from benchmate.metrics import StopProgram, TimedIterator, ManualTimedIterator


class CPUEvent:
    def __init__(self, **kwargs):
        self.start = 0

    def record(self):
        self.start = time.time()

    def elapsed_time(self, end):
        # shoudl return ms
        return (end.start - self.start) * 1000

    def synchronize(self):
        pass


def test_wrapper():
    batch = [1, 2]
    process_time = 0.1

    iterable = [(batch, 3) for i in range(10)]
    messages = []

    def push(**kwargs):
        nonlocal messages
        messages.append(kwargs)

    loader = TimedIterator(
        iterable, event_fn=CPUEvent, earlystop=50, raise_stop_program=True, push=push
    )

    with pytest.raises(StopProgram):
        for e in range(200):
            for i in loader:
                time.sleep(process_time)

    assert len(messages) == 117

    rate_acc = 0
    rate_count = 0
    for msg in messages:
        if rate := msg.get("rate"):
            rate_acc += rate
            rate_count += 1

    assert rate_count == 50, "Program should stop once we reached the necessary count"
    assert (
        abs((rate_acc / rate_count) - len(batch) / process_time) < 0.5
    ), "Computed rate should be close to theorical rate"


def test_accumulation_steps():
    batch = [1, 2]
    process_time = 0.1

    iterable = [(batch, 3) for i in range(40)]
    messages = []

    def push(**kwargs):
        nonlocal messages
        messages.append(kwargs)

    loader = ManualTimedIterator(
        iterable, 
        event_fn=CPUEvent, 
        earlystop=50, 
        raise_stop_program=True, 
        push=push,
        batch_size_fn=lambda x: len(x[0])
    )

    accumulation_steps = 32
    min_steps = 50
    epochs = (len(iterable) // accumulation_steps) * min_steps * 10

    with pytest.raises(StopProgram):
        for e in range(epochs):
            for j, _ in enumerate(loader):
                if (j + 1) % accumulation_steps == 0:
                    time.sleep(process_time)
                    loader.step()

    # there is more message here because we are doing more epochs
    # to get the same amount of observations
    assert len(messages) == 252

    rate_acc = 0
    rate_count = 0
    for msg in messages:
        if rate := msg.get("rate"):
            rate_acc += rate
            rate_count += 1

    print()
    print(len(batch) * accumulation_steps / process_time)
    print(rate_acc / rate_count)
    print()

    assert (
        abs((rate_acc / rate_count) - len(batch) * accumulation_steps / process_time) < 2
    ), "Computed rate should be close to theorical rate"

    assert rate_count == 50, "Program should stop once we reached the necessary count"
