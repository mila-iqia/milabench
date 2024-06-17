import time

import pytest

from benchmate.metrics import StopProgram, TimedIterator


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
