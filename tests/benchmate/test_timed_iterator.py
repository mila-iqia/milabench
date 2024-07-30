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


def fake_run(dataset_size, accumulation_steps = 32, loss_backward = 0.01, optimizer_step = 0.01, eps=2):
    batch = [1, 2]
    loss_backward = 0.01
    optimizer_step = 0.01

    iterable = [(batch, 3) for i in range(dataset_size)]
    messages = []

    def push(**kwargs):
        nonlocal messages
        messages.append(kwargs)

    min_steps = 50
    loader = ManualTimedIterator(
        iterable, 
        event_fn=CPUEvent, 
        earlystop=min_steps, 
        raise_stop_program=True, 
        push=push,
        batch_size_fn=lambda x: len(x[0])
    )

    accumulation_steps = 32
    epochs = (len(iterable) // accumulation_steps) * min_steps * 10

    normal_batch_count = len(iterable) // accumulation_steps - 1
    bigger_batch = 1

    normal_batch_size = len(batch) * accumulation_steps 
    bigger_batch_size = (normal_batch_size + len(batch) * len(iterable) % accumulation_steps)

    normal_step_time = loss_backward * accumulation_steps + optimizer_step
    bigger_step_time = loss_backward * (accumulation_steps + len(iterable) % accumulation_steps) + optimizer_step

    step_batch_size = (normal_batch_size * normal_batch_count +  bigger_batch_size * bigger_batch)/ (normal_batch_count + bigger_batch)
    step_batch_time = (normal_step_time * normal_batch_count +  bigger_step_time * bigger_batch)/ (normal_batch_count + bigger_batch)

    expected_rate = step_batch_size / step_batch_time
    
    with pytest.raises(StopProgram):
        for e in range(epochs):
            for j, _ in enumerate(loader):
                time.sleep(loss_backward)
                if (j + 1) % accumulation_steps == 0:
                    time.sleep(optimizer_step)
                    loader.step()

    rate_acc = 0
    rate_count = 0
    for msg in messages:
        if rate := msg.get("rate"):
            rate_acc += rate
            rate_count += 1

    empirical_rate = rate_acc / rate_count
    print(expected_rate)
    print(empirical_rate)
    print()

    assert (
        abs(empirical_rate - expected_rate) < eps
    ), "Computed rate should be close to theorical rate"

    assert rate_count == 50, "Program should stop once we reached the necessary count"

def test_accumulation_steps_perfect():
    fake_run(dataset_size=128, accumulation_steps=32, eps=2)
    
    # all steps have the same amount of batches
    # expected_rate : 193.93939393939394
    # empirical_rate: 192.66509115668907

def test_accumulation_steps_dataloader_is_bigger():
    fake_run(dataset_size=400 + 8, accumulation_steps=32, eps=18)

    # only the first batch of the epoch (1+) is wrong
    # so it is amortized nicely and does not impact us
    # expected_rate : 186.66666666666663
    # empirical_rate: 204.35944379315433

def test_accumulation_steps_dataloader_is_small():
    fake_run(dataset_size=40, accumulation_steps=32, eps=45)

    # only the first batch is "normal" which makes it wrong
    # `expected_rate` computation looks wrong here
    # expected_rate : 195.12195121951217
    # empirical_rate: 239.94828507678048