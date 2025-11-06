import time
import multiprocessing as mp
from cantilever.core.timer import timeit, show_timings, reset

LOADER_TO_ITER = 2
ITER_TO_ITER = 2
NEXT = 1
WORK = 1


class FakeIterator:
    def __init__(self):
        self.i = 0
        
    def __iter__(self):
        time.sleep(ITER_TO_ITER)
        return self
        
    def __next__(self):
        time.sleep(NEXT)
        self.i += 1
        
        if self.i >= 10:
            raise StopIteration()
        
        return [self.i]


class FakeDataloader:
    def __iter__(self):
        time.sleep(LOADER_TO_ITER)
        return FakeIterator()
    



def test_timed_iterator():
    from benchmate.metrics import TimedIterator, file_push
    from benchmate.observer import BenchObserver

    data = FakeDataloader()
    loader = TimedIterator.with_stdout(data)

    for i in loader:
        print(i)
        time.sleep(WORK)
        

def test_observer():
    from benchmate.metrics import TimedIterator, file_push
    from benchmate.observer import BenchObserver

    obs = BenchObserver(stdout=True)
    obs.pusher = file_push()
    data = FakeDataloader()


    for i in obs.loader(data):
        print(i)
        time.sleep(WORK)


    show_timings(force=True)


class FakeDataset:
    def __init__(self, s, n):
        self.s = s
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        time.sleep(self.s)
        return i


def test_double_iterator(s=0.1, c=0.1):
    from benchmate.metrics import TimedIterator
    from torch.utils.data import DataLoader
    
    print("===")
    data = DataLoader(
        FakeDataset(s),
        batch_size=16,
        num_workers=4,
        collate_fn=collate,
    )

    loader_1 = TimedIterator.with_stdout(data, earlystop=10)
     #loader_2 = TimedIterator.with_stdout(data, earlystop=10)

    iter_1 = iter(loader_1)
    iter_2 = iter(loader_1)

    for e in range(2):
        for i in iter_2:
            time.sleep(1)



def test_dataloader_timed_iterator():
    # dataloader_run(0)
    
    dataloader_run(0, 0.1, 1, 0.1)

    dataloader_run(1, 0.1, 1, 0.1)
    # 16 / (16 * 0.5 + 1) = 1.777777
    # 16 / (16 * 0.5 - 1) = 2.28
    # {"rate": 1.7759813184032993, "units": "items/s", "task": "train", "time": 1762312191.6276882}
    # {"rate": 1.999982297577191, "units": "items/s", "task": "train", "time": 1762312199.627759}
    # {"rate": 1.9994120538611577, "units": "items/s", "task": "train", "time": 1762312207.6301115}
    # {"rate": 1.9995584270970586, "units": "items/s", "task": "train", "time": 1762312215.6318781}
    # {"rate": 1.9995995965692595, "units": "items/s", "task": "train", "time": 1762312223.63348}
    # {"rate": 1.9994159854628586, "units": "items/s", "task": "train", "time": 1762312231.6358168}
    # {"rate": 1.999392157810872, "units": "items/s", "task": "train", "time": 1762312239.638249}
    # {"rate": 1.9993749426857004, "units": "items/s", "task": "train", "time": 1762312247.64075}
    # {"rate": 1.9995662318884755, "units": "items/s", "task": "train", "time": 1762312255.6424854}
    # {"rate": 1.9993826864769488, "units": "items/s", "task": "train", "time": 1762312263.6449554}


    dataloader_run(2, 0.1, 1, 0.1)
    # {"rate": 1.7768168524624766, "units": "items/s", "task": "train", "time": 1762312277.67241}
    # {"rate": 15.978856407865607, "units": "items/s", "task": "train", "time": 1762312278.6737332}
    # {"rate": 2.285903479216933, "units": "items/s", "task": "train", "time": 1762312285.6731539}
    # {"rate": 15.929828759156662, "units": "items/s", "task": "train", "time": 1762312286.677559}
    # {"rate": 2.286569331192533, "units": "items/s", "task": "train", "time": 1762312293.6749413}
    # {"rate": 15.97771129874969, "units": "items/s", "task": "train", "time": 1762312294.6763363}
    # {"rate": 2.2854265069349786, "units": "items/s", "task": "train", "time": 1762312301.6772177}
    # {"rate": 15.978563457352038, "units": "items/s", "task": "train", "time": 1762312302.6785593}
    # {"rate": 2.2853087541508974, "units": "items/s", "task": "train", "time": 1762312309.6798015}
    # {"rate": 15.98164187656372, "units": "items/s", "task": "train", "time": 1762312310.6809502}

    dataloader_run(3, 0.1, 1, 0.1)
    # {"rate": 1.7764977153709705, "units": "items/s", "task": "train", "time": 1762312325.7252388}
    # {"rate": 15.963272671394597, "units": "items/s", "task": "train", "time": 1762312326.7275395}
    # {"rate": 15.979933188413769, "units": "items/s", "task": "train", "time": 1762312327.7287953}
    # {"rate": 2.6674725530076233, "units": "items/s", "task": "train", "time": 1762312333.7269826}
    # {"rate": 15.978517803694002, "units": "items/s", "task": "train", "time": 1762312334.728327}
    # {"rate": 15.979720103713973, "units": "items/s", "task": "train", "time": 1762312335.7295961}
    # {"rate": 2.6669744210926236, "units": "items/s", "task": "train", "time": 1762312341.7289038}
    # {"rate": 15.979933188413769, "units": "items/s", "task": "train", "time": 1762312342.7301595}
    # {"rate": 15.982593420328126, "units": "items/s", "task": "train", "time": 1762312343.7312486}
    # {"rate": 2.6670946172103998, "units": "items/s", "task": "train", "time": 1762312349.730286}

    dataloader_run(4, 0.1, 1, 0.1)    
    # {"rate": 1.7756260232810182, "units": "items/s", "task": "train", "time": 1762312365.7828047}
    # {"rate": 15.929212429410985, "units": "items/s", "task": "train", "time": 1762312366.7872486}
    # {"rate": 15.984059019356259, "units": "items/s", "task": "train", "time": 1762312367.788246}
    # {"rate": 15.965904567905728, "units": "items/s", "task": "train", "time": 1762312368.7903814}
    # {"rate": 3.1990103159461087, "units": "items/s", "task": "train", "time": 1762312373.7919283}
    # {"rate": 15.925349166402663, "units": "items/s", "task": "train", "time": 1762312374.7966158}
    # {"rate": 15.979172197722209, "units": "items/s", "task": "train", "time": 1762312375.7979193}
    # {"rate": 15.982163308513844, "units": "items/s", "task": "train", "time": 1762312376.7990353}
    # {"rate": 3.1950179918598267, "units": "items/s", "task": "train", "time": 1762312381.8068318}
    # {"rate": 15.92156332422057, "units": "items/s", "task": "train", "time": 1762312382.8117583}




c = 0.1
d = list(range(16))

def collate(*args):
    global c, d
    if c:
        time.sleep(c)
    return d



def get_rate(events, skip_event):
    acc = 0
    cnt = 0

    for e in events:
        if (rate := e.get("rate")) and (e.get("batch_id") not in skip_event):
            acc += rate
            cnt += 1

    return acc / cnt

def get_elapsed(events, skip_event):
    acc = 0

    for e in events:
        if (elapsed := e.get("elapsed")) and (e.get("batch_id") not in skip_event):
            acc += elapsed

    return acc


def dataloader_run(method, worker, s, w, collate_sleep=None):
    reset()

    from threading import get_native_id
    import os

    mp.set_start_method(method, force=True)
    mp.freeze_support()

    from benchmate.metrics import TimedIterator
    from torch.utils.data import DataLoader
    
    global c
    c = collate_sleep

    batch_size = 16
    batch_count = 10
    skip_batch = 0
    epoch = 2
    total_batch = batch_count + skip_batch * epoch
    total_samples = batch_size * total_batch // epoch

    print("===")
    data = DataLoader(
        FakeDataset(s,  total_samples),
        batch_size=batch_size,
        num_workers=worker,
        collate_fn=collate,
        persistent_workers=False,
    )


    events = []
    def push(**kwargs):
        nonlocal events
        events.append(kwargs)

    loader = TimedIterator(data, earlystop=total_batch, push=push)

    with timeit(f"train"):
        for e in range(2):
            with timeit("epoch") as epoch_time:
                with timeit("iter"):
                    iterator = iter(loader)

                for i in range(skip_batch):
                    with timeit(f"step_{i}"):
                        next(iterator)
                        time.sleep(w)

                while True:
                    with timeit(f"step_n") as step_time:
                        with timeit("data"):
                            try:
                                i = next(iterator)
                            except StopIteration:
                                break

                        with timeit("compute"):
                            time.sleep(w)



    elapsed = epoch_time.timing.value.avg * epoch_time.timing.value.count
    batch_time = step_time.timing.value.avg * step_time.timing.value.count
    samples = batch_count * batch_size

    show_timings(force=True)
    skip_batch_id = tuple() # (0, 1, 7, 8)

    mb_rate = get_rate(events, skip_batch_id)
    mb_elapsed = get_elapsed(events, skip_batch_id)
    print(method)
    print(f"  (epoch) with CPU Timer {samples / elapsed:5.2f} (item/s)    {elapsed:5.2f} s")
    print(f"  (batch) with CPU Timer {samples / batch_time:5.2f} (item/s)    {batch_time:5.2f} s")
    print(f"               milabench {mb_rate:5.2f} (item/s)    {mb_elapsed:5.2f} s")

    # to_csv(events)
    show_events(events)
    

def show_events(events):
    for e in events:
        if "rate" in e:
            print(e)

def to_csv(events):
    headers = None
    for e in events:
        if "rate" in e:
            if headers is None:
                headers = list(e.keys())
                print(",".join(headers))

            print(",".join([str(e[k]) for k in headers]))


if __name__ == "__main__":
    dataloader_run("fork", 6, 0.1, 1, 0.1)  
    dataloader_run("spawn", 6, 0.1, 1, 0.1)    
