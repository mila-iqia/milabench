from giving import give, given

from milabench.utils import validation


class FakePack:
    def __init__(self, method, njobs) -> None:
        self.config = dict(name="fake", plan=dict(method=method, njobs=njobs))


def pack(method, njobs):
    return FakePack(method, njobs)


def newevent(method="per_gpu", njobs=0, **data):
    base = {
        "#run": dict(tag=["a", "b", "c"]),
        "#pack": pack(method, njobs),
    }
    base.update(data)
    return base


def gpudata(n, load, memory):
    base = dict(load=load, memory=memory)

    data = dict()
    for i in range(n):
        data[i] = base

    return {"gpudata": data}


def test_nan_layer():
    nan_replay = [
        newevent(loss=float("10")),
        newevent(loss=float("5")),
        newevent(loss=float("7")),
        newevent(loss=float("8")),
        newevent(loss=float("nan")),
        newevent(loss=float("nan")),
    ]
    with given() as gv:
        with validation("nan") as validations:
            for msg in nan_replay:
                give(**msg)


def test_planning_layer():
    planning_replay = []
    with given() as gv:
        with validation("planning") as validations:
            for msg in planning_replay:
                give(**msg)


def test_usage_layer_no_usage():
    usage_replay = [
        newevent(**gpudata(2, 0, (0, 1))),
        newevent(**gpudata(2, 0, (0, 1))),
        newevent(**gpudata(2, 0, (0, 1))),
        newevent(**gpudata(2, 0, (0, 1))),
    ]
    with given() as gv:
        with validation("usage") as validations:
            for msg in usage_replay:
                give(**msg)


def test_usage_layer_usage():
    usage_replay = [
        newevent(**gpudata(2, 1, (0.9, 1))),
        newevent(**gpudata(2, 1, (0.9, 1))),
        newevent(**gpudata(2, 1, (0.9, 1))),
        newevent(**gpudata(2, 1, (0.9, 1))),
    ]
    with given() as gv:
        with validation("usage") as validations:
            gv.subscribe(validation)
            
            for msg in usage_replay:
                give(**msg)
