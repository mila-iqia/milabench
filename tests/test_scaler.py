import pytest

from milabench.sizer import Sizer, SizerOptions, sizer_global


def test_scaler_use_override(multipack, config):
    sizer = Sizer(SizerOptions(size=64, autoscale=False), config("scaling"))
    for k, pack in multipack.packs.items():
        assert sizer.size(pack, "48Go") == 64


def test_scaler_use_optimized(multipack, config):
    sizer = Sizer(
        SizerOptions(
            size=None,
            autoscale=False,
            optimized=True,
        ),
        config("scaling"),
    )
    for k, pack in multipack.packs.items():
        assert sizer.size(pack, "48Go") == 138


_values = [
    ("5Go", 27),  # Not a multiple of 8
    ("6Go", 32),
    ("12Go", 64),
    ("18Go", 96),
    ("24Go", 128),
    ("30Go", 160),
    ("48Go", 256),
    ("72Go", 384),
]


@pytest.mark.parametrize("capacity,expected", _values)
def test_scaler_autoscaler_lerp(multipack, config, capacity, expected):
    sizer = Sizer(SizerOptions(size=None, autoscale=True), config("scaling"))
    for k, pack in multipack.packs.items():
        assert sizer.size(pack, capacity) == expected


_values_2 = [
    ("5Go", 24),  # a multiple of 8
    ("6Go", 32),
]


@pytest.mark.parametrize("capacity,expected", _values_2)
def test_scaler_autoscaler_lerp_multiple(multipack, config, capacity, expected):
    sizer = Sizer(
        SizerOptions(
            size=None,
            autoscale=True,
            multiple=8,
        ),
        config("scaling"),
    )
    for k, pack in multipack.packs.items():
        assert sizer.size(pack, capacity) == expected


def test_scaler_disabled(multipack):
    for k, pack in multipack.packs.items():
        assert pack.argv == []


def fakeexec(pack):
    from milabench.sizer import resolve_argv, scale_argv
    sized_args = scale_argv(pack, pack.argv)
    final_args = resolve_argv(pack, sized_args)
    return final_args


def test_scaler_enabled(multipack, config):
    from milabench.config import system_global
    import contextvars

    ctx = contextvars.copy_context()

    def update_ctx():
        sizer = Sizer(
            SizerOptions(
                size=None,
                autoscale=True,
                multiple=8,
            ),
            config("scaling"),
        )
        sizer_global.set(sizer)
        system = system_global.get()
        gpu = system.setdefault("gpu", dict())
        gpu["capacity"] = "41920 MiB"

    ctx.run(update_ctx)

    for k, pack in multipack.packs.items():
        assert ctx.run(lambda: fakeexec(pack)) == ["--batch_size", "232"]

        # Sizer is only enabled inside the context
        assert fakeexec(pack) == []
