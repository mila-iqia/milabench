import voir.instruments.gpu

from milabench.testing import replay_scenario, replay_validation_scenario
from milabench.utils import validation_layers


def test_error_layer(replayfolder):
    log = replay_scenario(replayfolder, "error")
    assert log.result() != 0


def test_error_layer_early_stop(replayfolder):
    log = replay_scenario(replayfolder, "error", "early_stop")
    assert log.result() == 0


def test_error_layer_early_stop_per_gpu(replayfolder):
    log = replay_scenario(replayfolder, "error", "early_stopping_per_gpu")
    assert log.result() == 0


def test_loss_layer(replayfolder):
    log = replay_scenario(replayfolder, "loss")
    assert log.result() != 0


def test_usage_layer_no_usage(replayfolder):
    log = replay_scenario(replayfolder, "usage", "no_usage")
    assert log.result() != 0


def test_usage_layer_usage(replayfolder):
    log = replay_scenario(replayfolder, "usage")
    assert log.result() == 0


def test_rate_layer(replayfolder):
    log = replay_scenario(replayfolder, "ensure_rate")
    assert log.result() != 0


def test_planning_layer_njobs_good(replayfolder):
    # Expected 3 jobs got 3 jobs
    log = replay_scenario(replayfolder, "planning", "planning_njobs_good")
    assert log.result() == 0


def test_planning_layer_njobs_bad(replayfolder):
    # Expected 3 jobs got 1 job
    log = replay_scenario(replayfolder, "planning", "planning_njobs_bad")
    assert log.result() != 0


def mock_gpu_info():
    return {
        "arch": "rocm",
        "gpus": [0, 1],
    }


def test_planning_layer_per_gpu_good(replayfolder, monkeypatch):
    # 2 GPU detected; expected 2 jobs got 2 jobs
    monkeypatch.setattr(voir.instruments.gpu, "get_gpu_info", mock_gpu_info)

    log = replay_scenario(replayfolder, "planning", "planning_per_gpu_good")
    assert log.result() == 0


def test_planning_layer_per_gpu_bad(replayfolder, monkeypatch):
    # 2 GPU detected; expected 2 jobs got 1 job
    monkeypatch.setattr(voir.instruments.gpu, "get_gpu_info", mock_gpu_info)

    log = replay_scenario(replayfolder, "planning", "planning_per_gpu_bad")
    assert log.result() != 0


def test_memory_tracking(replayfolder, config, tmp_path):
    import contextvars

    from milabench.sizer import (
        MemoryUsageExtractor,
        Sizer,
        SizerOptions,
        sizer_global,
        system_global,
    )

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
        system_global.set({"gpu": {"capacity": "41920 MiB"}})

    ctx.run(update_ctx)
    layer = ctx.run(lambda: MemoryUsageExtractor())

    layer.filepath = f"{tmp_path}/dummy"

    assert 123 not in layer.memory["benchio"]["model"]

    ctx.run(lambda: replay_validation_scenario(replayfolder, layer, filename="usage"))

    assert 123 in layer.memory["benchio"]["model"]


def test_exception_tracking(replayfolder, file_regression, capsys):
    layers = validation_layers("error")
    _ = replay_validation_scenario(replayfolder, *layers, filename="exception")
    error = layers[0]

    from milabench.utils import Summary

    # summary = Summary()
    # return_code = error.report(summary, short=False)
    # summary.show()
    # assert return_code != 0

    output = capsys.readouterr().out
    file_regression.check(output)
