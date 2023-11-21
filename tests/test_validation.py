import os

import voir.instruments.gpu

from milabench.utils import validation_layers, multilogger
from milabench.testing import interleave, replay


def replay_validation_scenario(folder, *validation, filename=None):
    """Replay events from a data file or folder"""
    gen = None

    path = folder / filename
    file = str(path) + ".txt"
    
    if os.path.isdir(path):
        files = [path / f for f in os.scandir(path)]
        gen = interleave(*files)

    if os.path.isfile(file):
        gen = replay(file)

    with multilogger(*validation) as log:
        for entry in gen:
            log(entry)

    return log


def replay_scenario(folder, name, filename=None):
    """Replay events from a data file or folder"""
    return replay_validation_scenario(
        folder, 
        *validation_layers(name), 
        filename=filename or name
    )
  
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
    
    
def test_memory_tracking(replayfolder, config):
    import contextvars
    from milabench.sizer import (
        MemoryUsageExtractor, Sizer, SizerOptions, sizer_global, system_global)

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
        system = system_global.set({
            "gpu": {
                "capacity": "41920 MiB"
            }
        })
        
    ctx.run(update_ctx)
    layer = MemoryUsageExtractor()

    ctx.run(lambda: replay_validation_scenario(
        replayfolder,
        layer,
        filename="usage"
    ))
    