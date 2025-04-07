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
    # This cannot work anymore because the batch resizing logic is not executed here
    if True:
        return
    
    import contextvars
    import yaml
    from milabench.system import apply_system, option
    
    conf = {
        "gpu": {
            "capacity": "41920 MiB"
        },
        "options": {
            "sizer": {
                "multiple": 8,
                "autoscale": 1
            }
        }
    }
    
    with apply_system(conf):
        from milabench.sizer import (
            MemoryUsageExtractor,
            Sizer,
            SizerOptions,
            sizer_global,
            system_global,
        )
        
        layer = MemoryUsageExtractor()
        with open(config("scaling"), "r") as sconf:
            layer.memory = yaml.safe_load(sconf)
            
        layer.filepath = f"{tmp_path}/dummy"

        print(system_global.get())
        # print(option("sizer.multiple", etype=int))
        # print(option("sizer.config", etype=str))
        # print(Sizer().scaling_config)
        assert 123 not in layer.memory["benchio"]["model"]

        replay_validation_scenario(replayfolder, layer, filename="usage")

        # print(layer.memory)
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
