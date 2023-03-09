import json
import pytest

from milabench.testing import milabench_cmd, resultfolder, has_result_folder


skip_tests = not has_result_folder()


_output = \
"""
                                       |   rufidoko |   sokilupa
                                       | 2023-02-23 | 2023-03-09
bench                |          metric |   16:16:31 |   16:02:53
----------------------------------------------------------------
bert                 |      train_rate |     243.05 |     158.50
convnext_large       |      train_rate |     210.11 |     216.23
dlrm                 |      train_rate |  338294.94 |  294967.41
efficientnet_b0      |      train_rate |     223.56 |     223.48
efficientnet_b4      |      train_rate |     220.81 |     214.64
efficientnet_b7      |      train_rate |     215.78 |     218.82
hf_reformer          |      train_rate |      44.90 |      32.09
hf_t5                |      train_rate |      31.99 |      21.40
learning_to_paint    |      train_rate |    4388.38 |    5601.29
ppo                  |      train_rate |    1258.90 |    1590.76
regnet_y_128gf       |      train_rate |      58.48 |      50.64
resnet152            |      train_rate |     340.08 |     498.87
resnet50             |      train_rate |     221.56 |     218.80
soft_actor_critic    |      train_rate |   23101.78 |   29161.20
speech_transformer   |      train_rate |     172.26 |     219.64
squeezenet1_1        |      train_rate |     215.57 |     216.66
stargan              |      train_rate |     726.28 |     637.50
super_slomo          |      train_rate |      58.39 |      46.47
td3                  |      train_rate |   20150.23 |   26453.40
vit_l_32             |      train_rate |        nan |     550.16
"""

@pytest.mark.skipif(skip_tests, reason="result data missing")
def test_compare(capsys):
    milabench_cmd("compare", resultfolder())
    
    captured = capsys.readouterr()
    assert captured.out in _output
    assert captured.err == ""


@pytest.mark.skipif(skip_tests, reason="result data missing")
def test_summary(capsys):
    milabench_cmd("summary", resultfolder() + "/sokilupa.2023-03-09_16:02:53.984696")
    
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    
    assert data['squeezenet1_1']['name'] == 'squeezenet1_1'
    assert data['squeezenet1_1']['train_rate']['median'] == 216.66385215244662


@pytest.mark.skipif(skip_tests, reason="result data missing")
def test_report_folder_does_average():
    milabench_cmd(
        "report", 
        "--runs", resultfolder()
    )


@pytest.mark.skipif(skip_tests, reason="result data missing")
def test_report_one_run():
    milabench_cmd(
        "report", 
        "--runs", resultfolder() + "/sokilupa.2023-03-09_16:02:53.984696"
    )
