import os

from milabench.dashboard.live import ReportMachinePerf
from milabench.log import LongDashFormatter
from milabench.testing import replay_zipfile


def test_dash(runs_folder):
    scenario = os.path.join(runs_folder, "8GPUs.zip")

    formatter = LongDashFormatter()
    formatter.prune_delay = 0

    replay_zipfile(scenario, formatter, sleep=0)


def test_live_report(runs_folder):
    scenario = os.path.join(runs_folder, "8GPUs.zip")

    formatter = ReportMachinePerf()
    # formatter.prune_delay = 0

    replay_zipfile(scenario, formatter, sleep=0)


def test_live_report(runs_folder):
    scenario = os.path.join(runs_folder, "8GPUs.zip")

    formatter = ReportMachinePerf()

    replay_zipfile(scenario, formatter, sleep=0)
