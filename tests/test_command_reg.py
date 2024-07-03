import os
from milabench.cli.dry import cli_dry, Arguments

from pytest import fixture

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def compare(base, capsys, file_regression):
    all = capsys.readouterr()
    stdout = all.out
    assert stdout != ""

    stdout = stdout.replace(str(base), "$BASE")
    stdout = stdout.replace(str(ROOT), "$SRC")
    file_regression.check(stdout)


@fixture
def set_reg_env(monkeypatch, standard_config, tmp_path):
    monkeypatch.setenv("MILABENCH_CONFIG", str(standard_config))
    monkeypatch.setenv("MILABENCH_BASE", str(tmp_path))


def test_command_reg_one_node(set_reg_env, tmp_path, capsys, file_regression):
    args = Arguments()
    args.ngpu = 8
    args.capacity = 80000
    args.nnodes = 1

    cli_dry(args)
    
    compare(str(tmp_path), capsys, file_regression)


def test_command_reg_two_nodes(set_reg_env, tmp_path, capsys, file_regression):
    args = Arguments()
    args.ngpu = 8
    args.capacity = 80000
    args.nnodes = 2
    
    cli_dry(args)

    compare(str(tmp_path), capsys, file_regression)