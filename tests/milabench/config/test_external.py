import os
import subprocess

import pytest

from milabench.config.external import resolve_extern_definition


class TestResolveExternDefinitionStringShortCircuit:
    """When pack['definition'] is already a string, the function returns early."""

    def test_string_definition_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "milabench.config.external.extern_location", str(tmp_path / "_extern")
        )
        pack = {"definition": "/some/local/path"}
        result = resolve_extern_definition(pack)
        assert result is None

    def test_string_definition_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "milabench.config.external.extern_location", str(tmp_path / "_extern")
        )
        pack = {"definition": "/some/local/path"}
        resolve_extern_definition(pack)
        assert pack["definition"] == "/some/local/path"

    def test_string_definition_creates_extern_dir(self, tmp_path, monkeypatch):
        extern = tmp_path / "_extern"
        monkeypatch.setattr(
            "milabench.config.external.extern_location", str(extern)
        )
        pack = {"definition": "anything"}
        resolve_extern_definition(pack)
        assert extern.is_dir()


class TestResolveExternDefinitionCloneSuccess:
    """When definition is a dict and the repo doesn't exist yet, git clone runs."""

    def test_clone_sets_definition_to_local_path(self, tmp_path, monkeypatch):
        extern = tmp_path / "_extern"
        monkeypatch.setattr(
            "milabench.config.external.extern_location", str(extern)
        )

        def fake_run(cmd, **kwargs):
            repo_name = "my-benchmarks"
            os.makedirs(os.path.join(str(extern), repo_name), exist_ok=True)
            return subprocess.CompletedProcess(cmd, returncode=0)

        monkeypatch.setattr("milabench.config.external.subprocess.run", fake_run)

        pack = {"definition": {"git": "https://github.com/user/my-benchmarks.git"}}
        resolve_extern_definition(pack)

        assert pack["definition"] == os.path.join(str(extern), "my-benchmarks")

    def test_clone_uses_default_branch_main(self, tmp_path, monkeypatch):
        extern = tmp_path / "_extern"
        monkeypatch.setattr(
            "milabench.config.external.extern_location", str(extern)
        )
        captured_cmds = []

        def fake_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            repo_name = "repo"
            os.makedirs(os.path.join(str(extern), repo_name), exist_ok=True)
            return subprocess.CompletedProcess(cmd, returncode=0)

        monkeypatch.setattr("milabench.config.external.subprocess.run", fake_run)

        pack = {"definition": {"git": "https://github.com/user/repo.git"}}
        resolve_extern_definition(pack)

        assert "-b" in captured_cmds[0]
        branch_idx = captured_cmds[0].index("-b")
        assert captured_cmds[0][branch_idx + 1] == "main"

    def test_clone_uses_custom_branch(self, tmp_path, monkeypatch):
        extern = tmp_path / "_extern"
        monkeypatch.setattr(
            "milabench.config.external.extern_location", str(extern)
        )
        captured_cmds = []

        def fake_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            repo_name = "repo"
            os.makedirs(os.path.join(str(extern), repo_name), exist_ok=True)
            return subprocess.CompletedProcess(cmd, returncode=0)

        monkeypatch.setattr("milabench.config.external.subprocess.run", fake_run)

        pack = {
            "definition": {
                "git": "https://github.com/user/repo.git",
                "branch": "develop",
            }
        }
        resolve_extern_definition(pack)

        branch_idx = captured_cmds[0].index("-b")
        assert captured_cmds[0][branch_idx + 1] == "develop"

    def test_clone_command_structure(self, tmp_path, monkeypatch):
        extern = tmp_path / "_extern"
        monkeypatch.setattr(
            "milabench.config.external.extern_location", str(extern)
        )
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            repo_name = "benchmarks"
            os.makedirs(os.path.join(str(extern), repo_name), exist_ok=True)
            return subprocess.CompletedProcess(cmd, returncode=0)

        monkeypatch.setattr("milabench.config.external.subprocess.run", fake_run)

        url = "https://github.com/org/benchmarks.git"
        pack = {"definition": {"git": url}}
        resolve_extern_definition(pack)

        cmd = captured["cmd"]
        assert cmd[0] == "git"
        assert cmd[1] == "clone"
        assert "--recurse-submodules" in cmd
        assert "-j8" in cmd
        assert url in cmd
        assert captured["kwargs"]["cwd"] == str(extern)
        assert captured["kwargs"]["text"] is True

    def test_clone_url_without_git_extension(self, tmp_path, monkeypatch):
        extern = tmp_path / "_extern"
        monkeypatch.setattr(
            "milabench.config.external.extern_location", str(extern)
        )

        def fake_run(cmd, **kwargs):
            os.makedirs(os.path.join(str(extern), "my-repo"), exist_ok=True)
            return subprocess.CompletedProcess(cmd, returncode=0)

        monkeypatch.setattr("milabench.config.external.subprocess.run", fake_run)

        pack = {"definition": {"git": "https://github.com/user/my-repo"}}
        resolve_extern_definition(pack)

        assert pack["definition"] == os.path.join(str(extern), "my-repo")


class TestResolveExternDefinitionCloneFailure:
    """When git clone returns non-zero, a RuntimeError is raised."""

    def test_raises_runtime_error_on_clone_failure(self, tmp_path, monkeypatch):
        extern = tmp_path / "_extern"
        monkeypatch.setattr(
            "milabench.config.external.extern_location", str(extern)
        )

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, returncode=128)

        monkeypatch.setattr("milabench.config.external.subprocess.run", fake_run)

        pack = {"definition": {"git": "https://github.com/user/repo.git"}}
        with pytest.raises(RuntimeError, match="Could not resolve definition"):
            resolve_extern_definition(pack)

    def test_definition_unchanged_on_failure(self, tmp_path, monkeypatch):
        extern = tmp_path / "_extern"
        monkeypatch.setattr(
            "milabench.config.external.extern_location", str(extern)
        )

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, returncode=1)

        monkeypatch.setattr("milabench.config.external.subprocess.run", fake_run)

        original_def = {"git": "https://github.com/user/repo.git"}
        pack = {"definition": original_def}
        with pytest.raises(RuntimeError):
            resolve_extern_definition(pack)
        assert pack["definition"] is original_def


class TestResolveExternDefinitionPathAlreadyExists:
    """When the repo directory already exists, git clone is skipped."""

    def test_skips_clone_when_path_exists(self, tmp_path, monkeypatch):
        extern = tmp_path / "_extern"
        extern.mkdir()
        (extern / "repo").mkdir()
        monkeypatch.setattr(
            "milabench.config.external.extern_location", str(extern)
        )
        clone_called = []

        def fake_run(cmd, **kwargs):
            clone_called.append(True)
            return subprocess.CompletedProcess(cmd, returncode=0)

        monkeypatch.setattr("milabench.config.external.subprocess.run", fake_run)

        pack = {"definition": {"git": "https://github.com/user/repo.git"}}
        resolve_extern_definition(pack)

        assert len(clone_called) == 0
        assert pack["definition"] == os.path.join(str(extern), "repo")

    def test_sets_definition_path_when_already_cloned(self, tmp_path, monkeypatch):
        extern = tmp_path / "_extern"
        extern.mkdir()
        (extern / "existing-bench").mkdir()
        monkeypatch.setattr(
            "milabench.config.external.extern_location", str(extern)
        )

        monkeypatch.setattr(
            "milabench.config.external.subprocess.run",
            lambda cmd, **kw: (_ for _ in ()).throw(AssertionError("should not clone")),
        )

        pack = {
            "definition": {"git": "https://github.com/org/existing-bench.git"}
        }
        resolve_extern_definition(pack)

        expected = os.path.join(str(extern), "existing-bench")
        assert pack["definition"] == expected


class TestExternLocationCreation:
    """The extern_location directory is always created if it doesn't exist."""

    def test_creates_nested_extern_dir(self, tmp_path, monkeypatch):
        extern = tmp_path / "deep" / "nested" / "_extern"
        monkeypatch.setattr(
            "milabench.config.external.extern_location", str(extern)
        )
        pack = {"definition": "a-string"}
        resolve_extern_definition(pack)
        assert extern.is_dir()

    def test_idempotent_when_extern_dir_exists(self, tmp_path, monkeypatch):
        extern = tmp_path / "_extern"
        extern.mkdir()
        monkeypatch.setattr(
            "milabench.config.external.extern_location", str(extern)
        )
        pack = {"definition": "a-string"}
        resolve_extern_definition(pack)
        assert extern.is_dir()
