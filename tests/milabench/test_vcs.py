import os
import subprocess
from unittest.mock import MagicMock, mock_open, patch

import pytest

from milabench.scripts.vcs import (
    _exec,
    find_pr_url,
    git_branch,
    github_repo_from_remote,
    read_previous,
    retrieve_git_versions,
    update_version_file,
    version_file,
)


# ---------------------------------------------------------------------------
# _exec
# ---------------------------------------------------------------------------

class TestExec:
    def test_successful_command(self):
        with patch("subprocess.check_output", return_value="  v1.0  ") as mock:
            result = _exec("git describe --always --tags", "<tag>")
            assert result == "v1.0"
            mock.assert_called_once()

    def test_called_process_error_returns_default(self):
        with patch(
            "subprocess.check_output",
            side_effect=subprocess.CalledProcessError(1, "git"),
        ):
            with pytest.warns(UserWarning, match="out of tree"):
                result = _exec("git describe --always --tags", "<tag>")
            assert result == "<tag>"


# ---------------------------------------------------------------------------
# retrieve_git_versions
# ---------------------------------------------------------------------------

class TestRetrieveGitVersions:
    def test_defaults(self):
        with patch("milabench.scripts.vcs._exec", side_effect=lambda cmd, d: d):
            info = retrieve_git_versions()
        assert info["tag"] == "<tag>"
        assert info["commit"] == "<commit>"
        assert info["date"] == "<date>"
        assert info["branch"] is None
        assert info["repo"] is None
        assert info["pr_url"] is None

    def test_include_pr(self):
        with (
            patch("milabench.scripts.vcs._exec", side_effect=lambda cmd, d: d),
            patch("milabench.scripts.vcs.git_branch", return_value="feature-x"),
            patch(
                "milabench.scripts.vcs.github_repo_from_remote",
                return_value="owner/repo",
            ),
            patch("milabench.scripts.vcs.find_pr_url", return_value="https://github.com/owner/repo/pull/42"),
        ):
            info = retrieve_git_versions(include_pr=True)
        assert info["branch"] == "feature-x"
        assert info["repo"] == "owner/repo"
        assert info["pr_url"] == "https://github.com/owner/repo/pull/42"

    def test_include_pr_uses_env_repo(self):
        with (
            patch("milabench.scripts.vcs._exec", side_effect=lambda cmd, d: d),
            patch("milabench.scripts.vcs.git_branch", return_value="dev"),
            patch.dict(os.environ, {"GITHUB_REPOSITORY": "env/repo"}, clear=False),
            patch("milabench.scripts.vcs.find_pr_url", return_value=None),
        ):
            info = retrieve_git_versions(include_pr=True)
        assert info["repo"] == "env/repo"


# ---------------------------------------------------------------------------
# github_repo_from_remote
# ---------------------------------------------------------------------------

class TestGithubRepoFromRemote:
    def test_ssh_remote(self):
        with patch("milabench.scripts.vcs._exec", return_value="git@github.com:owner/repo.git"):
            assert github_repo_from_remote() == "owner/repo"

    def test_https_remote(self):
        with patch("milabench.scripts.vcs._exec", return_value="https://github.com/owner/repo.git"):
            assert github_repo_from_remote() == "owner/repo"

    def test_https_remote_no_git_suffix(self):
        with patch("milabench.scripts.vcs._exec", return_value="https://github.com/owner/repo"):
            assert github_repo_from_remote() == "owner/repo"

    def test_no_remote_returns_none(self):
        with patch("milabench.scripts.vcs._exec", return_value=None):
            assert github_repo_from_remote() is None

    def test_non_github_remote_returns_none(self):
        with patch("milabench.scripts.vcs._exec", return_value="https://gitlab.com/owner/repo.git"):
            assert github_repo_from_remote() is None


# ---------------------------------------------------------------------------
# git_branch
# ---------------------------------------------------------------------------

class TestGitBranch:
    def test_from_github_head_ref(self):
        with patch.dict(os.environ, {"GITHUB_HEAD_REF": "pr-branch"}, clear=False):
            assert git_branch() == "pr-branch"

    def test_from_github_ref_name(self):
        env = {"GITHUB_REF_NAME": "main"}
        with (
            patch.dict(os.environ, env, clear=False),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("GITHUB_HEAD_REF", None)
            assert git_branch() == "main"

    def test_fallback_to_git(self):
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("milabench.scripts.vcs._exec", return_value="develop"),
        ):
            os.environ.pop("GITHUB_HEAD_REF", None)
            os.environ.pop("GITHUB_REF_NAME", None)
            assert git_branch() == "develop"


# ---------------------------------------------------------------------------
# find_pr_url
# ---------------------------------------------------------------------------

class TestFindPrUrl:
    def test_from_pr_number_env(self):
        env = {"GITHUB_PR_NUMBER": "99", "GITHUB_REPOSITORY": "org/proj"}
        with patch.dict(os.environ, env, clear=False):
            url = find_pr_url("org/proj", "feat")
            assert url == "https://github.com/org/proj/pull/99"

    def test_from_github_ref_pull(self):
        env = {
            "GITHUB_REF": "refs/pull/42/merge",
            "GITHUB_REPOSITORY": "org/proj",
        }
        with (
            patch.dict(os.environ, env, clear=False),
        ):
            os.environ.pop("GITHUB_PR_NUMBER", None)
            url = find_pr_url("org/proj", "feat")
            assert url == "https://github.com/org/proj/pull/42"

    def test_custom_server_url(self):
        env = {
            "GITHUB_PR_NUMBER": "10",
            "GITHUB_SERVER_URL": "https://ghe.corp.com",
            "GITHUB_REPOSITORY": "team/app",
        }
        with patch.dict(os.environ, env, clear=False):
            url = find_pr_url("team/app", "fix")
            assert url == "https://ghe.corp.com/team/app/pull/10"

    def test_no_repo_returns_none(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GITHUB_PR_NUMBER", None)
            os.environ.pop("GITHUB_REF", None)
            os.environ.pop("GITHUB_REPOSITORY", None)
            assert find_pr_url(None, "feat") is None

    def test_branch_main_returns_none(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GITHUB_PR_NUMBER", None)
            os.environ.pop("GITHUB_REF", None)
            os.environ.pop("GITHUB_REPOSITORY", None)
            assert find_pr_url("owner/repo", "main") is None

    def test_branch_master_returns_none(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GITHUB_PR_NUMBER", None)
            os.environ.pop("GITHUB_REF", None)
            os.environ.pop("GITHUB_REPOSITORY", None)
            assert find_pr_url("owner/repo", "master") is None

    def test_branch_head_returns_none(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GITHUB_PR_NUMBER", None)
            os.environ.pop("GITHUB_REF", None)
            os.environ.pop("GITHUB_REPOSITORY", None)
            assert find_pr_url("owner/repo", "HEAD") is None

    def test_api_lookup_success(self):
        import json
        pr_data = [{"html_url": "https://github.com/owner/repo/pull/7"}]
        resp_mock = MagicMock()
        resp_mock.read.return_value = json.dumps(pr_data).encode()
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("urllib.request.urlopen", return_value=resp_mock),
        ):
            os.environ.pop("GITHUB_PR_NUMBER", None)
            os.environ.pop("GITHUB_REF", None)
            os.environ.pop("GITHUB_REPOSITORY", None)
            os.environ.pop("GITHUB_TOKEN", None)
            os.environ.pop("GH_TOKEN", None)
            url = find_pr_url("owner/repo", "feature")
        assert url == "https://github.com/owner/repo/pull/7"

    def test_api_lookup_with_token(self):
        import json
        pr_data = [{"html_url": "https://github.com/owner/repo/pull/8"}]
        resp_mock = MagicMock()
        resp_mock.read.return_value = json.dumps(pr_data).encode()
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)

        with (
            patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_secret"}, clear=False),
            patch("urllib.request.urlopen", return_value=resp_mock),
            patch("urllib.request.Request") as req_cls,
        ):
            os.environ.pop("GITHUB_PR_NUMBER", None)
            os.environ.pop("GITHUB_REF", None)
            os.environ.pop("GITHUB_REPOSITORY", None)
            req_instance = MagicMock()
            req_cls.return_value = req_instance
            url = find_pr_url("owner/repo", "feature")
            req_instance.add_header.assert_called_once_with(
                "Authorization", "Bearer ghp_secret"
            )
        assert url == "https://github.com/owner/repo/pull/8"

    def test_api_lookup_empty_pulls(self):
        import json
        resp_mock = MagicMock()
        resp_mock.read.return_value = json.dumps([]).encode()
        resp_mock.__enter__ = lambda s: s
        resp_mock.__exit__ = MagicMock(return_value=False)

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("urllib.request.urlopen", return_value=resp_mock),
        ):
            os.environ.pop("GITHUB_PR_NUMBER", None)
            os.environ.pop("GITHUB_REF", None)
            os.environ.pop("GITHUB_REPOSITORY", None)
            os.environ.pop("GITHUB_TOKEN", None)
            os.environ.pop("GH_TOKEN", None)
            url = find_pr_url("owner/repo", "feature")
        assert url is None

    def test_api_lookup_exception_returns_none(self):
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("urllib.request.urlopen", side_effect=Exception("network error")),
        ):
            os.environ.pop("GITHUB_PR_NUMBER", None)
            os.environ.pop("GITHUB_REF", None)
            os.environ.pop("GITHUB_REPOSITORY", None)
            os.environ.pop("GITHUB_TOKEN", None)
            os.environ.pop("GH_TOKEN", None)
            url = find_pr_url("owner/repo", "feature")
        assert url is None

    def test_no_branch_returns_none(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GITHUB_PR_NUMBER", None)
            os.environ.pop("GITHUB_REF", None)
            os.environ.pop("GITHUB_REPOSITORY", None)
            assert find_pr_url("owner/repo", None) is None


# ---------------------------------------------------------------------------
# version_file
# ---------------------------------------------------------------------------

class TestVersionFile:
    def test_returns_path(self):
        vf = version_file()
        assert vf.endswith(os.path.join("milabench", "_version.py"))


# ---------------------------------------------------------------------------
# read_previous
# ---------------------------------------------------------------------------

class TestReadPrevious:
    def test_missing_file_returns_defaults(self, tmp_path):
        fake_path = str(tmp_path / "nonexistent" / "_version.py")
        with patch("milabench.scripts.vcs.version_file", return_value=fake_path):
            info = read_previous()
        assert info == ["<tag>", "<commit>", "<date>"]

    def test_reads_existing_file(self, tmp_path):
        vfile = tmp_path / "_version.py"
        vfile.write_text(
            '"""generated"""\n\n'
            '__tag__ = "v0.5.0"\n'
            '__commit__ = "abc123"\n'
            '__date__ = "2025-01-01"\n'
        )
        with patch("milabench.scripts.vcs.version_file", return_value=str(vfile)):
            info = read_previous()
        assert info[0] == "v0.5.0"
        assert info[1] == "abc123"
        assert info[2] == "2025-01-01"

    def test_partial_file(self, tmp_path):
        vfile = tmp_path / "_version.py"
        vfile.write_text('__tag__ = "v1.0"\n')
        with patch("milabench.scripts.vcs.version_file", return_value=str(vfile)):
            info = read_previous()
        assert info[0] == "v1.0"
        assert info[1] == "<commit>"
        assert info[2] == "<date>"


# ---------------------------------------------------------------------------
# update_version_file
# ---------------------------------------------------------------------------

class TestUpdateVersionFile:
    def test_writes_version_file(self, tmp_path):
        vfile = tmp_path / "_version.py"
        git_info = {
            "tag": "v2.0",
            "commit": "deadbeef",
            "date": "2025-06-01",
            "branch": None,
            "repo": None,
            "pr_url": None,
        }
        with (
            patch("milabench.scripts.vcs.version_file", return_value=str(vfile)),
            patch("milabench.scripts.vcs.retrieve_git_versions", return_value=git_info),
            patch("milabench.scripts.vcs.read_previous", return_value=["<tag>", "<commit>", "<date>"]),
        ):
            update_version_file()

        content = vfile.read_text()
        assert '"""This file is generated, do not modify"""' in content
        assert '__tag__ = "v2.0"' in content
        assert '__commit__ = "deadbeef"' in content
        assert '__date__ = "2025-06-01"' in content
        assert '__branch__ = "None"' in content
