"""Use to retrieve GIT version info, this file cannot import milabench modules
as it is executed as part of the installation process"""

import os
import subprocess
import warnings

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def _exec(cmd, default):
    try:
        return subprocess.check_output(
            cmd.split(" "), encoding="utf-8", cwd=ROOT
        ).strip()
    except subprocess.CalledProcessError:
        warnings.warn("out of tree; milabench could not retrieve version info")
        return default


def retrieve_git_versions(tag="<tag>", commit="<commit>", date="<date>", include_pr=False):
    branch = None
    repo = None
    pr_url = None

    if include_pr:
        branch = git_branch()
        repo = os.environ.get("GITHUB_REPOSITORY") or github_repo_from_remote()
        pr_url = find_pr_url(repo, branch)

    return {
        "tag": _exec("git describe --always --tags", tag),
        "commit": _exec("git rev-parse HEAD", commit),
        "date": _exec("git show -s --format=%ci", date),
        "branch": branch,
        "repo": repo,
        "pr_url": pr_url,
    }


def github_repo_from_remote():
    """Extract GitHub owner/repo from the git remote URL."""
    remote = _exec("git remote get-url origin", None)
    if not remote:
        return None

    if remote.startswith("git@github.com:"):
        return remote.split(":", 1)[1].removesuffix(".git")

    if "github.com/" in remote:
        path = remote.split("github.com/", 1)[1]
        return path.removesuffix(".git")

    return None


def git_branch():
    """Return the current branch name."""
    return os.environ.get("GITHUB_HEAD_REF") or os.environ.get(
        "GITHUB_REF_NAME", _exec("git rev-parse --abbrev-ref HEAD", None)
    )


def find_pr_url(repo, branch):
    """Try to find a PR URL from GitHub CI env vars, or by querying the GitHub API."""
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")

    pr_number = os.environ.get("GITHUB_PR_NUMBER")
    if not pr_number:
        github_ref = os.environ.get("GITHUB_REF", "")
        if github_ref.startswith("refs/pull/"):
            pr_number = github_ref.split("/")[2]

    ci_repo = os.environ.get("GITHUB_REPOSITORY", repo)
    if ci_repo and pr_number:
        return f"{server}/{ci_repo}/pull/{pr_number}"

    if not repo or not branch or branch in ("main", "master", "HEAD"):
        return None

    try:
        import urllib.request
        import json

        url = f"https://api.github.com/repos/{repo}/pulls?head={repo.split('/')[0]}:{branch}&state=all&per_page=1"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})

        pat = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if pat:
            req.add_header("Authorization", f"Bearer {pat}")

        with urllib.request.urlopen(req, timeout=5) as resp:
            pulls = json.loads(resp.read())
            if pulls:
                return pulls[0]["html_url"]
    except Exception:
        pass

    return None


def version_file():
    return os.path.join(ROOT, "milabench", "_version.py")

def read_previous():
    info = ["<tag>", "<commit>", "<date>"]
    
    if not os.path.exists(version_file()):
        return info
    
    with open(version_file(), "r") as file:
        for line in file.readlines():
            if "tag" in line:
                _, v = line.split("=")
                info[0] = v.strip().strip('"')

            if "commit" in line:
                _, v = line.split("=")
                info[1] = v.strip().strip('"')

            if "date" in line:
                _, v = line.split("=")
                info[2] = v.strip().strip('"')

    return info


def update_version_file():
    version_info = retrieve_git_versions(*read_previous())

    with open(version_file(), "w") as file:
        file.write('"""')
        file.write("This file is generated, do not modify")
        file.write('"""\n\n')

        for key, data in version_info.items():
            file.write(f'__{key}__ = "{data}"\n')


if __name__ == "__main__":
    update_version_file()
