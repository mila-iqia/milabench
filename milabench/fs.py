import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import pathspec
import requests


def manifest_ignorer(manifest):
    with open(manifest, "r") as mf:
        spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, mf)

    def ignore(directory, files):
        n = len(directory)
        file_paths = [str(XPath(directory) / file) for file in files]
        return [p[n + 1 :] for p in file_paths if not spec.match_file(p)]

    return ignore


def _restat(file, orperm=0o000, andperm=0o777):
    stat = os.stat(file)
    new_status = (stat.st_mode | orperm) & andperm
    os.chmod(file, new_status)


class XPath(type(Path())):
    def __len__(self):
        return len(str(self))

    def open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if encoding is None and "b" not in mode:
            encoding = "utf-8"
        return super().open(mode, buffering, encoding, errors, newline)

    def prepend_lines(self, *lines):
        """Prepend lines to a file.

        Arguments:
            lines: The lines to prepend, in order.
        """
        lines = "\n".join([*lines, ""])
        new_content = lines + self.read_text()
        self.write_text(new_content)

    def append_lines(self, *lines):
        """Append lines to a file.

        Arguments:
            lines: The lines to prepend, in order.
        """
        lines = "\n".join([*lines, ""])
        new_content = self.read_text() + lines
        self.write_text(new_content)

    def copy(self, path):
        path = XPath(path)
        if not path.exists():
            os.makedirs(path.parent, exist_ok=True)
        if self.is_dir():
            shutil.copytree(self, path)
        else:
            shutil.copy(self, path)

    def merge_into(self, dest, manifest="*", move=False, readonly=False):
        if isinstance(manifest, Path):
            if manifest.exists():
                manifest = manifest.open().read()
            else:
                manifest = ""
        manifest = manifest.splitlines()
        spec = pathspec.PathSpec.from_lines("gitwildmatch", manifest)
        for rel_file in spec.match_tree(self):
            filepath = self / rel_file
            if filepath.is_dir():
                os.makedirs(self, exist_ok=True)
            else:
                destpath = dest / rel_file
                if not destpath.parent.exists():
                    os.makedirs(destpath.parent, exist_ok=True)
                if destpath.exists() and readonly:
                    _restat(destpath, orperm=0o200)
                if move:
                    shutil.move(filepath, destpath)
                else:
                    shutil.copy2(filepath, destpath)
                if readonly:
                    _restat(destpath, andperm=0o555)

    def sub(self, pattern, replacement):
        """Replace parts of the file using a regular expression.

        Arguments:
            pattern: The pattern to replace.
            replacement: The replacement.
        """
        new_content = re.sub(pattern, str(replacement), self.read_text())
        self.write_text(new_content)

    def download(self, url, dest=None):
        """Download a file.

        Arguments:
            url: The URL to download from.
            dest: The path to download into.
        """
        if dest is None:
            dest = urlparse(url).path.split("/")[-1]
        dest = self / dest
        r = requests.get(url, stream=True)
        total = int(r.headers.get("content-length") or "1024")
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=max(8192, total // 100)):
                f.write(chunk)
                f.flush()

    def clone_subtree(self, remote, commit, subtree=None, dest=None):
        """Clone a subtree of a git repository and move it to the given destination.

        Arguments:
            remote: URI for the Git repository.
            commit: The branch or commit to pull from.
            subtree: Path to the subtree to copy.
            dest: Path to which the subtree should be moved (must not already exist).
        """
        from git import GitCommandError, GitCommandNotFound, Repo
        
        if dest is None:
            dest = self
        elif not dest.is_absolute():
            dest = self / dest

        tmp = tempfile.mkdtemp()
        repo = Repo.init(tmp)
        origin = repo.create_remote("origin", remote)
        origin.fetch(commit, depth=1)
        if subtree is not None:
            try:
                repo.git.sparse_checkout("init")
                repo.git.sparse_checkout("set", subtree)
            except (GitCommandNotFound, GitCommandError):
                print("git sparse-checkout not available; doing a full checkout")
        repo.git.checkout(commit)

        if subtree:
            path = XPath(tmp) / subtree
        else:
            path = XPath(tmp)
        shutil.copytree(path, dest, dirs_exist_ok=True)
        shutil.rmtree(tmp)
        return dest

    def rm(self):
        """Remove this file or directory tree."""
        if not self.exists():
            return
        elif self.is_dir():
            shutil.rmtree(self)
        else:
            self.unlink()
