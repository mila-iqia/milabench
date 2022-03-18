import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from git import Repo


class XPath(type(Path())):
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

    def sub(self, pattern, replacement):
        """Replace parts of the file using a regular expression.

        Arguments:
            pattern: The pattern to replace.
            replacement: The replacement.
        """
        new_content = re.sub(pattern, replacement, self.read_text())
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

    def clone_subtree(self, remote, commit, subtree, dest=None):
        """Clone a subtree of a git repository and move it to the given destination.

        Arguments:
            remote: URI for the Git repository.
            commit: The branch or commit to pull from.
            subtree: Path to the subtree to copy.
            dest: Path to which the subtree should be moved (must not already exist).
        """
        if dest is None:
            dest = Path(subtree).name
        dest = self / dest

        if dest.exists():
            raise shutil.Error(f"Destination path {dest} already exists")

        tmp = tempfile.mkdtemp()
        repo = Repo.init(tmp)
        origin = repo.create_remote("origin", remote)
        origin.fetch(commit, depth=1)
        repo.git.sparse_checkout("init")
        repo.git.sparse_checkout("set", subtree)
        repo.git.checkout(commit)

        path = Path(tmp) / subtree
        if not path.is_dir():
            dest.parent.mkdir(exist_ok=True)

        shutil.move(path, dest)
        shutil.rmtree(tmp)
        return dest

    def rm(self):
        """Remove this file or directory tree."""
        if self.is_dir():
            shutil.rmtree(self)
        else:
            self.unlink()
