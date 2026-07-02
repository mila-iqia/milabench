import os
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from milabench.fs import XPath, _restat, manifest_ignorer


# ---------------------------------------------------------------------------
# _restat
# ---------------------------------------------------------------------------

class TestRestat:
    def test_restat_sets_or_permission(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        os.chmod(f, 0o644)
        _restat(f, orperm=0o100)
        mode = stat.S_IMODE(os.stat(f).st_mode)
        assert mode & 0o100

    def test_restat_clears_with_andperm(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        os.chmod(f, 0o755)
        _restat(f, andperm=0o555)
        mode = stat.S_IMODE(os.stat(f).st_mode)
        assert not (mode & 0o200)

    def test_restat_combined(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        os.chmod(f, 0o600)
        _restat(f, orperm=0o044, andperm=0o644)
        mode = stat.S_IMODE(os.stat(f).st_mode)
        assert mode == 0o644


# ---------------------------------------------------------------------------
# manifest_ignorer
# ---------------------------------------------------------------------------

class TestManifestIgnorer:
    def test_ignorer_filters_files(self, tmp_path):
        manifest = tmp_path / ".manifest"
        manifest.write_text("*.py\n")

        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("")
        (src / "data.csv").write_text("")
        (src / "readme.md").write_text("")

        ignore_fn = manifest_ignorer(str(manifest))
        ignored = ignore_fn(str(src), ["main.py", "data.csv", "readme.md"])
        assert "main.py" not in ignored
        assert "data.csv" in ignored
        assert "readme.md" in ignored

    def test_ignorer_wildcard_all(self, tmp_path):
        manifest = tmp_path / ".manifest"
        manifest.write_text("*\n")

        ignore_fn = manifest_ignorer(str(manifest))
        ignored = ignore_fn("/some/dir", ["a.txt", "b.py"])
        assert ignored == []

    def test_ignorer_empty_manifest(self, tmp_path):
        manifest = tmp_path / ".manifest"
        manifest.write_text("")

        ignore_fn = manifest_ignorer(str(manifest))
        ignored = ignore_fn("/dir", ["a.txt", "b.py"])
        assert set(ignored) == {"a.txt", "b.py"}


# ---------------------------------------------------------------------------
# XPath basics
# ---------------------------------------------------------------------------

class TestXPathBasics:
    def test_len(self, tmp_path):
        p = XPath(tmp_path / "hello")
        assert len(p) == len(str(p))

    def test_len_root(self):
        p = XPath("/")
        assert len(p) == 1

    def test_is_path_subclass(self):
        p = XPath("/tmp")
        assert isinstance(p, Path)


# ---------------------------------------------------------------------------
# XPath.open
# ---------------------------------------------------------------------------

class TestXPathOpen:
    def test_open_default_encoding_utf8(self, tmp_path):
        f = XPath(tmp_path / "utf8.txt")
        f.write_text("héllo")
        with f.open("r") as fh:
            assert fh.encoding == "utf-8"
            assert fh.read() == "héllo"

    def test_open_binary_mode_no_encoding(self, tmp_path):
        f = XPath(tmp_path / "bin.dat")
        f.write_bytes(b"\x00\x01")
        with f.open("rb") as fh:
            assert fh.read() == b"\x00\x01"

    def test_open_explicit_encoding(self, tmp_path):
        f = XPath(tmp_path / "latin.txt")
        f.write_text("café", encoding="latin-1")
        with f.open("r", encoding="latin-1") as fh:
            assert fh.encoding == "latin-1"
            assert fh.read() == "café"

    def test_open_write_mode(self, tmp_path):
        f = XPath(tmp_path / "out.txt")
        with f.open("w") as fh:
            fh.write("written")
        assert f.read_text() == "written"


# ---------------------------------------------------------------------------
# XPath.prepend_lines / append_lines
# ---------------------------------------------------------------------------

class TestXPathPrependAppend:
    def test_prepend_lines(self, tmp_path):
        f = XPath(tmp_path / "file.txt")
        f.write_text("original")
        f.prepend_lines("first", "second")
        content = f.read_text()
        assert content.startswith("first\nsecond\n")
        assert content.endswith("original")

    def test_prepend_lines_empty_file(self, tmp_path):
        f = XPath(tmp_path / "empty.txt")
        f.write_text("")
        f.prepend_lines("line1")
        assert f.read_text() == "line1\n"

    def test_append_lines(self, tmp_path):
        f = XPath(tmp_path / "file.txt")
        f.write_text("original")
        f.append_lines("third", "fourth")
        content = f.read_text()
        assert content.startswith("original")
        assert content.endswith("third\nfourth\n")

    def test_append_lines_empty_file(self, tmp_path):
        f = XPath(tmp_path / "empty.txt")
        f.write_text("")
        f.append_lines("added")
        assert f.read_text() == "added\n"

    def test_prepend_then_append(self, tmp_path):
        f = XPath(tmp_path / "file.txt")
        f.write_text("middle")
        f.prepend_lines("top")
        f.append_lines("bottom")
        assert f.read_text() == "top\nmiddle" + "bottom\n"


# ---------------------------------------------------------------------------
# XPath.copy
# ---------------------------------------------------------------------------

class TestXPathCopy:
    def test_copy_file(self, tmp_path):
        src = XPath(tmp_path / "src.txt")
        src.write_text("data")
        dest = tmp_path / "dest.txt"
        src.copy(dest)
        assert Path(dest).read_text() == "data"

    def test_copy_file_creates_parent_dirs(self, tmp_path):
        src = XPath(tmp_path / "src.txt")
        src.write_text("data")
        dest = tmp_path / "a" / "b" / "dest.txt"
        src.copy(dest)
        assert Path(dest).read_text() == "data"

    def test_copy_directory(self, tmp_path):
        src_dir = XPath(tmp_path / "srcdir")
        src_dir.mkdir()
        (src_dir / "child.txt").write_text("child")
        dest = tmp_path / "destdir"
        src_dir.copy(dest)
        assert (dest / "child.txt").read_text() == "child"

    def test_copy_preserves_original(self, tmp_path):
        src = XPath(tmp_path / "orig.txt")
        src.write_text("keep")
        src.copy(tmp_path / "copy.txt")
        assert src.read_text() == "keep"


# ---------------------------------------------------------------------------
# XPath.sub
# ---------------------------------------------------------------------------

class TestXPathSub:
    def test_sub_replaces_pattern(self, tmp_path):
        f = XPath(tmp_path / "code.py")
        f.write_text("foo = 1\nbar = foo + 2")
        f.sub(r"foo", "baz")
        assert f.read_text() == "baz = 1\nbar = baz + 2"

    def test_sub_regex_groups(self, tmp_path):
        f = XPath(tmp_path / "ver.txt")
        f.write_text("version=1.2.3")
        f.sub(r"version=(\d+)\.(\d+)\.(\d+)", "version=9.9.9")
        assert f.read_text() == "version=9.9.9"

    def test_sub_no_match_unchanged(self, tmp_path):
        f = XPath(tmp_path / "noop.txt")
        f.write_text("hello world")
        f.sub(r"xyz", "abc")
        assert f.read_text() == "hello world"

    def test_sub_numeric_replacement(self, tmp_path):
        f = XPath(tmp_path / "num.txt")
        f.write_text("count=0")
        f.sub(r"\d+", 42)
        assert f.read_text() == "count=42"


# ---------------------------------------------------------------------------
# XPath.rm
# ---------------------------------------------------------------------------

class TestXPathRm:
    def test_rm_file(self, tmp_path):
        f = XPath(tmp_path / "gone.txt")
        f.write_text("bye")
        assert f.exists()
        f.rm()
        assert not f.exists()

    def test_rm_directory(self, tmp_path):
        d = XPath(tmp_path / "gone_dir")
        d.mkdir()
        (d / "inner.txt").write_text("x")
        d.rm()
        assert not d.exists()

    def test_rm_nonexistent_is_noop(self, tmp_path):
        p = XPath(tmp_path / "nope")
        p.rm()
        assert not p.exists()

    def test_rm_nested_directory(self, tmp_path):
        d = XPath(tmp_path / "a" / "b" / "c")
        d.mkdir(parents=True)
        (d / "deep.txt").write_text("deep")
        top = XPath(tmp_path / "a")
        top.rm()
        assert not top.exists()


# ---------------------------------------------------------------------------
# XPath.merge_into
# ---------------------------------------------------------------------------

class TestXPathMergeInto:
    def test_merge_into_copies_matched_files(self, tmp_path):
        src = XPath(tmp_path / "src")
        src.mkdir()
        (src / "a.py").write_text("a")
        (src / "b.txt").write_text("b")
        dest = XPath(tmp_path / "dest")
        dest.mkdir()

        src.merge_into(dest, manifest="*.py")
        assert (dest / "a.py").read_text() == "a"
        assert not (dest / "b.txt").exists()

    def test_merge_into_wildcard(self, tmp_path):
        src = XPath(tmp_path / "src")
        src.mkdir()
        (src / "x.txt").write_text("x")
        (src / "y.txt").write_text("y")
        dest = XPath(tmp_path / "dest")
        dest.mkdir()

        src.merge_into(dest, manifest="*")
        assert (dest / "x.txt").exists()
        assert (dest / "y.txt").exists()

    def test_merge_into_move(self, tmp_path):
        src = XPath(tmp_path / "src")
        src.mkdir()
        (src / "moveme.txt").write_text("moving")
        dest = XPath(tmp_path / "dest")
        dest.mkdir()

        src.merge_into(dest, manifest="*", move=True)
        assert (dest / "moveme.txt").read_text() == "moving"
        assert not (src / "moveme.txt").exists()

    def test_merge_into_readonly(self, tmp_path):
        src = XPath(tmp_path / "src")
        src.mkdir()
        (src / "ro.txt").write_text("readonly")
        dest = XPath(tmp_path / "dest")
        dest.mkdir()

        src.merge_into(dest, manifest="*", readonly=True)
        dest_file = dest / "ro.txt"
        assert dest_file.exists()
        mode = stat.S_IMODE(os.stat(dest_file).st_mode)
        assert not (mode & 0o200)

    def test_merge_into_readonly_overwrites_existing(self, tmp_path):
        src = XPath(tmp_path / "src")
        src.mkdir()
        (src / "ro.txt").write_text("v2")
        dest = XPath(tmp_path / "dest")
        dest.mkdir()
        existing = dest / "ro.txt"
        existing.write_text("v1")
        os.chmod(existing, 0o444)

        src.merge_into(dest, manifest="*", readonly=True)
        assert (dest / "ro.txt").read_text() == "v2"

    def test_merge_into_creates_dest_subdirs(self, tmp_path):
        src = XPath(tmp_path / "src")
        sub = src / "sub"
        sub.mkdir(parents=True)
        (sub / "nested.txt").write_text("nested")
        dest = XPath(tmp_path / "dest")

        src.merge_into(dest, manifest="*")
        assert (dest / "sub" / "nested.txt").read_text() == "nested"

    def test_merge_into_manifest_as_path_exists(self, tmp_path):
        src = XPath(tmp_path / "src")
        src.mkdir()
        (src / "a.py").write_text("a")
        (src / "b.txt").write_text("b")
        dest = XPath(tmp_path / "dest")
        dest.mkdir()
        manifest = tmp_path / "manifest.txt"
        manifest.write_text("*.py\n")

        src.merge_into(dest, manifest=Path(manifest))
        assert (dest / "a.py").exists()
        assert not (dest / "b.txt").exists()

    def test_merge_into_manifest_as_path_missing(self, tmp_path):
        src = XPath(tmp_path / "src")
        src.mkdir()
        (src / "a.py").write_text("a")
        dest = XPath(tmp_path / "dest")
        dest.mkdir()
        missing_manifest = Path(tmp_path / "nonexistent.txt")

        src.merge_into(dest, manifest=missing_manifest)
        assert not (dest / "a.py").exists()

    def test_merge_into_with_directory_entry(self, tmp_path):
        src = XPath(tmp_path / "src")
        src.mkdir()
        subdir = src / "pkg"
        subdir.mkdir()
        (subdir / "mod.py").write_text("mod")
        dest = XPath(tmp_path / "dest")
        dest.mkdir()

        src.merge_into(dest, manifest="*")
        assert (dest / "pkg" / "mod.py").read_text() == "mod"


# ---------------------------------------------------------------------------
# XPath.download  (mock requests to avoid network)
# ---------------------------------------------------------------------------

class TestXPathDownload:
    def test_download_default_dest(self, tmp_path):
        d = XPath(tmp_path)
        chunk = b"file-content-here"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(chunk))}
        mock_response.iter_content.return_value = [chunk]

        with patch("milabench.fs.requests.get", return_value=mock_response):
            d.download("https://example.com/path/archive.tar.gz")

        result = tmp_path / "archive.tar.gz"
        assert result.exists()
        assert result.read_bytes() == chunk

    def test_download_explicit_dest(self, tmp_path):
        d = XPath(tmp_path)
        chunk = b"data"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(chunk))}
        mock_response.iter_content.return_value = [chunk]

        with patch("milabench.fs.requests.get", return_value=mock_response):
            d.download("https://example.com/file.bin", dest="custom.bin")

        assert (tmp_path / "custom.bin").read_bytes() == chunk

    def test_download_no_content_length(self, tmp_path):
        d = XPath(tmp_path)
        chunk = b"x" * 100
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.iter_content.return_value = [chunk]

        with patch("milabench.fs.requests.get", return_value=mock_response):
            d.download("https://example.com/nosize.dat")

        assert (tmp_path / "nosize.dat").read_bytes() == chunk

    def test_download_multiple_chunks(self, tmp_path):
        d = XPath(tmp_path)
        chunks = [b"aa", b"bb", b"cc"]
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "6"}
        mock_response.iter_content.return_value = chunks

        with patch("milabench.fs.requests.get", return_value=mock_response):
            d.download("https://example.com/multi.bin")

        assert (tmp_path / "multi.bin").read_bytes() == b"aabbcc"


# ---------------------------------------------------------------------------
# XPath.clone_subtree  (mock git to avoid real clones)
# ---------------------------------------------------------------------------

class TestXPathCloneSubtree:
    def _make_mock_repo(self, tmp_checkout, files):
        """Create real files inside a temp dir to simulate a checkout."""
        for name, content in files.items():
            p = Path(tmp_checkout) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

    @patch("git.Repo")
    @patch("tempfile.mkdtemp")
    def test_clone_subtree_no_subtree(self, mock_mkdtemp, mock_repo_cls, tmp_path):
        checkout_dir = tmp_path / "checkout"
        checkout_dir.mkdir()
        self._make_mock_repo(checkout_dir, {"file.txt": "root"})
        mock_mkdtemp.return_value = str(checkout_dir)

        mock_repo = MagicMock()
        mock_repo_cls.init.return_value = mock_repo

        dest = XPath(tmp_path / "dest")
        base = XPath(tmp_path / "base")
        base.mkdir()
        result = base.clone_subtree("https://github.com/x/y", "main", dest=dest)

        assert result == dest
        assert (dest / "file.txt").read_text() == "root"

    @patch("git.Repo")
    @patch("tempfile.mkdtemp")
    def test_clone_subtree_with_subtree(self, mock_mkdtemp, mock_repo_cls, tmp_path):
        checkout_dir = tmp_path / "checkout"
        checkout_dir.mkdir()
        self._make_mock_repo(checkout_dir, {"sub/inner.txt": "inner"})
        mock_mkdtemp.return_value = str(checkout_dir)

        mock_repo = MagicMock()
        mock_repo_cls.init.return_value = mock_repo

        dest = XPath(tmp_path / "dest")
        base = XPath(tmp_path / "base")
        base.mkdir()
        result = base.clone_subtree(
            "https://github.com/x/y", "main", subtree="sub", dest=dest
        )
        assert (dest / "inner.txt").read_text() == "inner"

    @patch("git.Repo")
    @patch("tempfile.mkdtemp")
    def test_clone_subtree_dest_none_uses_self(self, mock_mkdtemp, mock_repo_cls, tmp_path):
        checkout_dir = tmp_path / "checkout"
        checkout_dir.mkdir()
        self._make_mock_repo(checkout_dir, {"f.txt": "hi"})
        mock_mkdtemp.return_value = str(checkout_dir)

        mock_repo = MagicMock()
        mock_repo_cls.init.return_value = mock_repo

        base = XPath(tmp_path / "base")
        result = base.clone_subtree("https://github.com/x/y", "main")
        assert result == base
        assert (base / "f.txt").read_text() == "hi"

    @patch("git.Repo")
    @patch("tempfile.mkdtemp")
    def test_clone_subtree_relative_dest(self, mock_mkdtemp, mock_repo_cls, tmp_path):
        checkout_dir = tmp_path / "checkout"
        checkout_dir.mkdir()
        self._make_mock_repo(checkout_dir, {"r.txt": "rel"})
        mock_mkdtemp.return_value = str(checkout_dir)

        mock_repo = MagicMock()
        mock_repo_cls.init.return_value = mock_repo

        base = XPath(tmp_path / "base")
        base.mkdir()
        rel_dest = XPath("rel_out")
        result = base.clone_subtree("https://github.com/x/y", "main", dest=rel_dest)
        expected = base / "rel_out"
        assert result == expected

    @patch("git.Repo")
    @patch("tempfile.mkdtemp")
    def test_clone_subtree_sparse_checkout_fallback(self, mock_mkdtemp, mock_repo_cls, tmp_path):
        from git import GitCommandNotFound

        checkout_dir = tmp_path / "checkout"
        checkout_dir.mkdir()
        self._make_mock_repo(checkout_dir, {"sub/x.txt": "x"})
        mock_mkdtemp.return_value = str(checkout_dir)

        mock_repo = MagicMock()
        mock_repo.git.sparse_checkout.side_effect = GitCommandNotFound("sparse-checkout", "not found")
        mock_repo_cls.init.return_value = mock_repo

        base = XPath(tmp_path / "base")
        base.mkdir()
        dest = XPath(tmp_path / "dest")
        base.clone_subtree("https://github.com/x/y", "main", subtree="sub", dest=dest)
        assert (dest / "x.txt").read_text() == "x"
