"""Tests for milabench.dependencies.pin — pure logic and file-processing functions."""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.version import Version

from milabench.dependencies.pin import (
    _PinBlock,
    _compat_conditions_match,
    _extract_common_constraints,
    _normalize_backend_version,
    _parse_constraint_file,
    _strip_index_urls_from_constraint_file,
    constraint_filename,
    get_constraint_file,
)
from milabench.dependencies.platforms import (
    BackendConfig,
    CompatEntry,
    CompatRule,
    IndexConfig,
    PlatformConfig,
)
from milabench.dependencies.pin import (
    _build_constraints_content,
    _build_index_args,
    _resolve_compat_constraints,
)


# ---------------------------------------------------------------------------
# constraint_filename
# ---------------------------------------------------------------------------
class TestConstraintFilename:
    def test_cuda_with_arch(self):
        assert (
            constraint_filename("cuda", "130", "2.12.0", "x86_64")
            == "constraints.cuda130.torch2120.x86_64.txt"
        )

    def test_rocm_with_arch(self):
        assert (
            constraint_filename("rocm", "7.1", "2.10.0", "aarch64")
            == "constraints.rocm71.torch2100.aarch64.txt"
        )

    def test_cpu_no_backend_version(self):
        assert (
            constraint_filename("cpu", "", "2.12.0", "x86_64")
            == "constraints.cpu.torch2120.x86_64.txt"
        )

    def test_no_arch(self):
        assert (
            constraint_filename("cuda", "130", "2.12.0")
            == "constraints.cuda130.torch2120.txt"
        )

    def test_empty_arch_explicit(self):
        assert (
            constraint_filename("cuda", "126", "2.10.0", "")
            == "constraints.cuda126.torch2100.txt"
        )

    def test_four_part_torch(self):
        assert (
            constraint_filename("cuda", "130", "2.12.1", "x86_64")
            == "constraints.cuda130.torch2121.x86_64.txt"
        )


# ---------------------------------------------------------------------------
# get_constraint_file
# ---------------------------------------------------------------------------
class TestGetConstraintFile:
    def test_returns_path_under_pin_dir(self, tmp_path):
        result = get_constraint_file(tmp_path, "cuda", "130", "2.12.0")
        assert result == tmp_path / "constraints.cuda130.torch2120.txt"

    def test_with_arch(self, tmp_path):
        result = get_constraint_file(tmp_path, "rocm", "7.2", "2.11.0", "x86_64")
        assert result == tmp_path / "constraints.rocm72.torch2110.x86_64.txt"

    def test_is_path_object(self, tmp_path):
        result = get_constraint_file(tmp_path, "cpu", "", "2.10.0")
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# _normalize_backend_version
# ---------------------------------------------------------------------------
class TestNormalizeBackendVersion:
    def test_three_digit_compact(self):
        assert _normalize_backend_version("130") == "13.0"

    def test_three_digit_compact_126(self):
        assert _normalize_backend_version("126") == "12.6"

    def test_already_dotted(self):
        assert _normalize_backend_version("7.1") == "7.1"

    def test_already_dotted_long(self):
        assert _normalize_backend_version("13.0.1") == "13.0.1"

    def test_empty_string(self):
        assert _normalize_backend_version("") == ""

    def test_two_digit_compact(self):
        assert _normalize_backend_version("72") == "7.2"

    def test_single_digit(self):
        assert _normalize_backend_version("6") == ".6"


# ---------------------------------------------------------------------------
# _compat_conditions_match
# ---------------------------------------------------------------------------
class TestCompatConditionsMatch:
    def test_single_condition_match(self):
        known = {"torch": Version("2.12.0")}
        assert _compat_conditions_match("torch>=2.11", known) is True

    def test_single_condition_no_match(self):
        known = {"torch": Version("2.9.0")}
        assert _compat_conditions_match("torch>=2.11", known) is False

    def test_multiple_conditions_all_match(self):
        known = {"torch": Version("2.12.0"), "cuda": Version("13.0")}
        assert _compat_conditions_match("torch>=2.11,cuda>=13", known) is True

    def test_multiple_conditions_partial_match(self):
        known = {"torch": Version("2.12.0"), "cuda": Version("12.6")}
        assert _compat_conditions_match("torch>=2.11,cuda>=13", known) is False

    def test_missing_key(self):
        known = {"torch": Version("2.12.0")}
        assert _compat_conditions_match("torch>=2.11,cuda>=13", known) is False

    def test_empty_known(self):
        assert _compat_conditions_match("torch>=2.11", {}) is False

    def test_exact_version(self):
        known = {"torch": Version("2.11.0")}
        assert _compat_conditions_match("torch==2.11.0", known) is True

    def test_less_than(self):
        known = {"torch": Version("2.10.0")}
        assert _compat_conditions_match("torch<2.11", known) is True

    def test_invalid_condition_format(self):
        known = {"torch": Version("2.12.0")}
        assert _compat_conditions_match("", known) is False

    def test_whitespace_in_conditions(self):
        known = {"torch": Version("2.12.0"), "cuda": Version("13.0")}
        assert _compat_conditions_match("torch>=2.11, cuda>=13", known) is True


# ---------------------------------------------------------------------------
# _PinBlock
# ---------------------------------------------------------------------------
class TestPinBlock:
    def test_package_name_simple(self):
        b = _PinBlock(package_line="numpy==1.26.4")
        assert b.package_name == "numpy"

    def test_package_name_hyphenated(self):
        b = _PinBlock(package_line="scikit-learn==1.5.0")
        assert b.package_name == "scikit-learn"

    def test_as_text_no_via(self):
        b = _PinBlock(package_line="numpy==1.26.4")
        assert b.as_text() == "numpy==1.26.4"

    def test_as_text_with_via_comments(self):
        b = _PinBlock(
            package_line="numpy==1.26.4",
            via_comments=["    # via", "    #   pandas", "    #   scipy"],
        )
        expected = "numpy==1.26.4\n    # via\n    #   pandas\n    #   scipy"
        assert b.as_text() == expected

    def test_default_via_comments_empty(self):
        b = _PinBlock(package_line="torch==2.12.0")
        assert b.via_comments == []

    def test_package_name_no_version(self):
        b = _PinBlock(package_line="torch")
        assert b.package_name == "torch"


# ---------------------------------------------------------------------------
# _parse_constraint_file
# ---------------------------------------------------------------------------
class TestParseConstraintFile:
    def test_basic_parsing(self, tmp_path):
        content = (
            "# Pinned with: cuda=130 torch=2.12.0\n"
            "# Generated by: milabench pin\n"
            "\n"
            "numpy==1.26.4\n"
            "    # via\n"
            "    #   pandas\n"
            "scipy==1.14.0\n"
        )
        f = tmp_path / "constraints.txt"
        f.write_text(content)
        header, blocks = _parse_constraint_file(f)
        assert len(header) == 3
        assert header[0] == "# Pinned with: cuda=130 torch=2.12.0"
        assert len(blocks) == 2
        assert blocks[0].package_name == "numpy"
        assert blocks[0].via_comments == ["    # via", "    #   pandas"]
        assert blocks[1].package_name == "scipy"
        assert blocks[1].via_comments == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        header, blocks = _parse_constraint_file(f)
        assert header == []
        assert blocks == []

    def test_header_only(self, tmp_path):
        content = "# Just a comment\n# Another comment\n"
        f = tmp_path / "header_only.txt"
        f.write_text(content)
        header, blocks = _parse_constraint_file(f)
        assert len(header) == 2
        assert blocks == []

    def test_no_header(self, tmp_path):
        content = "numpy==1.26.4\nscipy==1.14.0\n"
        f = tmp_path / "no_header.txt"
        f.write_text(content)
        header, blocks = _parse_constraint_file(f)
        assert header == []
        assert len(blocks) == 2

    def test_blank_lines_between_blocks(self, tmp_path):
        content = (
            "numpy==1.26.4\n"
            "\n"
            "scipy==1.14.0\n"
            "    # via\n"
            "    #   something\n"
            "\n"
            "torch==2.12.0\n"
        )
        f = tmp_path / "blanks.txt"
        f.write_text(content)
        header, blocks = _parse_constraint_file(f)
        assert len(blocks) == 3
        assert blocks[0].package_name == "numpy"
        assert blocks[1].package_name == "scipy"
        assert blocks[2].package_name == "torch"

    def test_via_comments_attached_correctly(self, tmp_path):
        content = (
            "aiohttp==3.9.0\n"
            "    # via\n"
            "    #   dep-a\n"
            "    #   dep-b\n"
            "boto3==1.34.0\n"
            "    # via\n"
            "    #   dep-c\n"
        )
        f = tmp_path / "via.txt"
        f.write_text(content)
        _, blocks = _parse_constraint_file(f)
        assert len(blocks) == 2
        assert len(blocks[0].via_comments) == 3
        assert len(blocks[1].via_comments) == 2


# ---------------------------------------------------------------------------
# _strip_index_urls_from_constraint_file
# ---------------------------------------------------------------------------
class TestStripIndexUrls:
    def test_removes_index_lines(self, tmp_path):
        content = (
            "--index-url https://pypi.org/simple\n"
            "--extra-index-url https://download.pytorch.org/whl/cu130\n"
            "--find-links https://example.com/wheels\n"
            "numpy==1.26.4\n"
        )
        f = tmp_path / "c.txt"
        f.write_text(content)
        _strip_index_urls_from_constraint_file(f, "cuda", "130", "2.12.0")
        result = f.read_text()
        assert "--index-url" not in result
        assert "--extra-index-url" not in result
        assert "--find-links" not in result
        assert "numpy==1.26.4" in result

    def test_replaces_autogenerated_header(self, tmp_path):
        content = (
            "# This file was autogenerated by uv\n"
            "numpy==1.26.4\n"
        )
        f = tmp_path / "c.txt"
        f.write_text(content)
        _strip_index_urls_from_constraint_file(f, "cuda", "130", "2.12.0")
        result = f.read_text()
        assert "# Pinned with: cuda=130 torch=2.12.0" in result
        assert "# Generated by: milabench pin" in result
        assert "This file was autogenerated" not in result

    def test_adds_header_if_missing(self, tmp_path):
        content = "numpy==1.26.4\n"
        f = tmp_path / "c.txt"
        f.write_text(content)
        _strip_index_urls_from_constraint_file(f, "cuda", "130", "2.12.0")
        result = f.read_text()
        lines = result.splitlines()
        assert lines[0] == "# Pinned with: cuda=130 torch=2.12.0"
        assert lines[1] == "# Generated by: milabench pin"

    def test_strips_uv_command_comment(self, tmp_path):
        content = (
            "#    uv pip compile --no-build /tmp/toml-deps-abc123.txt\n"
            "numpy==1.26.4\n"
        )
        f = tmp_path / "c.txt"
        f.write_text(content)
        _strip_index_urls_from_constraint_file(f, "cuda", "130", "2.12.0")
        result = f.read_text()
        assert "uv pip compile" not in result

    def test_normalizes_temp_paths(self, tmp_path):
        content = (
            "# via -r /tmp/toml-deps-abc123.txt\n"
            "# -c /tmp/toml-constraints-xyz789.txt\n"
            "numpy==1.26.4\n"
        )
        f = tmp_path / "c.txt"
        f.write_text(content)
        _strip_index_urls_from_constraint_file(f, "cuda", "130", "2.12.0")
        result = f.read_text()
        assert "requirements.in" in result
        assert "constraints.in" in result
        assert "/tmp/toml-deps-" not in result
        assert "/tmp/toml-constraints-" not in result

    def test_cpu_backend_no_version(self, tmp_path):
        content = "numpy==1.26.4\n"
        f = tmp_path / "c.txt"
        f.write_text(content)
        _strip_index_urls_from_constraint_file(f, "cpu", "", "2.12.0")
        result = f.read_text()
        assert "# Pinned with: cpu torch=2.12.0" in result

    def test_with_arch_in_header(self, tmp_path):
        content = "numpy==1.26.4\n"
        f = tmp_path / "c.txt"
        f.write_text(content)
        _strip_index_urls_from_constraint_file(f, "cuda", "130", "2.12.0", "x86_64")
        result = f.read_text()
        assert "arch=x86_64" in result


# ---------------------------------------------------------------------------
# _extract_common_constraints
# ---------------------------------------------------------------------------
class TestExtractCommonConstraints:
    def _write_constraint(self, path, header_lines, packages):
        """Helper: write a fake constraint file with header + package blocks."""
        parts = list(header_lines)
        for pkg_line, via in packages:
            parts.append(pkg_line)
            parts.extend(via)
        path.write_text("\n".join(parts) + "\n")

    def test_basic_extraction(self, tmp_path):
        f1 = tmp_path / "constraints.cuda130.torch2120.txt"
        f2 = tmp_path / "constraints.rocm72.torch2120.txt"

        common_pkgs = [
            ("numpy==1.26.4", ["    # via", "    #   pandas"]),
            ("scipy==1.14.0", []),
        ]
        unique_f1 = [("cupy==13.0", [])]
        unique_f2 = [("rccl==1.0", [])]

        self._write_constraint(
            f1,
            ["# Header cuda"],
            common_pkgs + unique_f1,
        )
        self._write_constraint(
            f2,
            ["# Header rocm"],
            common_pkgs + unique_f2,
        )

        _extract_common_constraints([f1, f2], tmp_path)

        common_file = tmp_path / "constraints.common.txt"
        assert common_file.exists()
        common_text = common_file.read_text()
        assert "numpy==1.26.4" in common_text
        assert "scipy==1.14.0" in common_text
        assert "cupy" not in common_text
        assert "rccl" not in common_text

        # Individual files should now only have unique packages + -c reference
        f1_text = f1.read_text()
        assert "cupy==13.0" in f1_text
        assert "-c constraints.common.txt" in f1_text
        assert "numpy==1.26.4" not in f1_text

        f2_text = f2.read_text()
        assert "rccl==1.0" in f2_text
        assert "-c constraints.common.txt" in f2_text
        assert "numpy==1.26.4" not in f2_text

    def test_no_common_packages(self, tmp_path, capsys):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        self._write_constraint(f1, ["# H1"], [("pkg-a==1.0", [])])
        self._write_constraint(f2, ["# H2"], [("pkg-b==2.0", [])])

        _extract_common_constraints([f1, f2], tmp_path)

        common_file = tmp_path / "constraints.common.txt"
        assert not common_file.exists()
        captured = capsys.readouterr()
        assert "No common packages" in captured.out

    def test_all_packages_common(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        pkgs = [("torch==2.12.0", []), ("numpy==1.26.4", [])]
        self._write_constraint(f1, ["# H1"], pkgs)
        self._write_constraint(f2, ["# H2"], pkgs)

        _extract_common_constraints([f1, f2], tmp_path)

        common_file = tmp_path / "constraints.common.txt"
        assert common_file.exists()
        common_text = common_file.read_text()
        assert "torch==2.12.0" in common_text
        assert "numpy==1.26.4" in common_text

        # Individual files should only have header + -c ref + no packages
        f1_text = f1.read_text()
        assert "-c constraints.common.txt" in f1_text
        assert "torch==2.12.0" not in f1_text

    def test_strips_old_c_references_in_header(self, tmp_path):
        """Header lines starting with '-c ' are stripped before re-adding the new ref."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"

        # Manually write files so the -c line lands in the header
        # (parser puts comment/blank lines before first package into the header)
        f1.write_text(
            "# H1\n"
            "numpy==1.0\n"
        )
        f2.write_text(
            "# H2\n"
            "numpy==1.0\n"
        )

        _extract_common_constraints([f1, f2], tmp_path)

        common_file = tmp_path / "constraints.common.txt"
        assert common_file.exists()
        f1_text = f1.read_text()
        assert "-c constraints.common.txt" in f1_text
        assert "# H1" in f1_text


# ---------------------------------------------------------------------------
# _build_index_args (uses PlatformConfig dataclasses directly)
# ---------------------------------------------------------------------------
class TestBuildIndexArgs:
    def test_index_url_only(self):
        config = PlatformConfig(
            backends={
                "cuda": BackendConfig(
                    name="cuda",
                    indexes=IndexConfig(
                        index_url="https://download.pytorch.org/whl/cu{cuda}",
                    ),
                )
            }
        )
        args = _build_index_args(config, "cuda", {"cuda": "130"})
        assert args == [
            "--index-url",
            "https://download.pytorch.org/whl/cu130",
        ]

    def test_extra_index_and_find_links(self):
        config = PlatformConfig(
            backends={
                "cuda": BackendConfig(
                    name="cuda",
                    indexes=IndexConfig(
                        index_url="https://pypi.org/simple",
                        extra_index_url=["https://download.pytorch.org/whl/cu{cuda}"],
                        find_links=["https://example.com/{torch}"],
                    ),
                )
            }
        )
        args = _build_index_args(config, "cuda", {"cuda": "130", "torch": "2.12.0"})
        assert "--index-url" in args
        assert "--extra-index-url" in args
        assert "https://download.pytorch.org/whl/cu130" in args
        assert "--find-links" in args
        assert "https://example.com/2.12.0" in args

    def test_unknown_backend_uses_default_pypi(self):
        config = PlatformConfig()
        args = _build_index_args(config, "xpu", {})
        assert args == ["--index-url", "https://pypi.org/simple"]

    def test_no_index_url(self):
        config = PlatformConfig(
            backends={
                "cpu": BackendConfig(
                    name="cpu",
                    indexes=IndexConfig(index_url=""),
                )
            }
        )
        args = _build_index_args(config, "cpu", {})
        assert "--index-url" not in args


# ---------------------------------------------------------------------------
# _build_constraints_content
# ---------------------------------------------------------------------------
class TestBuildConstraintsContent:
    def test_simple_constraints(self):
        config = PlatformConfig(
            backends={
                "cuda": BackendConfig(
                    name="cuda",
                    constraints={"torch": "=={torch}", "torchvision": ">=0.18"},
                )
            }
        )
        lines = _build_constraints_content(config, "cuda", {"torch": "2.12.0"})
        assert "torch==2.12.0" in lines
        assert "torchvision>=0.18" in lines

    def test_empty_constraints(self):
        config = PlatformConfig(
            backends={"cuda": BackendConfig(name="cuda")}
        )
        lines = _build_constraints_content(config, "cuda", {})
        assert lines == []

    def test_includes_compat_constraints(self):
        config = PlatformConfig(
            vars={"torch": "2.12.0", "cuda": "130"},
            backends={"cuda": BackendConfig(name="cuda")},
            compat={
                "torchao": CompatEntry(
                    package="torchao",
                    rules=[
                        CompatRule(conditions="torch>=2.11", constraint="<0.18"),
                    ],
                )
            },
        )
        lines = _build_constraints_content(config, "cuda", {"torch": "2.12.0", "cuda": "130"})
        assert "torchao<0.18" in lines


# ---------------------------------------------------------------------------
# _resolve_compat_constraints
# ---------------------------------------------------------------------------
class TestResolveCompatConstraints:
    def test_no_compat_section(self):
        config = PlatformConfig()
        assert _resolve_compat_constraints(config) == []

    def test_first_match_wins(self):
        config = PlatformConfig(
            vars={"torch": "2.12.0"},
            compat={
                "torchao": CompatEntry(
                    package="torchao",
                    rules=[
                        CompatRule(conditions="torch>=2.12", constraint="<0.20"),
                        CompatRule(conditions="torch>=2.11", constraint="<0.18"),
                    ],
                )
            },
        )
        lines = _resolve_compat_constraints(config, {"torch": "2.12.0"})
        assert lines == ["torchao<0.20"]

    def test_no_rules_match(self):
        config = PlatformConfig(
            vars={"torch": "2.9.0"},
            compat={
                "torchao": CompatEntry(
                    package="torchao",
                    rules=[
                        CompatRule(conditions="torch>=2.11", constraint="<0.18"),
                    ],
                )
            },
        )
        lines = _resolve_compat_constraints(config, {"torch": "2.9.0"})
        assert lines == []

    def test_cuda_version_normalized(self):
        config = PlatformConfig(
            vars={},
            compat={
                "torchao": CompatEntry(
                    package="torchao",
                    rules=[
                        CompatRule(conditions="cuda>=13", constraint="<0.20"),
                    ],
                )
            },
        )
        lines = _resolve_compat_constraints(config, {"cuda": "130"})
        assert lines == ["torchao<0.20"]

    def test_invalid_version_skipped(self):
        config = PlatformConfig(
            vars={},
            compat={
                "torchao": CompatEntry(
                    package="torchao",
                    rules=[
                        CompatRule(conditions="torch>=2.11", constraint="<0.18"),
                    ],
                )
            },
        )
        lines = _resolve_compat_constraints(config, {"torch": "not_a_version"})
        assert lines == []

    def test_multiple_packages(self):
        config = PlatformConfig(
            vars={"torch": "2.12.0"},
            compat={
                "torchao": CompatEntry(
                    package="torchao",
                    rules=[
                        CompatRule(conditions="torch>=2.11", constraint="<0.18"),
                    ],
                ),
                "flash-attn": CompatEntry(
                    package="flash-attn",
                    rules=[
                        CompatRule(conditions="torch>=2.10", constraint=">=2.5"),
                    ],
                ),
            },
        )
        lines = _resolve_compat_constraints(config, {"torch": "2.12.0"})
        assert "torchao<0.18" in lines
        assert "flash-attn>=2.5" in lines
