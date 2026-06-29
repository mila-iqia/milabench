"""Tests for milabench.dependencies module."""

import tempfile
from pathlib import Path
from textwrap import dedent

import pytest

from milabench.dependencies.platforms import (
    PlatformConfig,
    BackendConfig,
    IndexConfig,
    OverrideConfig,
    DiscoveryConfig,
    PinMatrix,
    CompatEntry,
    CompatRule,
    load_platform_config,
    _normalize_overrides,
)
from milabench.dependencies.requirements import (
    BenchmarkRequirements,
    has_toml_requirements,
    load_benchmark_requirements,
    resolve_benchmark,
)
from milabench.dependencies.pin import (
    constraint_filename,
    get_constraint_file,
    _collect_all_toml_deps,
    _build_index_args,
    _build_constraints_content,
    _strip_index_urls_from_constraint_file,
    _get_combinations,
    _normalize_backend_version,
    _resolve_compat_constraints,
    _compat_conditions_match,
)
from milabench.dependencies.discovery import (
    _parse_wheel_filename,
    _platform_to_arch,
    _filter_latest_patch,
    parse_available_wheels,
    DiscoveryConfig as DiscConfig,
    WheelInfo,
)
from milabench.dependencies.install import (
    install_args,
    get_index_args,
    InstallArgs,
)
from milabench.dependencies.overrides import (
    get_overrides,
    _resolve_override,
    ResolvedOverride,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def platforms_toml(tmp_path):
    """Write a minimal platforms.toml and return its path."""
    content = dedent("""\
        [vars]
        cuda = "130"
        rocm = "7.1"
        torch = "2.12.0"
        vllm = "v0.18.1"

        [pin.discovery]
        torch_index = "https://download.pytorch.org/whl/torch/"
        torch_min = "2.10"
        backends = ["cuda", "rocm", "cpu"]
        platforms = ["manylinux_2_28_x86_64", "manylinux_2_28_aarch64"]
        latest_patch_only = true

        [cuda.indexes]
        index-url = "https://pypi.org/simple"
        extra-index-url = [
            "https://download.pytorch.org/whl/cu{cuda}",
        ]
        find-links = [
            "https://github.com/milabench/wheels/releases/expanded_assets/cu{cuda}-wheels",
        ]

        [cuda.constraints]
        torchao = "<0.17.0"
        torch = ">=2.10"

        [cuda.overrides.torchao]
        install = "pip install --no-build-isolation 'torchao<0.17.0'"

        [cuda.overrides.vllm]
        install = "pip install 'vllm @ git+https://github.com/vllm-project/vllm.git@{vllm}'"
        env = { VLLM_TARGET_DEVICE = "cuda" }

        [rocm.indexes]
        index-url = "https://pypi.org/simple"
        extra-index-url = [
            "https://download.pytorch.org/whl/rocm{rocm}",
        ]

        [rocm.constraints]
        torch = ">=2.10"

        [cpu.indexes]
        index-url = "https://pypi.org/simple"
        extra-index-url = [
            "https://download.pytorch.org/whl/cpu",
        ]

        [cpu.constraints]
        torch = ">=2.10"
    """)
    path = tmp_path / "platforms.toml"
    path.write_text(content)
    return path


@pytest.fixture
def benchmark_dir(tmp_path):
    """Create a fake benchmark directory with requirements.toml."""
    bench = tmp_path / "benchmarks" / "fakebench"
    bench.mkdir(parents=True)
    toml_content = dedent("""\
        [common]
        dependencies = [
            "voir>=0.2.19,<0.3",
            "torch",
            "numpy",
        ]

        [cuda]
        dependencies = [
            "flashinfer-python",
            "torchcodec",
        ]

        [rocm]

        [hpu]
        enabled = false

        [xpu]
        enabled = false
    """)
    (bench / "requirements.toml").write_text(toml_content)
    return bench


@pytest.fixture
def benchmark_dir_disabled_rocm(tmp_path):
    """Create a benchmark with rocm disabled."""
    bench = tmp_path / "benchmarks" / "cudaonly"
    bench.mkdir(parents=True)
    toml_content = dedent("""\
        [common]
        dependencies = ["torch", "voir"]

        [cuda]
        dependencies = ["flashinfer-python"]

        [rocm]
        enabled = false

        [xpu]
        enabled = false
    """)
    (bench / "requirements.toml").write_text(toml_content)
    return bench


# ─── platforms.py tests ───────────────────────────────────────────────────────

class TestPlatformConfig:
    def test_load_platform_config(self, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        assert config.vars["cuda"] == "130"
        assert config.vars["rocm"] == "7.1"
        assert config.vars["torch"] == "2.12.0"
        assert config.vars["vllm"] == "v0.18.1"

    def test_discovery_parsed(self, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        assert config.discovery is not None
        assert config.discovery.torch_min == "2.10"
        assert config.discovery.backends == ["cuda", "rocm", "cpu"]
        assert config.discovery.platforms == ["manylinux_2_28_x86_64", "manylinux_2_28_aarch64"]
        assert config.discovery.latest_patch_only is True

    def test_pin_matrix_is_none_when_discovery(self, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        assert config.pin_matrix is None

    def test_backends_parsed(self, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        assert "cuda" in config.backends
        assert "rocm" in config.backends
        assert "cpu" in config.backends
        cuda = config.backends["cuda"]
        assert cuda.indexes.extra_index_url == ["https://download.pytorch.org/whl/cu{cuda}"]
        assert cuda.constraints["torchao"] == "<0.17.0"

    def test_overrides_parsed(self, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        cuda = config.backends["cuda"]
        assert "torchao" in cuda.overrides
        assert "vllm" in cuda.overrides
        assert cuda.overrides["vllm"].env == {"VLLM_TARGET_DEVICE": "cuda"}

    def test_resolve_vars_with_overrides(self, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        resolved = config.resolve_vars({"cuda": "126", "torch": "2.11.0"})
        assert resolved["cuda"] == "126"
        assert resolved["torch"] == "2.11.0"
        assert resolved["rocm"] == "7.1"  # unchanged

    def test_resolve_string(self, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        result = config.resolve_string("https://download.pytorch.org/whl/cu{cuda}")
        assert result == "https://download.pytorch.org/whl/cu130"

    def test_get_backend_unknown_returns_default(self, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        backend = config.get_backend("unknown_backend")
        assert backend.name == "unknown_backend"
        assert backend.indexes.index_url == "https://pypi.org/simple"
        assert backend.constraints == {}


class TestPinMatrix:
    def test_combinations(self):
        matrix = PinMatrix(
            torch=["2.10.0", "2.12.0"],
            backends={"cuda": ["126", "130"], "rocm": ["7.1"]},
        )
        combos = matrix.combinations()
        assert ("cuda", "126", "2.10.0") in combos
        assert ("cuda", "130", "2.12.0") in combos
        assert ("rocm", "7.1", "2.10.0") in combos
        assert len(combos) == 6

    def test_exclusions(self):
        matrix = PinMatrix(
            torch=["2.10.0", "2.12.0"],
            backends={"cuda": ["126", "130"]},
            exclude=[{"cuda": "126", "torch": "2.12.0"}],
        )
        combos = matrix.combinations()
        assert ("cuda", "126", "2.12.0") not in combos
        assert ("cuda", "130", "2.12.0") in combos
        assert len(combos) == 3


# ─── requirements.py tests ───────────────────────────────────────────────────

class TestRequirements:
    def test_has_toml_requirements(self, benchmark_dir):
        assert has_toml_requirements(benchmark_dir) is True

    def test_has_toml_requirements_missing(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert has_toml_requirements(empty) is False

    def test_load_benchmark_requirements(self, benchmark_dir):
        reqs = load_benchmark_requirements(benchmark_dir)
        assert reqs.common == ["voir>=0.2.19,<0.3", "torch", "numpy"]
        assert "cuda" in reqs.backends
        assert reqs.backends["cuda"].dependencies == ["flashinfer-python", "torchcodec"]

    def test_is_enabled(self, benchmark_dir):
        reqs = load_benchmark_requirements(benchmark_dir)
        assert reqs.is_enabled("cuda") is True
        assert reqs.is_enabled("rocm") is True
        assert reqs.is_enabled("hpu") is False
        assert reqs.is_enabled("xpu") is False
        # Unknown backend with no section → enabled (inherits common)
        assert reqs.is_enabled("cpu") is True

    def test_is_enabled_disabled(self, benchmark_dir_disabled_rocm):
        reqs = load_benchmark_requirements(benchmark_dir_disabled_rocm)
        assert reqs.is_enabled("rocm") is False

    def test_get_dependencies_cuda(self, benchmark_dir):
        reqs = load_benchmark_requirements(benchmark_dir)
        deps = reqs.get_dependencies("cuda")
        assert "voir>=0.2.19,<0.3" in deps
        assert "torch" in deps
        assert "numpy" in deps
        assert "flashinfer-python" in deps
        assert "torchcodec" in deps

    def test_get_dependencies_rocm(self, benchmark_dir):
        reqs = load_benchmark_requirements(benchmark_dir)
        deps = reqs.get_dependencies("rocm")
        assert "voir>=0.2.19,<0.3" in deps
        assert "torch" in deps
        assert "flashinfer-python" not in deps

    def test_get_dependencies_disabled(self, benchmark_dir):
        reqs = load_benchmark_requirements(benchmark_dir)
        deps = reqs.get_dependencies("hpu")
        assert deps == []

    def test_resolve_benchmark(self, benchmark_dir, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        deps = resolve_benchmark(benchmark_dir, "cuda", config)
        assert "voir>=0.2.19,<0.3" in deps
        assert "flashinfer-python" in deps

    def test_resolve_benchmark_disabled_raises(self, benchmark_dir, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        with pytest.raises(ValueError, match="does not support backend 'hpu'"):
            resolve_benchmark(benchmark_dir, "hpu", config)

    def test_resolve_benchmark_variable_interpolation(self, tmp_path, platforms_toml):
        bench = tmp_path / "benchmarks" / "varbench"
        bench.mkdir(parents=True)
        (bench / "requirements.toml").write_text(dedent("""\
            [common]
            dependencies = ["torch"]

            [cuda]
            dependencies = ["vllm=={vllm}+cu{cuda}"]
        """))
        config = load_platform_config(path=platforms_toml)
        deps = resolve_benchmark(bench, "cuda", config)
        assert "vllm==v0.18.1+cu130" in deps

    def test_resolve_benchmark_missing_var_raises(self, tmp_path, platforms_toml):
        bench = tmp_path / "benchmarks" / "badbench"
        bench.mkdir(parents=True)
        (bench / "requirements.toml").write_text(dedent("""\
            [common]
            dependencies = ["pkg=={nonexistent_var}"]
        """))
        config = load_platform_config(path=platforms_toml)
        with pytest.raises(ValueError, match="nonexistent_var"):
            resolve_benchmark(bench, "cuda", config)


# ─── discovery.py tests ──────────────────────────────────────────────────────

class TestDiscovery:
    def test_parse_cuda_wheel(self):
        info = _parse_wheel_filename(
            "torch-2.12.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl"
        )
        assert info is not None
        assert info.torch_version == "2.12.0"
        assert info.backend_type == "cuda"
        assert info.backend_version == "130"
        assert info.python_version == "312"
        assert info.platform == "manylinux_2_28_x86_64"

    def test_parse_rocm_wheel(self):
        info = _parse_wheel_filename(
            "torch-2.11.0+rocm7.1-cp312-cp312-manylinux_2_28_x86_64.whl"
        )
        assert info is not None
        assert info.backend_type == "rocm"
        assert info.backend_version == "7.1"
        assert info.torch_version == "2.11.0"

    def test_parse_cpu_wheel(self):
        info = _parse_wheel_filename(
            "torch-2.12.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl"
        )
        assert info is not None
        assert info.backend_type == "cpu"
        assert info.backend_version == ""
        assert info.torch_version == "2.12.1"

    def test_parse_aarch64_wheel(self):
        info = _parse_wheel_filename(
            "torch-2.10.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl"
        )
        assert info is not None
        assert info.platform == "manylinux_2_28_aarch64"

    def test_parse_invalid_line(self):
        assert _parse_wheel_filename("not a wheel filename") is None
        assert _parse_wheel_filename("numpy-1.26.0-cp312-cp312-linux_x86_64.whl") is None

    def test_parse_html_link(self):
        info = _parse_wheel_filename(
            '<a href="torch-2.12.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl">torch-2.12.0</a>'
        )
        assert info is not None
        assert info.torch_version == "2.12.0"

    def test_platform_to_arch(self):
        assert _platform_to_arch("manylinux_2_28_x86_64") == "x86_64"
        assert _platform_to_arch("manylinux_2_28_aarch64") == "aarch64"
        assert _platform_to_arch("win_amd64") == "amd64"

    def test_filter_latest_patch(self):
        combos = [
            ("cuda", "130", "2.12.0", "x86_64"),
            ("cuda", "130", "2.12.1", "x86_64"),
            ("cuda", "130", "2.11.0", "x86_64"),
            ("cuda", "130", "2.10.0", "x86_64"),
        ]
        filtered = _filter_latest_patch(combos)
        versions = [t[2] for t in filtered]
        assert "2.12.1" in versions
        assert "2.12.0" not in versions  # superseded by 2.12.1
        assert "2.11.0" in versions
        assert "2.10.0" in versions

    def test_filter_latest_patch_per_arch(self):
        combos = [
            ("cuda", "130", "2.12.0", "x86_64"),
            ("cuda", "130", "2.12.1", "x86_64"),
            ("cuda", "130", "2.12.0", "aarch64"),
            # no 2.12.1 for aarch64
        ]
        filtered = _filter_latest_patch(combos)
        x86 = [t for t in filtered if t[3] == "x86_64"]
        arm = [t for t in filtered if t[3] == "aarch64"]
        assert x86[0][2] == "2.12.1"  # latest for x86
        assert arm[0][2] == "2.12.0"  # only version for aarch64

    def test_parse_available_wheels(self):
        content = dedent("""\
            torch-2.12.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl
            torch-2.12.0+rocm7.1-cp312-cp312-manylinux_2_28_x86_64.whl
            torch-2.12.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl
            not-a-wheel.tar.gz
            numpy-1.26.0.whl
        """)
        wheels = parse_available_wheels(content)
        assert len(wheels) == 3
        backends = {w.backend_type for w in wheels}
        assert backends == {"cuda", "rocm", "cpu"}


# ─── pin.py tests ────────────────────────────────────────────────────────────

class TestPin:
    def test_constraint_filename_cuda(self):
        assert constraint_filename("cuda", "130", "2.12.0", "x86_64") == \
            "constraints.cuda130.torch2120.x86_64.txt"

    def test_constraint_filename_rocm(self):
        assert constraint_filename("rocm", "7.1", "2.10.0", "aarch64") == \
            "constraints.rocm71.torch2100.aarch64.txt"

    def test_constraint_filename_cpu(self):
        assert constraint_filename("cpu", "", "2.12.0", "x86_64") == \
            "constraints.cpu.torch2120.x86_64.txt"

    def test_constraint_filename_no_arch(self):
        assert constraint_filename("cuda", "130", "2.12.0") == \
            "constraints.cuda130.torch2120.txt"

    def test_get_constraint_file(self, tmp_path):
        path = get_constraint_file(tmp_path, "cuda", "130", "2.12.0", "x86_64")
        assert path == tmp_path / "constraints.cuda130.torch2120.x86_64.txt"

    def test_collect_all_toml_deps(self, benchmark_dir, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        benchmarks_dir = benchmark_dir.parent
        deps = _collect_all_toml_deps(benchmarks_dir, "cuda", config)
        assert "voir>=0.2.19,<0.3" in deps
        assert "flashinfer-python" in deps
        assert "torch" in deps

    def test_collect_all_toml_deps_rocm_no_flashinfer(self, benchmark_dir, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        benchmarks_dir = benchmark_dir.parent
        deps = _collect_all_toml_deps(benchmarks_dir, "rocm", config)
        assert "voir>=0.2.19,<0.3" in deps
        assert "torch" in deps
        assert "flashinfer-python" not in deps

    def test_collect_skips_disabled_backend(self, benchmark_dir_disabled_rocm, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        benchmarks_dir = benchmark_dir_disabled_rocm.parent
        deps = _collect_all_toml_deps(benchmarks_dir, "rocm", config)
        assert "flashinfer-python" not in deps
        # The benchmark is disabled for rocm, so no deps collected
        assert deps == []

    def test_build_index_args(self, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        args = _build_index_args(config, "cuda")
        assert "--index-url" in args
        assert "https://pypi.org/simple" in args
        assert "--extra-index-url" in args
        assert "https://download.pytorch.org/whl/cu130" in args

    def test_build_index_args_with_override(self, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        args = _build_index_args(config, "cuda", overrides={"cuda": "126"})
        assert "https://download.pytorch.org/whl/cu126" in args

    def test_build_constraints_content(self, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        lines = _build_constraints_content(config, "cuda")
        assert "torchao<0.17.0" in lines
        assert "torch>=2.10" in lines

    def test_strip_index_urls(self, tmp_path):
        pin_file = tmp_path / "test_constraints.txt"
        pin_file.write_text(dedent("""\
            # This file was autogenerated by uv via the following command:
            --index-url https://pypi.org/simple
            --extra-index-url https://download.pytorch.org/whl/cu130
            --find-links https://github.com/foo
            torch==2.12.0
            numpy==2.3.0
        """))
        _strip_index_urls_from_constraint_file(pin_file, "cuda", "130", "2.12.0", "x86_64")
        content = pin_file.read_text()
        assert "--index-url" not in content
        assert "--extra-index-url" not in content
        assert "--find-links" not in content
        assert "torch==2.12.0" in content
        assert "numpy==2.3.0" in content
        assert "# Pinned with: cuda=130 torch=2.12.0 arch=x86_64" in content

    def test_get_combinations_discovery(self, platforms_toml, monkeypatch):
        config = load_platform_config(path=platforms_toml)

        fake_combos = [
            ("cuda", "130", "2.12.0", "x86_64"),
            ("rocm", "7.1", "2.12.0", "x86_64"),
        ]
        monkeypatch.setattr(
            "milabench.dependencies.pin.discover_combinations",
            lambda dc: fake_combos,
        )
        # Need to also patch the import inside _get_combinations
        import milabench.dependencies.discovery as disc_mod
        monkeypatch.setattr(disc_mod, "discover_combinations", lambda dc: fake_combos)
        monkeypatch.setattr(disc_mod, "print_discovered_combinations", lambda c: None)

        combos = _get_combinations(config)
        assert len(combos) == 2
        assert combos[0] == ("cuda", "130", "2.12.0", "x86_64")

    def test_get_combinations_legacy_matrix(self, tmp_path):
        content = dedent("""\
            [vars]
            cuda = "130"
            torch = "2.10.0"

            [pin.matrix]
            torch = ["2.10.0"]

            [pin.matrix.backends]
            cuda = ["130"]
        """)
        path = tmp_path / "platforms.toml"
        path.write_text(content)
        config = load_platform_config(path=path)
        combos = _get_combinations(config)
        assert combos == [("cuda", "130", "2.10.0", "")]


# ─── install.py tests ────────────────────────────────────────────────────────

class TestInstall:
    def test_get_index_args(self, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        args = get_index_args(config, "cuda")
        assert "--index-url" in args
        assert "https://pypi.org/simple" in args
        assert "--extra-index-url" in args
        assert "https://download.pytorch.org/whl/cu130" in args

    def test_get_index_args_rocm(self, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        args = get_index_args(config, "rocm")
        assert "https://download.pytorch.org/whl/rocm7.1" in args

    def test_install_args_generates_temp_file(self, benchmark_dir, platforms_toml, tmp_path):
        config = load_platform_config(path=platforms_toml)
        pin_dir = tmp_path / ".pin"
        pin_dir.mkdir()

        result = install_args(
            benchmark_path=benchmark_dir,
            platform_config=config,
            backend="cuda",
            pin_dir=pin_dir,
            unpinned=True,
        )
        try:
            assert result.requirements_file.exists()
            content = result.requirements_file.read_text()
            assert "torch" in content
            assert "flashinfer-python" in content
            assert result.constraint_file is None  # unpinned
        finally:
            result.cleanup()
        assert not result.requirements_file.exists()

    def test_install_args_finds_constraint_file(self, benchmark_dir, platforms_toml, tmp_path):
        config = load_platform_config(path=platforms_toml)
        pin_dir = tmp_path / ".pin"
        pin_dir.mkdir()

        # Create a constraint file that should be found
        import platform as plat
        arch = plat.machine()
        constraint_file = get_constraint_file(pin_dir, "cuda", "130", "2.12.0", arch)
        constraint_file.write_text("torch==2.12.0\n")

        result = install_args(
            benchmark_path=benchmark_dir,
            platform_config=config,
            backend="cuda",
            pin_dir=pin_dir,
        )
        try:
            assert result.constraint_file is not None
            assert result.constraint_file == constraint_file
        finally:
            result.cleanup()

    def test_install_args_as_pip_args(self, benchmark_dir, platforms_toml, tmp_path):
        config = load_platform_config(path=platforms_toml)
        pin_dir = tmp_path / ".pin"
        pin_dir.mkdir()

        result = install_args(
            benchmark_path=benchmark_dir,
            platform_config=config,
            backend="cuda",
            pin_dir=pin_dir,
            unpinned=True,
        )
        try:
            pip_args = result.as_pip_args()
            assert "-r" in pip_args
            assert "--index-url" in pip_args
            assert "--extra-index-url" in pip_args
            assert "-c" not in pip_args  # unpinned
        finally:
            result.cleanup()


# ─── overrides.py tests ──────────────────────────────────────────────────────

class TestOverrides:
    def test_resolve_override(self):
        cfg = OverrideConfig(
            package="vllm",
            install="pip install 'vllm @ git+https://github.com/vllm-project/vllm.git@{vllm}'",
            env={"VLLM_TARGET_DEVICE": "cuda"},
        )
        variables = {"vllm": "v0.18.1", "cuda": "130"}
        result = _resolve_override(cfg, variables, source="platform")
        assert result is not None
        assert "v0.18.1" in result.install_command
        assert result.env == {"VLLM_TARGET_DEVICE": "cuda"}
        assert result.source == "platform"

    def test_resolve_override_no_install(self):
        cfg = OverrideConfig(package="foo", install=None)
        result = _resolve_override(cfg, {}, source="platform")
        assert result is None

    def test_resolve_override_missing_var(self):
        cfg = OverrideConfig(
            package="foo",
            install="pip install foo=={missing}",
        )
        with pytest.raises(ValueError, match="missing"):
            _resolve_override(cfg, {"cuda": "130"}, source="platform")

    def test_get_overrides_platform_level(self, benchmark_dir, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        overrides = get_overrides(benchmark_dir, "cuda", config)
        packages = [o.package for o in overrides]
        assert "torchao" in packages
        assert "vllm" in packages

    def test_get_overrides_no_overrides_for_rocm(self, benchmark_dir, platforms_toml):
        config = load_platform_config(path=platforms_toml)
        overrides = get_overrides(benchmark_dir, "rocm", config)
        assert overrides == []

    def test_benchmark_override_supersedes_platform(self, tmp_path, platforms_toml):
        bench = tmp_path / "benchmarks" / "override_test"
        bench.mkdir(parents=True)
        (bench / "requirements.toml").write_text(dedent("""\
            [common]
            dependencies = ["torch"]

            [cuda]
            dependencies = []

            [cuda.overrides.torchao]
            install = "pip install torchao==0.15.0"
        """))
        config = load_platform_config(path=platforms_toml)
        overrides = get_overrides(bench, "cuda", config)
        torchao_override = next(o for o in overrides if o.package == "torchao")
        assert "0.15.0" in torchao_override.install_command
        assert torchao_override.source == "benchmark"


# ─── normalize_overrides tests ───────────────────────────────────────────────

class TestNormalizeOverrides:
    def test_backend_cu130(self):
        result = _normalize_overrides({"backend": "cu130"})
        assert result == {"cuda": "130"}

    def test_backend_cuda130(self):
        result = _normalize_overrides({"backend": "cuda130"})
        assert result == {"cuda": "130"}

    def test_backend_rocm72(self):
        result = _normalize_overrides({"backend": "rocm7.2"})
        assert result == {"rocm": "7.2"}

    def test_backend_cpu(self):
        result = _normalize_overrides({"backend": "cpu"})
        assert result == {}

    def test_torch_two_parts(self):
        result = _normalize_overrides({"torch": "2.10"})
        assert result == {"torch": "2.10.0"}

    def test_torch_three_parts_unchanged(self):
        result = _normalize_overrides({"torch": "2.12.1"})
        assert result == {"torch": "2.12.1"}

    def test_combined(self):
        result = _normalize_overrides({"backend": "cu130", "torch": "2.11"})
        assert result == {"cuda": "130", "torch": "2.11.0"}

    def test_backend_does_not_override_explicit(self):
        result = _normalize_overrides({"backend": "cu130", "cuda": "126"})
        assert result["cuda"] == "126"

    def test_resolve_vars_applies_normalization(self):
        config = PlatformConfig(vars={"cuda": "130", "torch": "2.12.0"})
        resolved = config.resolve_vars({"backend": "cu126", "torch": "2.11"})
        assert resolved["cuda"] == "126"
        assert resolved["torch"] == "2.11.0"


# ─── compat tests ────────────────────────────────────────────────────────────

@pytest.fixture
def compat_toml(tmp_path):
    """Write a platforms.toml with [compat.*] sections."""
    content = dedent("""\
        [vars]
        cuda = "130"
        rocm = "7.1"
        torch = "2.12.0"

        [cuda.indexes]
        index-url = "https://pypi.org/simple"

        [cuda.constraints]
        torch = ">=2.10"

        [rocm.indexes]
        index-url = "https://pypi.org/simple"

        [cpu.indexes]
        index-url = "https://pypi.org/simple"

        [compat.torchao]
        "torch>=2.12" = "<0.19"
        "torch>=2.11,torch<2.12" = "<0.18"
        "torch>=2.10,torch<2.11" = "<0.17"

        [compat.torchcodec]
        "torch>=2.10,torch<2.11" = "<0.12"

        [compat.flashinfer-python]
        "cuda>=13" = ">=0.6"
        "cuda>=12,cuda<13" = ">=0.5,<0.6"

        [compat.mslk]
        "torch>=2.12,torch<2.13" = ">=1.2,<1.3"
        "torch>=2.11,torch<2.12" = ">=1.1,<1.2"
        "torch>=2.10,torch<2.11" = ">=1.0,<1.1"
    """)
    path = tmp_path / "platforms.toml"
    path.write_text(content)
    return path


class TestCompatParsing:
    def test_compat_sections_parsed(self, compat_toml):
        config = load_platform_config(path=compat_toml)
        assert "torchao" in config.compat
        assert "torchcodec" in config.compat
        assert "flashinfer-python" in config.compat
        assert "mslk" in config.compat

    def test_compat_rules_order(self, compat_toml):
        config = load_platform_config(path=compat_toml)
        torchao = config.compat["torchao"]
        assert torchao.package == "torchao"
        assert len(torchao.rules) == 3
        assert torchao.rules[0].conditions == "torch>=2.12"
        assert torchao.rules[0].constraint == "<0.19"
        assert torchao.rules[2].conditions == "torch>=2.10,torch<2.11"
        assert torchao.rules[2].constraint == "<0.17"

    def test_compat_entry_structure(self, compat_toml):
        config = load_platform_config(path=compat_toml)
        flashinfer = config.compat["flashinfer-python"]
        assert flashinfer.rules[0].conditions == "cuda>=13"
        assert flashinfer.rules[0].constraint == ">=0.6"


class TestNormalizeBackendVersion:
    def test_compact_three_digits(self):
        assert _normalize_backend_version("130") == "13.0"

    def test_compact_three_digits_126(self):
        assert _normalize_backend_version("126") == "12.6"

    def test_already_dotted(self):
        assert _normalize_backend_version("7.1") == "7.1"

    def test_empty(self):
        assert _normalize_backend_version("") == ""


class TestCompatConditionsMatch:
    def test_single_match(self):
        from packaging.version import Version
        known = {"torch": Version("2.11.0")}
        assert _compat_conditions_match("torch>=2.11", known) is True

    def test_single_no_match(self):
        from packaging.version import Version
        known = {"torch": Version("2.10.0")}
        assert _compat_conditions_match("torch>=2.11", known) is False

    def test_multi_condition_all_match(self):
        from packaging.version import Version
        known = {"torch": Version("2.11.0"), "cuda": Version("13.0")}
        assert _compat_conditions_match("torch>=2.11,cuda>=13", known) is True

    def test_multi_condition_partial_match(self):
        from packaging.version import Version
        known = {"torch": Version("2.11.0"), "cuda": Version("12.6")}
        assert _compat_conditions_match("torch>=2.11,cuda>=13", known) is False

    def test_unknown_variable_no_match(self):
        from packaging.version import Version
        known = {"torch": Version("2.11.0")}
        assert _compat_conditions_match("cuda>=13", known) is False

    def test_range_condition(self):
        from packaging.version import Version
        known = {"torch": Version("2.11.0")}
        assert _compat_conditions_match("torch>=2.11,torch<2.12", known) is True
        known = {"torch": Version("2.12.0")}
        assert _compat_conditions_match("torch>=2.11,torch<2.12", known) is False


class TestResolveCompatConstraints:
    def test_torch_2100_cuda130(self, compat_toml):
        config = load_platform_config(path=compat_toml)
        lines = _resolve_compat_constraints(config, {"torch": "2.10.0", "cuda": "130"})
        assert "torchao<0.17" in lines
        assert "torchcodec<0.12" in lines
        assert "flashinfer-python>=0.6" in lines
        assert "mslk>=1.0,<1.1" in lines

    def test_torch_2110_cuda126(self, compat_toml):
        config = load_platform_config(path=compat_toml)
        lines = _resolve_compat_constraints(config, {"torch": "2.11.0", "cuda": "126"})
        assert "torchao<0.18" in lines
        assert "flashinfer-python>=0.5,<0.6" in lines
        assert "mslk>=1.1,<1.2" in lines
        # torchcodec rule only applies to torch<2.11
        assert not any("torchcodec" in l for l in lines)

    def test_torch_2120_cuda130(self, compat_toml):
        config = load_platform_config(path=compat_toml)
        lines = _resolve_compat_constraints(config, {"torch": "2.12.0", "cuda": "130"})
        assert "torchao<0.19" in lines
        assert "flashinfer-python>=0.6" in lines
        assert "mslk>=1.2,<1.3" in lines

    def test_first_match_wins(self, compat_toml):
        config = load_platform_config(path=compat_toml)
        # torch 2.12.0 matches both "torch>=2.12" and would match "torch>=2.11,torch<2.12" → no
        # Only first rule should fire
        lines = _resolve_compat_constraints(config, {"torch": "2.12.0", "cuda": "130"})
        assert "torchao<0.19" in lines
        assert "torchao<0.18" not in lines

    def test_no_compat_if_empty(self):
        config = PlatformConfig(vars={"torch": "2.12.0", "cuda": "130"})
        lines = _resolve_compat_constraints(config, {"torch": "2.12.0", "cuda": "130"})
        assert lines == []

    def test_rocm_backend_no_cuda_rules(self, tmp_path):
        """When cuda is absent from vars entirely, cuda-only compat rules don't fire."""
        content = dedent("""\
            [vars]
            rocm = "7.1"
            torch = "2.11.0"

            [rocm.indexes]
            index-url = "https://pypi.org/simple"

            [compat.torchao]
            "torch>=2.11,torch<2.12" = "<0.18"

            [compat.flashinfer-python]
            "cuda>=13" = ">=0.6"
        """)
        path = tmp_path / "platforms_rocm_only.toml"
        path.write_text(content)
        config = load_platform_config(path=path)
        lines = _resolve_compat_constraints(config, {"torch": "2.11.0", "rocm": "7.1"})
        assert "torchao<0.18" in lines
        # flashinfer has cuda-only conditions → should not match (no cuda var)
        assert not any("flashinfer" in l for l in lines)

    def test_build_constraints_includes_compat(self, compat_toml):
        config = load_platform_config(path=compat_toml)
        lines = _build_constraints_content(config, "cuda", {"torch": "2.10.0", "cuda": "130"})
        # Static constraint
        assert "torch>=2.10" in lines
        # Compat-derived
        assert "torchao<0.17" in lines
        assert "flashinfer-python>=0.6" in lines
