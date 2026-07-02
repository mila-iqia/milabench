"""Tests for milabench.config.config module.

Covers all classes, type alias, and the SystemConfig import.
"""

import pytest

from milabench.config.config import (
    VoirSettings,
    ExecutionPlan,
    BenchmarkConfiguration,
    BenchmarkConfigurationFile,
    BenchmarkConfigurations,
    Configuration,
)


# ---------------------------------------------------------------------------
# VoirSettings
# ---------------------------------------------------------------------------


class TestVoirSettings:
    def test_class_exists(self):
        assert VoirSettings is not None

    def test_options_annotations(self):
        ann = VoirSettings.Options.__annotations__
        assert "stop" in ann
        assert ann["stop"] is int
        assert "interval" in ann
        assert ann["interval"] is str

    def test_validation_usage_annotations(self):
        ann = VoirSettings.Validation.Usage.__annotations__
        assert "gpu_load_threshold" in ann
        assert ann["gpu_load_threshold"] is float
        assert "gpu_mem_threshold" in ann
        assert ann["gpu_mem_threshold"] is float

    def test_validation_annotations(self):
        ann = VoirSettings.Validation.__annotations__
        assert "usage" in ann
        assert ann["usage"] is VoirSettings.Validation.Usage

    def test_top_level_annotations(self):
        ann = VoirSettings.__annotations__
        assert "options" in ann
        assert ann["options"] is VoirSettings.Options
        assert "validation" in ann
        assert ann["validation"] is VoirSettings.Validation

    def test_instantiate(self):
        vs = VoirSettings()
        assert isinstance(vs, VoirSettings)

    def test_options_instantiate(self):
        opts = VoirSettings.Options()
        assert isinstance(opts, VoirSettings.Options)

    def test_validation_instantiate(self):
        val = VoirSettings.Validation()
        assert isinstance(val, VoirSettings.Validation)

    def test_usage_instantiate(self):
        usage = VoirSettings.Validation.Usage()
        assert isinstance(usage, VoirSettings.Validation.Usage)

    def test_set_options_attributes(self):
        opts = VoirSettings.Options()
        opts.stop = 10
        opts.interval = "5s"
        assert opts.stop == 10
        assert opts.interval == "5s"

    def test_set_usage_attributes(self):
        usage = VoirSettings.Validation.Usage()
        usage.gpu_load_threshold = 0.8
        usage.gpu_mem_threshold = 0.5
        assert usage.gpu_load_threshold == 0.8
        assert usage.gpu_mem_threshold == 0.5


# ---------------------------------------------------------------------------
# ExecutionPlan
# ---------------------------------------------------------------------------


class TestExecutionPlan:
    def test_class_exists(self):
        assert ExecutionPlan is not None

    def test_annotations(self):
        ann = ExecutionPlan.__annotations__
        assert "method" in ann
        assert ann["method"] is str
        assert "n" in ann
        assert ann["n"] is int

    def test_instantiate(self):
        ep = ExecutionPlan()
        assert isinstance(ep, ExecutionPlan)

    def test_set_attributes(self):
        ep = ExecutionPlan()
        ep.method = "njobs"
        ep.n = 4
        assert ep.method == "njobs"
        assert ep.n == 4


# ---------------------------------------------------------------------------
# BenchmarkConfiguration
# ---------------------------------------------------------------------------


class TestBenchmarkConfiguration:
    def test_class_exists(self):
        assert BenchmarkConfiguration is not None

    def test_all_annotations_present(self):
        ann = BenchmarkConfiguration.__annotations__
        expected = {
            "enabled": bool,
            "url": str,
            "definition": str,
            "inherits": str,
            "group": str,
            "install_group": str,
            "weight": float,
            "num_machines": int,
            "max_duration": int,
        }
        for key, typ in expected.items():
            assert key in ann, f"Missing annotation: {key}"
            assert ann[key] is typ, f"Wrong type for {key}: expected {typ}, got {ann[key]}"

    def test_voir_annotation(self):
        assert BenchmarkConfiguration.__annotations__["voir"] is VoirSettings

    def test_plan_annotation(self):
        assert BenchmarkConfiguration.__annotations__["plan"] is ExecutionPlan

    def test_tags_annotation(self):
        assert BenchmarkConfiguration.__annotations__["tags"] == list[str]

    def test_requires_capabilities_annotation(self):
        assert BenchmarkConfiguration.__annotations__["requires_capabilities"] == list[str]

    def test_argv_annotation(self):
        assert BenchmarkConfiguration.__annotations__["argv"] is any

    def test_instantiate(self):
        bc = BenchmarkConfiguration()
        assert isinstance(bc, BenchmarkConfiguration)

    def test_set_typical_attributes(self):
        bc = BenchmarkConfiguration()
        bc.enabled = True
        bc.url = "https://github.com/example/repo"
        bc.definition = "benchmarks/test"
        bc.group = "llm"
        bc.tags = ["monogpu"]
        bc.weight = 1.5
        bc.num_machines = 2
        bc.max_duration = 600
        assert bc.enabled is True
        assert bc.url == "https://github.com/example/repo"
        assert bc.tags == ["monogpu"]
        assert bc.weight == 1.5
        assert bc.num_machines == 2
        assert bc.max_duration == 600


# ---------------------------------------------------------------------------
# BenchmarkConfigurationFile
# ---------------------------------------------------------------------------


class TestBenchmarkConfigurationFile:
    def test_class_exists(self):
        assert BenchmarkConfigurationFile is not None

    def test_annotations(self):
        ann = BenchmarkConfigurationFile.__annotations__
        assert "include" in ann
        assert ann["include"] == list[str]
        assert "benchmarks" in ann
        assert ann["benchmarks"] == dict[str, BenchmarkConfiguration]

    def test_instantiate(self):
        bcf = BenchmarkConfigurationFile()
        assert isinstance(bcf, BenchmarkConfigurationFile)


# ---------------------------------------------------------------------------
# BenchmarkConfigurations type alias
# ---------------------------------------------------------------------------


class TestBenchmarkConfigurations:
    def test_is_dict_type(self):
        assert BenchmarkConfigurations == dict[str, BenchmarkConfiguration]

    def test_usable_as_type_hint(self):
        data: BenchmarkConfigurations = {}
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_class_exists(self):
        assert Configuration is not None

    def test_system_annotation(self):
        from milabench.system import SystemConfig

        ann = Configuration.__annotations__
        assert "system" in ann
        assert ann["system"] is SystemConfig

    def test_benchmarks_annotation(self):
        ann = Configuration.__annotations__
        assert "benchmarks" in ann
        assert ann["benchmarks"] is BenchmarkConfigurations

    def test_instantiate(self):
        cfg = Configuration()
        assert isinstance(cfg, Configuration)


# ---------------------------------------------------------------------------
# Import-level coverage
# ---------------------------------------------------------------------------


class TestModuleImport:
    def test_import_system_config(self):
        """The module imports SystemConfig from milabench.system."""
        from milabench.system import SystemConfig

        assert SystemConfig is not None

    def test_all_exports_importable(self):
        """All top-level names from config.config are importable."""
        import milabench.config.config as mod

        names = [
            "VoirSettings",
            "ExecutionPlan",
            "BenchmarkConfiguration",
            "BenchmarkConfigurationFile",
            "BenchmarkConfigurations",
            "Configuration",
        ]
        for name in names:
            assert hasattr(mod, name), f"{name} not found in module"

    def test_nested_class_access(self):
        """Nested classes are accessible through their parent."""
        assert hasattr(VoirSettings, "Options")
        assert hasattr(VoirSettings, "Validation")
        assert hasattr(VoirSettings.Validation, "Usage")

    def test_class_hierarchy_independence(self):
        """Each top-level class is independent (no shared base)."""
        classes = [VoirSettings, ExecutionPlan, BenchmarkConfiguration,
                   BenchmarkConfigurationFile, Configuration]
        for cls in classes:
            assert cls.__bases__ == (object,)
