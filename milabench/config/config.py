

from ..system import SystemConfig


class VoirSettings:
    class Options:
        stop: int
        interval: str
    
    class Validation:
        class Usage:
            gpu_load_threshold: float
            gpu_mem_threshold: float
        
        usage: Usage

    options: Options
    validation: Validation


class ExecutionPlan:
    method: str
    n: int


class BenchmarkConfiguration:
    enabled: bool
    voir: VoirSettings
    url: str
    definition: str
    inherits: str
    group: str
    tags: list[str]
    install_group: str
    plan: ExecutionPlan
    argv: any
    weight: float
    num_machines: int
    requires_capabilities: list[str]
    max_duration: int


class BenchmarkConfigurationFile:
    include: list[str]
    benchmarks: dict[str, BenchmarkConfiguration]


BenchmarkConfigurations = dict[str, BenchmarkConfiguration]


class Configuration:
    # Meta configuration, impacts how benchmark are executed globally
    system: SystemConfig

    # Specify how to run the benchmarks
    benchmarks: BenchmarkConfigurations

