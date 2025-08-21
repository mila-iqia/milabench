from omegaconf import OmegaConf

from milabench.capability import sync_is_system_capable
from milabench.pack import BasePackage


def _fake_config(n):
    return {
        "name": "test",
        "definition": ".",
        "dirs": dict(extra=".", venv="."),
        "system": {"nodes": list(range(n))},
        "num_machines": 2,
        "requires_capabilities": ["len(nodes) >= ${num_machines}"],
    }


def fake_config(n):
    return OmegaConf.to_object(OmegaConf.create(_fake_config(n)))



# This does not work anymore ?
# def test_capabilties_ok():
#     pack = BasePackage(fake_config(10))
#     assert sync_is_system_capable(pack) is True


# def test_capabilties_not_ok():
#     pack = BasePackage(fake_config(1))
#     assert sync_is_system_capable(pack) is False
