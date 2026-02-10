from copy import deepcopy

from .syslog import syslog


async def _failure(pack, condition):
    msg = f"Skip {pack.config['name']} because the following capability is not satisfied: {condition}"
    await pack.message(msg)



def is_system_capable_with_reasons(pack) -> bool:
    # eval add __builtins__ to the dictionary, we copy it to not
    # spoil our beautiful config
    capability_context = deepcopy(pack.config["system"])
    is_compatible = True
    whys = []

    for condition in pack.config.get("requires_capabilities", []):
        if not eval(condition, capability_context):
            is_compatible = False
            whys.append(condition)

    return is_compatible, whys


def is_system_capable(pack) -> bool:
    is_compatible, whys = is_system_capable_with_reasons(pack)

    for reason in whys:
        syslog(
            "Skip {name} because the following capability is not satisfied: {condition}", 
            name=pack.config['name'],
            condition=reason
        )

    return is_compatible


async def is_system_capable_report(pack) -> bool:
    is_compatible, whys = is_system_capable_with_reasons(pack)

    for reason in whys:
        await _failure(pack, reason)

    return is_compatible
