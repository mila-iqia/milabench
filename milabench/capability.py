from copy import deepcopy

async def _failure(pack, condition):
    msg = f"Skip {pack.config['name']} because the following capability is not satisfied: {condition}"
    await pack.message(msg)


async def is_system_capable(pack):
    # eval add __builtins__ to the dictionary, we copy it to not
    # spoil our beautiful config
    capability_context = deepcopy(pack.config["system"])
    is_compatible = True

    for condition in pack.config.get("requires_capabilities", []):
        if not eval(condition, capability_context):
            await _failure(pack, condition)
            is_compatible = False

    return is_compatible


def sync_is_system_capable(pack):
    import asyncio

    loop = asyncio.get_event_loop()
    task = loop.create_task(is_system_capable(pack))
    return loop.run_until_complete(task)
