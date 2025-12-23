import os



def _get_flag(name, type, default):
    return type(os.getenv(name, default))


def get_poll_interval(value):
    return _get_flag("BENCHMATE_POLL_INTERVAL", float, value)


def get_observation_count(value):
    return _get_flag("BENCHMATE_OBSERVATION_COUNT", int, value)


poll_interval_default = get_poll_interval(0.25)


log_pattern = _get_flag("BENCHMATE_LOG_MODE", str, 'lean')
