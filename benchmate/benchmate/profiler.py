import os
from contextlib import contextmanager

from .toggles import _get_flag


_DEFAULT_TRACE_DIR = "/tmp/milabench-trace"


def is_profiling_enabled():
    return _get_flag("MILABENCH_PROFILE", int, 0) != 0


def _get_trace_dir():
    return _get_flag("MILABENCH_PROFILE_DIR", str, _DEFAULT_TRACE_DIR)


@contextmanager
def jax_profiler(trace_dir=None, enabled=None):
    """Context manager that optionally activates JAX's XLA profiler trace.

    When enabled, produces a trace viewable in TensorBoard or Perfetto
    (ui.perfetto.dev). When disabled, this is a complete no-op.

    Named scopes (``jax.named_scope``) placed in the training loop are
    always free; they only become visible when a trace is active.

    Enable via:
      - ``enabled=True`` argument, or
      - ``MILABENCH_PROFILE=1`` environment variable

    The trace directory defaults to ``/tmp/milabench-trace`` and can be
    overridden with ``MILABENCH_PROFILE_DIR``.
    """
    if enabled is None:
        enabled = is_profiling_enabled()

    if not enabled:
        yield
        return

    import jax

    if trace_dir is None:
        trace_dir = _get_trace_dir()

    os.makedirs(trace_dir, exist_ok=True)
    print(f"[profiler] Tracing to {trace_dir}")

    with jax.profiler.trace(trace_dir):
        yield
