from typing import TypeVar, Callable
try:
    from typing import ParamSpec  # type: ignore
except ImportError:
    from typing_extensions import ParamSpec
import inspect
import functools

P = ParamSpec("P")
T = TypeVar("T")

# NOTE: Making `fixed_args` and `**fixed_kwargs` below with P.args and P.kwargs makes the type
# checkers give an error when we don't pass ALL the parameters, which isn't exactly what I want
# here.
# What I want is some way of indicating that, when passed, these need to be valid args or kwargs to
# the callable `cls`.
def with_fixed_constructor_arguments(
    cls: Callable[P, T], *fixed_args, **fixed_kwargs,
) -> Callable[P, T]:
    """ Returns a callable that fixes some of the arguments to the type or callable `cls`.
    """
    if not fixed_args and not fixed_kwargs:
        # Not fixing any arguments, return the callable as-is.
        return cls
    # NOTE: There's apparently no need to pass cls.__init__ for classes. So we can do the same for
    # either classes or functions.
    init_signature = inspect.signature(cls)
    try:
        bound_fixed_args = init_signature.bind_partial(*fixed_args, **fixed_kwargs)
    except TypeError as err:
        raise TypeError(f"Unable to bind fixed values for {cls}: {err}") from err

    @functools.wraps(cls)
    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        bound_args = init_signature.bind_partial(*args, **kwargs)
        for argument_name, fixed_value in bound_fixed_args.arguments:
            print(
                f"Ignoring value {bound_args.arguments[argument_name]} for argument "
                f"{argument_name}, using fixed value of {fixed_value} instead."
            )
        bound_args.arguments.update(bound_fixed_args.arguments)
        bound_args.apply_defaults()
        return cls(*bound_args.args, **bound_args.kwargs)

    return _wrapped

