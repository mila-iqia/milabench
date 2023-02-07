from contextlib import contextmanager, ExitStack
import importlib
import pkgutil



def discover_validation_layers(module):
    """Discover validation layer inside the milabench.validation module"""
    path = module.__path__
    name = module.__name__

    layers = {}

    for _, name, _ in pkgutil.iter_modules(path, name + "."):
        layers[name] = importlib.import_module(name)

    # Remove the interface definition
    layers.pop('validation')
    
    return layers


VALIDATION_LAYERS = discover_validation_layers()


@contextmanager
def validation(gv, *layer_names):
    """Combine validation layers into a single context manager"""
    results = dict()

    with ExitStack() as stack:

        for layer_name in layer_names:
            layer = VALIDATION_LAYERS.get(layer_name)

            if layer is not None:
                results[layer_name] = stack.enter_context(layer(gv))
            else:
                names = list(VALIDATION_LAYERS.keys())
                raise RuntimeError(f"Layer `{layer_name}` does not exist: {names}")

        yield results
