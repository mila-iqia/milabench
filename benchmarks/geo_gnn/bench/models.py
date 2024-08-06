from types import SimpleNamespace as NS

from torch_geometric.nn.models import PNA as _PNA, DimeNet as _DimeNet

models = {}


def register_model(fn):
    models[fn.__name__] = fn
    return fn


@register_model
def DimeNet(args):
    return NS(
        category="3d",
        model=_DimeNet(
            hidden_channels=64,
            out_channels=1,
            num_blocks=6,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            cutoff=10.0,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
        ),
    )


@register_model
def PNA(args):
    return NS(
        category="2d",
        model=_PNA(
            hidden_channels=64,
            # TODO: Need help to know what parameters to use.
        ),
    )
