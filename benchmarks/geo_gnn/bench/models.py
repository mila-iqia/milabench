from types import SimpleNamespace as NS

from torch_geometric.nn.models import PNA as _PNA, DimeNet as _DimeNet

models = {}


def register_model(fn):
    models[fn.__name__] = fn
    return fn


@register_model
def DimeNet(args, **extras):
    return NS(
        category="3d",
        model=_DimeNet(
            hidden_channels=64,
            out_channels=2,
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
def PNA(args, degree):
    return NS(
        category="2d",
        model=_PNA(
            # Basic GCNN setup
            in_channels=1, 
            # out_channels=75,
            # edge_dim=50, 
            towers=5, 
            # pre_layers=1, 
            # post_layers=1,
            hidden_channels=75,
            num_layers=64,
            # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PNAConv.html
            aggregators=['mean', 'min', 'max', 'std'],
            scalers=['identity', 'amplification', 'attenuation'],
            # Histogram of in-degrees of nodes in the training set, used by scalers to normalize
            deg=degree(),
        ),
    )
