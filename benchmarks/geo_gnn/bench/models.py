from types import SimpleNamespace as NS

from torch_geometric.nn.models import PNA as _PNA, DimeNet as _DimeNet

models = {}


def register_model(fn):
    models[fn.__name__] = fn
    return fn


@register_model
def DimeNet(args, sample, **extras):
    # The directional message passing neural network (DimeNet) from the “Directional Message Passing for Molecular Graphs” paper. 
    # DimeNet transforms messages based on the angle between them in a rotation-equivariant fashion.
    
    # PCQM4Mv2Subset: Data(x=[18, 9], edge_index=[2, 40], edge_attr=[40, 3], y=3.0476751256, pos=[18, 3], smiles='Cc1ccc([C@H]2[CH]c3cnccc3[N]C2=O)cc1')
    # QM9:            Data(x=[5, 11], edge_index=[2, 8], edge_attr=[8, 4], y=[1, 19], pos=[5, 3], z=[5], smiles='[H]C([H])([H])[H]', name='gdb_1', idx=[1])
    try:
        batch_size, out_channels = sample.y.shape
    except:
        out_channels = 1
    
    return NS(
        category="3d",
        model=_DimeNet(
            hidden_channels=64,
            out_channels=out_channels,
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
def PNA(args, sample, degree):
    # The Graph Neural Network from the “Principal Neighbourhood Aggregation for Graph Nets” paper, 
    # using the PNAConv operator for message passing.
    
    out_channels = 1
    if hasattr(sample.y, "shape") and len(sample.y.shape) > 1:
        out_channels = sample.y.shape[-1]

    _, in_channels = sample.x.shape

    return NS(
        category="2d",
        model=_PNA(
            # Basic GCNN setup
            in_channels=in_channels, 
            out_channels=out_channels,
            hidden_channels=64,
            num_layers=64,
            # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PNAConv.html
            aggregators=['mean', 'min', 'max', 'std'],
            scalers=['identity', 'amplification', 'attenuation'],
            # Histogram of in-degrees of nodes in the training set, used by scalers to normalize
            deg=degree(),
        ),
    )
