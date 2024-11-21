from torch_geometric.nn.models import PNA as _PNA, DimeNet as _DimeNet

import torch

from benchmate.models import model_size


print(model_size(_DimeNet(
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
)

))

print(model_size(
_PNA(
    # Basic GCNN setup
    in_channels=1, 
    out_channels=1,
    hidden_channels=64,
    num_layers=64,
    # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.PNAConv.html
    aggregators=['mean', 'min', 'max', 'std'],
    scalers=['identity', 'amplification', 'attenuation'],
    # Histogram of in-degrees of nodes in the training set, used by scalers to normalize
    deg=torch.tensor(4),
)))