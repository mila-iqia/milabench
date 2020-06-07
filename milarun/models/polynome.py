from milarun.lib import init_torch, coleo_main, dataloop, iteration_wrapper
from itertools import repeat
from coleo import auto_cli, Argument, default
import torch
import torch.nn.functional as F
import torch.nn as nn


@coleo_main
def main(exp):
    torch_settings = init_torch()

    # Degree of the polynomial
    poly_degree: Argument & int = default(4)

    # Number of examples per batch
    batch_size: Argument & int = default(64)

    torch_settings = init_torch()
    device = torch_settings.device

    W_target = torch.randn(poly_degree, 1) * 5
    b_target = torch.randn(1) * 5

    def make_features(x):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        x = x.unsqueeze(1)
        return torch.cat([x ** i for i in range(1, poly_degree + 1)], 1)

    def f(x):
        """Approximated function."""
        return x.mm(W_target) + b_target.item()

    def poly_desc(W, b):
        """Creates a string description of a polynomial."""
        result = 'y = '
        for i, w in enumerate(W):
            result += '{:+.2f} x^{} '.format(w, len(W) - i)
        result += '{:+.2f}'.format(b[0])
        return result

    def get_batch():
        """Builds a batch i.e. (x, f(x)) pair."""
        random = torch.randn(batch_size)
        x = make_features(random)
        y = f(x)
        return x, y

    def dataset():
        while True:
            yield get_batch()

    # Define model
    fc = torch.nn.Linear(W_target.size(0), 1)
    fc.to(device)

    wrapper = iteration_wrapper(exp, sync=torch_settings.sync)

    for it, (batch_x, batch_y) in dataloop(dataset(), wrapper=wrapper):
        it.set_count(batch_size)

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Reset gradients
        fc.zero_grad()

        # Forward pass
        output = F.smooth_l1_loss(fc(batch_x), batch_y)
        loss = output.item()

        it.log(loss=loss)

        # Backward pass
        output.backward()

        # Apply gradients
        for param in fc.parameters():
            param.data.add_(-0.01 * param.grad.data)

    print('==> Learned function:\t',
          poly_desc(fc.weight.view(-1), fc.bias))
    print('==> Actual function:\t',
          poly_desc(W_target.view(-1), b_target))
