import numpy as np
from argparse import ArgumentParser
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import count
from coleo import Argument, default
from milarun.lib import init_torch, coleo_main, iteration_wrapper, dataloop


def generate_wave_data(T, L, N):
    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')
    return data


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

        self.device = next(self.parameters()).device
        self.dtype = next(self.parameters()).dtype

    def to(self, device=None, dtype=None):
        model = super().to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        return model

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51).to(device=self.device, dtype=self.dtype)
        c_t = torch.zeros(input.size(0), 51).to(device=self.device, dtype=self.dtype)
        h_t2 = torch.zeros(input.size(0), 51).to(device=self.device, dtype=self.dtype)
        c_t2 = torch.zeros(input.size(0), 51).to(device=self.device, dtype=self.dtype)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


to_type = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
}


@coleo_main
def main(exp):
    # Model float type
    dtype: Argument & str = default("float32")

    # Number of samples
    samples: Argument & int = default(100)

    torch_settings = init_torch()
    device = torch_settings.device

    data = generate_wave_data(20, 1000, samples)

    _dtype = to_type[dtype]

    input = torch.from_numpy(data[3:, :-1]).to(device=device, dtype=_dtype)
    target = torch.from_numpy(data[3:, 1:]).to(device=device, dtype=_dtype)

    test_input = torch.from_numpy(data[:3, :-1]).to(device=device, dtype=_dtype)
    test_target = torch.from_numpy(data[:3, 1:]).to(device=device, dtype=_dtype)

    # build the model
    seq = Sequence().to(device=device, dtype=_dtype)
    criterion = nn.MSELoss().to(device=device, dtype=_dtype)

    optimizer = optim.SGD(seq.parameters(), lr=0.01)

    total_time = 0

    seq.train()

    wrapper = iteration_wrapper(exp, sync=torch_settings.sync)

    for it, _ in dataloop(count(), wrapper=wrapper):
        it.set_count(samples)

        def closure():
            optimizer.zero_grad()
            out = seq(input.to(device=device, dtype=_dtype))
            loss = criterion(out, target)
            loss.backward()
            it.log(loss=loss.item())
            return loss

        optimizer.step(closure)
