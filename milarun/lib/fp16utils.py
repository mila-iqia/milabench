import torch.nn as nn


class OptimizerAdapter:
    def __init__(self, optimizer, half=False, *args, **kwargs):
        if half:
            import apex.fp16_utils.fp16_optimizer as apex_optimizer
            self.optimizer = apex_optimizer.FP16_Optimizer(optimizer, *args, **kwargs)
        else:
            self.optimizer = optimizer

        self.half = half

    def backward(self, loss):
        if loss is None:
            raise RuntimeError('')

        if self.half:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        return self.optimizer

    def step(self):
        return self.optimizer.step()

    def zero_grad(self):
        return self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class ModelAdapter(nn.Module):
    def __init__(self, model, half=False):
        super(ModelAdapter, self).__init__()

        self.model = model
        self.transform = lambda x: x

        if half:
            from apex.fp16_utils import network_to_half
            self.model = network_to_half(model)
            self.transform = lambda x: x.half()

    def forward(self, input):
        return self.model(self.transform(input))
