
from nn.basis import Parameter


class Optimizer:
    lr: float

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    lr: float
    params: list[Parameter]
    def __init__(self, params: list[Parameter], lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.variable.grad is not None:
                param.variable.data -= self.lr * param.variable.grad.data

