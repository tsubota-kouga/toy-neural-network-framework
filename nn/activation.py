
import numpy as np

from nn.basis import Module, Variable


class ReLU(Module):
    def forward(self, input: Variable) -> Variable:
        return Variable.where(input.data > 0., input, 0.)

class Sigmoid(Module):
    def forward(self, input: Variable) -> Variable:
        self.data_args = [input]
        data = Variable(1 / (1 + np.exp(-input.data)))
        data.node = self
        self.outputs = [data]
        return data

    def backward(self, dydx: Variable) -> list[Variable]:
        assert self.outputs is not None
        assert len(self.outputs) == 1
        output, = self.outputs
        assert isinstance(output, Variable)
        return [Variable(data=output.data * (1 - output.data) * dydx.data)]

class Step(Module):
    def forward(self, input: Variable) -> Variable:
        self.data_args = [input]
        data = Variable(np.where(input.data > 0., 1., 0.))
        data.node = self
        self.outputs = [data]
        return data

    def backward(self, dydx: Variable) -> list[Variable]:
        return [dydx]
