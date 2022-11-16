
import numpy as np

from nn.basis import Module, Parameter, Variable


class Linear(Module):
    in_features: int
    out_features: int
    weight: Parameter
    bias: Parameter
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            data=np.random.rand(in_features, out_features))
        self.bias = Parameter(data=np.zeros((1, out_features)))

    def forward(self, input: Variable) -> Variable:
        '''
        input: [N, ..., H_in]
        output: [N, ..., H_out]
        '''
        return input @ self.weight.variable + self.bias.variable

