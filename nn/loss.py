
from nn.basis import Module, Variable


class BCELoss(Module):
    def forward(self, input: Variable, target: Variable) -> Variable:
        return ((-target * input.log()) - (1. - target) * (1. - input).log()).mean(axis=0)

class BCEWithLogitsLoss(Module):
    def forward(self, input: Variable, target: Variable) -> Variable:
        return ((-target * (-(1 + (-input).exp())).log()) - (1. - target) * (-input - (1 + (-input).exp()).log())).mean(axis=0)

class MSELoss(Module):
    def forward(self, input: Variable, target: Variable) -> Variable:
        return ((input - target) ** 2).mean(axis=0)
