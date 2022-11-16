
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Union
import numpy as np

DataType = Union[int, float, np.ndarray]

EPS = 1e-8


@dataclass
class Node:
    data_args: list[Variable]
    outputs: list[Variable]
    args: list[Any] = field(default_factory=list)

    def backward(self, dydx: Variable) -> list[Variable]:
        raise NotImplementedError

class NegNode(Node):
    def backward(self, dydx: Variable) -> list[Variable]:
        return [-dydx]

class AddNode(Node):
    def backward(self, dydx: Variable) -> list[Variable]:
        return [dydx, dydx]

class SubNode(Node):
    def backward(self, dydx: Variable) -> list[Variable]:
        return [dydx, Variable(-dydx.data)]

class MulNode(Node):
    def backward(self, dydx: Variable) -> list[Variable]:
        assert len(self.data_args) == 2
        a, b = self.data_args
        return [Variable(b.data * dydx.data), Variable(a.data * dydx.data)]

class DivNode(Node):
    def backward(self, dydx: Variable) -> list[Variable]:
        assert len(self.data_args) == 2
        a, b = self.data_args
        return [
            Variable(dydx.data / (b.data + EPS)),
            Variable(-a.data * dydx.data / (b.data ** 2 + EPS))]

class PowNode(Node):
    def backward(self, dydx: Variable) -> list[Variable]:
        assert len(self.data_args) == 2
        a, b = self.data_args
        return [
            Variable((b.data - 1) * (a.data ** (b.data - 1)) * dydx.data),
            Variable((a.data ** b.data) * np.log(a.data + EPS) * dydx.data)]

class ExpNode(Node):
    def backward(self, dydx: Variable) -> list[Variable]:
        assert len(self.outputs) == 1
        o, = self.outputs
        return [Variable(o.data * dydx.data)]

class LogNode(Node):
    def backward(self, dydx: Variable) -> list[Variable]:
        assert len(self.outputs) == 1
        a, = self.data_args
        return [Variable(dydx.data / (a.data + EPS))]

class SumNode(Node):
    def backward(self, dydx: Variable) -> list[Variable]:
        assert len(self.outputs) == 1
        a, = self.data_args
        assert len(self.args) == 1
        axis, = self.args
        dim = a.data.shape[axis]
        return [
            Variable(
                np.expand_dims(dydx.data, axis=axis)
                  .repeat(dim, axis=axis))]

class WhereNode(Node):
    def backward(self, dydx: Variable) -> list[Variable]:
        assert len(self.args) == 1
        condition, = self.args
        return [
            Variable(np.where(condition, dydx.data, 0.)),
            Variable(np.where(condition, 0., dydx.data))]

class MatMulNode(Node):
    def backward(self, dydx: Variable) -> list[Variable]:
        assert len(self.data_args) == 2
        a, b = self.data_args
        return [Variable(dydx.data @ b.data.T), Variable(a.data.T @ dydx.data)]


class Variable:
    data: np.ndarray
    grad: Optional[Variable]
    node: Optional[Union[Node, Module]]

    def __init__(self, data: np.ndarray, node: Optional[Node]=None):
        self.data = data
        self.grad = None
        self.node = node

    @staticmethod
    def wrap(data: Union[DataType, Variable]) -> Variable:
        if type(data) is int or type(data) is float:
            return Variable(data=np.array(data))
        elif isinstance(data, np.ndarray):
            return Variable(data=data)
        elif isinstance(data, Variable):
            return data
        else:
            raise ValueError

    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __pos__(self) -> Variable:
        return self

    def __neg__(self) -> Variable:
        data = Variable(-self.data)
        neg_node = NegNode(data_args=[self], outputs=[data])
        data.node = neg_node
        return data

    def __add__(self, other: Union[DataType, Variable]) -> Variable:
        other = Variable.wrap(other)
        data = Variable(self.data + other.data)
        add_node = AddNode(data_args=[self, other], outputs=[data])
        data.node = add_node
        return data

    def __radd__(self, other: Union[DataType, Variable]) -> Variable:
        return self.__add__(other)

    def __sub__(self, other: Union[DataType, Variable]) -> Variable:
        return self.__add__(-other)

    def __rsub__(self, other: Union[DataType, Variable]) -> Variable:
        return (-self).__add__(other)

    def __mul__(self, other: Union[DataType, Variable]) -> Variable:
        other = Variable.wrap(other)
        data = Variable(self.data * other.data)
        mul_node = MulNode(data_args=[self, other], outputs=[data])
        data.node = mul_node
        return data

    def __rmul__(self, other: Union[DataType, Variable]) -> Variable:
        return self.__mul__(other)

    def __truediv__(self, other: Union[DataType, Variable]) -> Variable:
        other = Variable.wrap(other)
        data = Variable(self.data / (other.data + EPS))
        div_node = DivNode(data_args=[self, other], outputs=[data])
        data.node = div_node
        return data

    def __rtruediv__(self, other: Union[DataType, Variable]) -> Variable:
        other = Variable.wrap(other)
        data = Variable(other.data / (self.data + EPS))
        div_node = DivNode(data_args=[other, self], outputs=[data])
        data.node = div_node
        return data

    def __pow__(self, other: Union[DataType, Variable]) -> Variable:
        other = Variable.wrap(other)
        data = Variable(self.data ** other.data)
        pow_node = PowNode(data_args=[self, other], outputs=[data])
        data.node = pow_node
        return data

    def exp(self) -> Variable:
        data = Variable(np.exp(self.data))
        exp_node = ExpNode(data_args=[self], outputs=[data])
        data.node = exp_node
        return data

    def log(self) -> Variable:
        data = Variable(np.log(self.data + EPS))
        exp_node = LogNode(data_args=[self], outputs=[data])
        data.node = exp_node
        return data

    def sum(self, axis: int) -> Variable:
        data = Variable(self.data.mean(axis=axis))
        sum_node = SumNode(data_args=[self], outputs=[data], args=[axis])
        data.node = sum_node
        return data

    def mean(self, axis: int) -> Variable:
        dim = self.data.shape[axis]
        return self.sum(axis=axis) / dim

    def __matmul__(self, other: Union[DataType, Variable]) -> Variable:
        other = Variable.wrap(other)
        data = Variable(self.data @ other.data)
        matmul_node = MatMulNode(data_args=[self, other], outputs=[data])
        data.node = matmul_node
        return data

    @staticmethod
    def where(condition: np.ndarray, a: Union[DataType, Variable], b: Union[DataType, Variable]) -> Variable:
        a, b = Variable.wrap(a), Variable.wrap(b)
        data = Variable(np.where(condition, a.data, b.data))
        where_node = WhereNode(data_args=[a, b], outputs=[data], args=[condition])
        data.node = where_node
        return data


    def backward(self, dydx: Optional[Union[DataType, Variable]] = None):
        if dydx is None:
            dydx = Variable(np.ones_like(self.data))
        else:
            dydx = Variable.wrap(dydx)
        if self.shape() != dydx.shape():
            num_expanded_dim = len(dydx.shape()) - len(self.shape())
            grad_: np.ndarray = dydx.data.sum(axis=tuple(np.arange(num_expanded_dim)))
            axes = []
            for i, (self_d, dydx_d) in enumerate(zip(self.shape(), dydx.shape())):
                if self_d != dydx_d:
                    assert self_d == 1
                    axes.append(i)
            grad_ = grad_.sum(axis=tuple(axes))
            grad = Variable(grad_)
        else:
            grad = dydx
        if self.grad is None:
            self.grad = grad
        else:
            self.grad.data += grad.data
        if self.node is not None:
            if isinstance(self.node, Node):
                diff = self.node.backward(dydx)
                for a, d in zip(self.node.data_args, diff):
                    a.backward(d)
            else:  # Module, self implemented backward
                diff = self.node.backward(dydx)
                assert self.node.data_args is not None
                for a, d in zip(self.node.data_args, diff):
                    a.backward(d)
        else:
            pass  # get to a leaf node

    def zero_grad(self):
        self.grad = None
        if self.node is not None and self.node.data_args is not None:
            for variable in self.node.data_args:
                variable.zero_grad()
                variable.node = None
            self.node = None


class Parameter:
    variable: Variable
    def __init__(self, data: Union[Variable, np.ndarray]):
        self.variable = Variable.wrap(data)


class ParameterList:
    params: list[Parameter]
    def __init__(self, data: list[Union[Variable, np.ndarray]]):
        for _data in data:
            self.params.append(
                    Parameter(Variable.wrap(_data)))

    def __iter__(self) -> Iterator[Parameter]:
        return self.params.__iter__()


class Module:
    data_args: Optional[list[Variable]] = None
    outputs: Optional[list[Variable]] = None

    def forward(self, *inputs: Any) -> Any:
        raise NotImplementedError

    def backward(self, *outputs: Any) -> list[Variable]:
        raise NotImplementedError

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        for attr in vars(self).values():
            if isinstance(attr, Parameter):
                params.append(attr)
            elif isinstance(attr, ParameterList):
                for param in attr:
                    params.append(param)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, ModuleList):
                for module in attr:
                    params.extend(module.parameters())
        return params


class ModuleList:
    modules: list[Module]
    def __init__(self, *modules: Module):
        self.modules = list(modules)

    def __iter__(self) -> Iterator[Module]:
        return self.modules.__iter__()


class Sequential(Module):
    modules: ModuleList
    def __init__(self, *modules: Module):
        self.modules = ModuleList(*modules)

    def forward(self, input: Variable) -> Variable:
        x = input
        for module in self.modules:
            x = module.forward(x)
            assert isinstance(x, Variable)
        return x

