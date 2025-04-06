import math
import numpy as np

DELTA_IN = 1e-5


class Value:

    def __init__(self, data, _children=(), _op="", label="", delta_in=DELTA_IN):
        self.data = data
        self.grad = 0.0
        self.delta_in = self.data + delta_in
        self.delta_out = 0.0
        self.eta = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(data=self.data + other.data, _children=(self, other), _op="+")
        out.delta_in = self.delta_in + other.delta_in

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            self.delta_out = out.delta_out
            other.delta_out = out.delta_out

        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        out.delta_in = self.delta_in * other.delta_in

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            self.delta_out = out.delta_out
            other.delta_out = out.delta_out

        out._backward = _backward

        return out

    def tanh(self):

        def _func(x):
            t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
            return t

        t = _func(self.data)
        out = Value(t, (self,), "tanh")
        out.delta_in = _func(self.delta_in)

        def _backward():
            self.grad += (1 - t**2) * out.grad
            self.delta_out = out.delta_out

        out._backward = _backward

        return out

    def backward(self):

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node.delta_out = self.delta_in
            node._backward()
