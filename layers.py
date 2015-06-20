"""
Very simple neural network framework
"""
import numpy as np

class LinearLayer():
    def __init__(self, num_inputs, num_units, scale=0.01):
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.W = np.random.random((num_inputs, num_units)) * scale
        self.b = np.zeros(num_units)

    def __str__(self):
        return "LinearLayer(%i, %i)" % (self.num_inputs, self.num_units)

    def fprop(self, x, train=True):
        self.x = x
        self.a = np.dot(x, self.W) + self.b
        return self.a

    def bprop(self, delta_in):
        x_t = np.transpose(self.x)
        self.grad_W = np.dot(x_t, delta_in)
        self.grad_b = delta_in.sum(axis=0)

        W_T = np.transpose(self.W)
        self.delta_out = np.dot(delta_in,W_T)
        return self.delta_out

    def update_params(self, lr):
        self.W = self.W - self.grad_W*lr
        self.b = self.b - self.grad_b*lr


class SigmoidActivationLayer():
    def __str__(self):
        return "Sigmoid()"

    def fprop(self, x, train=True):
        self.a = 1.0 / (1+np.exp(-x))
        return self.a

    def bprop(self, delta_in):
        delta_out = self.a * (1 - self.a)*delta_in
        return delta_out

    def update_params(self, lr):
        pass


class ReluActivationLayer():
    def __str__(self):
        return "ReLU()"

    def fprop(self, x, train=True):
        self.a = np.maximum(0, x)
        return self.a

    def bprop(self, delta_in):
        return delta_in * (self.a > 0).astype(self.a.dtype)

    def update_params(self, lr):
        pass


class IdentityLayer():
    def __str__(self):
        return "Identity()"

    def fprop(self, x, train=True):
        return x

    def bprop(self, delta_in):
        return delta_in

    def update_params(self, lr):
        pass


class DropoutLayer():
    def __str__(self):
        return "Dropout(%f)" % self.p

    def __init__(self, p=0.5):
        self.p = p

    def fprop(self, x, train=True):
        if train:
            mask = np.random.random(x.shape) > (1-self.p)
            self.a = x*mask
            return self.a
        else:
            scale = 1.0 / self.p  # p=1 drop nothing -> no scaling
            return self.a * scale

    def bprop(self, delta_in):
        delta_out = delta_in*self.a
        return delta_out

    def update_params(self, lr):
        pass


class SoftmaxActivationLayer():
    def __str__(self):
        return "Softmax()"

    def fprop(self, x, train=True):
        x_exp = np.exp(x)
        normalizer = x_exp.sum(axis=-1, keepdims=True)
        self.a = x_exp / normalizer
        return self.a

    def bprop(self, delta_in):
        return delta_in

    def update_params(self, lr):
        pass


class AddLayer():
    def __str__(self):
        return "AddLayer()"

    def fprop(self, x1, x2, train=True):
        # f = x1 + x2
        self.a = x1 + x2
        return self.a

    def bprop(self, delta_in):
        return [delta_in, delta_in]

    def update_params(self, lr):
        pass


class MulLayer():
    def __str__(self):
        return "MulLayer()"

    def fprop(self, x1, x2, train=True):
        self.a1 = x1
        self.a2 = x2
        return self.a1 * self.a2

    def bprop(self, delta_in):
        # f = x1 * x2
        return [delta_in*self.a2, delta_in*self.a1]

    def update_params(self, lr):
        pass


class MeanSquaredLoss():
    def __str__(self):
        return "MeanSquaredLoss()"

    def fprop(self, x, t):
        num_batches = x.shape[0]
        cost = 0.5 * (x-t)**2 / num_batches
        self.a = np.mean(np.sum(cost, axis=-1))
        return self.a

    def bprop(self, y, t):
        delta_out = y-t
        return delta_out

    def update_params(self):
        pass


class CrossEntropyLoss():
    def __str__(self):
        return "CrossEntropyLoss()"

    def fprop(self, x, t):
        tol = 1e-8
        self.a = np.mean(np.sum(-t * np.log(x + tol), axis=-1))
        return self.a

    def bprop(self, y, t):
        delta_out = y-t
        return delta_out

    def update_params(self):
        pass
