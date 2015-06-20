import numpy as np


class RNNLayer():
    def __init__(self, num_inputs, num_units, activation,
                 W_in=None, b_in=None, W_hid=None, b_hid=None):
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.activation = activation

        if W_in is None and b_in is None:
            self.in_layer = LinearLayerRNN(num_inputs, num_units)
        else:
            self.in_layer = LinearLayerRNN(
                num_inputs, num_units, W=W_in, b=b_in)

        if W_hid is None and b_hid is None:
            self.hid_layer = LinearLayerRNN(num_units, num_units)
        else:
            self.hid_layer = LinearLayerRNN(
                num_units, num_units, W=W_hid, b=b_hid)

    def __str__(self):
        return "RNNLayer(%i, %i)" % (self.num_inputs, self.num_units)

    def fprop(self, x, h_prev, train=True):
        in_a = self.in_layer.fprop(x)
        hid_a = self.hid_layer.fprop(h_prev)
        self.a = self.activation.fprop(in_a + hid_a)
        return self.a

    def bprop(self, delta_in_x, delta_in_hid):
        delta_in = self.activation.bprop(delta_in_x + delta_in_hid)
        delta_out_x = self.in_layer.bprop(delta_in)
        delta_out_hid = self.hid_layer.bprop(delta_in)

        return delta_out_hid

    def get_grads(self):
        return self.in_layer.get_grads() + self.hid_layer.get_grads()



    def update_params(self, W_in_grad, b_in_grad, W_hid_grad, b_hid_grad, lr):
        self.in_layer.update_params(W_in_grad, b_in_grad, lr)
        self.hid_layer.update_params(W_hid_grad, b_hid_grad, lr)

    def get_params(self):
        return self.in_layer.get_params() + self.hid_layer.get_params()


class LinearLayerRNN():
    def __init__(self, num_inputs, num_units, scale=0.01, W=None, b=None, name=""):
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.name = name

        if W is None:
            self.W = np.random.random((num_inputs, num_units)) * scale
        else:
            assert W.shape == (num_inputs, num_units)
            self.W = W

        if b is None:
            self.b = np.zeros(num_units)
        else:
            assert b.shape == (num_units, )
            self.b = b

    def __str__(self):
        return "LinearLayer%s(%i, %i)" % (
            self.name, self.num_inputs, self.num_units)

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


    # note that this takes gradients as input because we need to sum over
    # all timesteps
    def update_params(self, grad_W, grad_b, lr):
        self.W -= grad_W*lr
        self.b -= grad_b*lr

    def get_params(self):
        return [self.W, self.b]

    def get_grads(self):
        return [self.grad_W, self.grad_b]

    def reset_grads(self):
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)