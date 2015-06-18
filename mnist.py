import numpy as np
from confusionmatrix import ConfusionMatrix


def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out


# Load Mnist data and convert vector represention to one-hot
data = np.load('mnist.npz')
num_classes = 10
x_train = data['X_train']
targets_train = data['y_train']
targets_train = onehot(targets_train, num_classes)


class LinearLayer():
    def __init__(self, num_inputs, num_units, scale=0.01):
        self.W = np.random.random((num_inputs, num_units)) * scale

    def fprop(self, x):
        self.x = x
        self.a = np.dot(x, self.W)
        return self.a

    def bprop(self, delta_in):
        x_t = np.transpose(self.x)
        self.grad = np.dot(x_t, delta_in)

        W_T = np.transpose(self.W)
        self.delta_out = np.dot(delta_in,W_T)
        return self.delta_out

    def update_params(self, lr):
        self.W = self.W - self.grad*lr


class SigmoidActivationLayer():
    def fprop(self, x):
        self.a = 1.0 / (1+np.exp(-x))
        return self.a

    def bprop(self, delta_in):
        delta_out = self.a * (1 - self.a)*delta_in
        return delta_out

    def update_params(self, lr):
        pass


class ReluActivationLayer():
    def fprop(self, x):
        self.a = np.maximum(0, x)
        return self.a

    def bprop(self, delta_in):
        return delta_in * (self.a > 0).astype(self.a.dtype)

    def update_params(self, lr):
        pass


class IdentityLayer():
    def fprop(self, x):
        return x

    def bprop(self, delta_in):
        return delta_in

    def update_params(self, lr):
        pass

class DropoutLayer():
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
    def fprop(self, x):
        x_exp = np.exp(x)
        normalizer = x_exp.sum(axis=-1, keepdims=True)
        self.a = x_exp / normalizer
        return self.a

    def bprop(self, delta_in):
        return delta_in

    def update_params(self, lr):
        pass


class MeanSquaredLoss():
    def fprop(self, x, t):
        num_batches = x.shape[0]
        cost = 0.5 * (x-t)**2 / num_batches
        return np.sum(cost)

    def bprop(self, y, t):
        delta_out = y-t
        return delta_out

    def update_params(self):
        pass


class CrossEntropyLoss():
    def fprop(self, x, t):
        tol = 1e-8
        return np.sum(t * np.log(x + tol))

    def bprop(self, y, t):
        delta_out = y-t
        return delta_out

    def update_params(self):
        pass

num_samples, num_inputs = x_train.shape
num_hidden_units = 100

l_hid_pre1 = LinearLayer(num_inputs, num_hidden_units)
l_hid_act1 = ReluActivationLayer()
l_drp      = IdentityLayer()
l_hid_pre2 = LinearLayer(num_hidden_units, num_classes)
l_hid_act2 = SoftmaxActivationLayer()

LossLayer = CrossEntropyLoss()


def forward(x):
    out_hid_pre1 = l_hid_pre1.fprop(x)
    out_hid_act1 = l_hid_act1.fprop(out_hid_pre1)
    out_drp      = l_drp.fprop(out_hid_act1)
    out_hid_pre2 = l_hid_pre2.fprop(out_drp)
    y_probs = l_hid_act2.fprop(out_hid_pre2)
    return y_probs


def backward(y_probs, targets):
    delta1 = LossLayer.bprop(y_probs, targets)
    delta2 = l_hid_act2.bprop(delta1)
    delta3 = l_hid_pre2.bprop(delta2)
    delta_drop = l_drp.bprop(delta3)
    delta4 = l_hid_act1.bprop(delta_drop)
    delta5 = l_hid_pre1.bprop(delta4)

batch_size = 200
num_epochs = 50
learning_rate = 0.001
num_samples = x_train.shape[0]
num_batches = num_samples // batch_size


acc = []
for epoch in range(num_epochs):
    confusion = ConfusionMatrix(num_classes)
    for i in range(num_batches):
        idx = range(i*batch_size, (i+1)*batch_size)
        x_batch = x_train[idx]
        target_batch = targets_train[idx]

        y_probs = forward(x_batch)
        loss = LossLayer.fprop(y_probs, target_batch)
        backward(y_probs, target_batch)

        l_hid_pre1.update_params(learning_rate)
        l_hid_pre2.update_params(learning_rate)

        confusion.batch_add(target_batch.argmax(-1), y_probs.argmax(-1))

    acc += [confusion.accuracy()]

print confusion







