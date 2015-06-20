"""
Simple container for feedforward networks
"""


class FeedForwardNetwork():
    def __init__(self):
        self.layers = []

    def __str__(self):
        if len(self.layers) > 0:
            s = "input "
            for l in self.layers:
                s += " --> " + str(l)
            s += " --> output"
        else:
            s = "No Layers"
        return s

    def add(self, layer):
        self.layers += [layer]

    def forward(self, x, train=True):
        for layer in self.layers:
            x = layer.fprop(x, train=train)
        return x

    def backward(self, delta):
        for layer in reversed(self.layers):
            delta = layer.bprop(delta)

    def update(self, lr):
        for layer in self.layers:
            layer.update_params(lr)