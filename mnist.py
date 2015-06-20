import numpy as np
from confusionmatrix import ConfusionMatrix
from layers import *
from utils import onehot


# Load Mnist data and convert vector represention to one-hot
data = np.load('mnist.npz')
num_classes = 10
x_train = data['X_train']
targets_train = data['y_train']
targets_train = onehot(targets_train, num_classes)
num_samples, num_inputs = x_train.shape
num_hidden_units = 100

batch_size = 200
num_epochs = 50
learning_rate = 0.001
num_samples = x_train.shape[0]
num_batches = num_samples // batch_size




ffn = FeedforwardNetwork()
ffn.add(LinearLayer(num_inputs, num_hidden_units))
ffn.add(ReluActivationLayer())
ffn.add(LinearLayer(num_hidden_units, num_classes))
ffn.add(SoftmaxActivationLayer())
losslayer = CrossEntropyLoss()

print "Network"
print ffn
print "Loss", losslayer
print ""


acc = []
for epoch in range(num_epochs):
    confusion = ConfusionMatrix(num_classes)
    for i in range(num_batches):
        idx = range(i*batch_size, (i+1)*batch_size)
        x_batch = x_train[idx]
        target_batch = targets_train[idx]

        y_probs =  ffn.forward(x_batch)
        loss = losslayer.fprop(y_probs, target_batch)
        delta = losslayer.bprop(y_probs, target_batch)
        ffn.backward(delta)
        ffn.update(learning_rate)
        confusion.batch_add(target_batch.argmax(-1), y_probs.argmax(-1))
    curr_acc = confusion.accuracy()
    print "Epoch %i : Loss %f Train acc %f" % (epoch, loss, curr_acc)
    acc += [curr_acc]






