import numpy as np
from confusionmatrix import ConfusionMatrix
from utils import onehot
from rnn_layers import RNNLayer, LinearLayerRNN
from layers import SigmoidActivationLayer, SoftmaxActivationLayer, \
    CrossEntropyLoss, AddLayer




VERBOSE = False
num_samples = 1000
seq_len = 20
num_inputs = 5
num_classes = 3
X = np.random.random((num_samples, seq_len, num_inputs)).astype('float32')
targets_train = X.sum(axis=-1)
targets_train = targets_train.flatten()
for i, y_i in enumerate(targets_train):
    if y_i < 2:
        targets_train[i] = 0
    elif y_i < 4:
        targets_train[i] = 1
    else:
        targets_train[i] = 2

targets_train = onehot(targets_train, num_classes)

targets_train = targets_train.reshape((num_samples, seq_len, num_classes))



# shift
targets_train = np.pad(targets_train, mode='constant',
                       pad_width=((0, 0), (2, 0), (0, 0)))
targets_train = targets_train[:, :-2]
targets_train = targets_train.astype('int')

num_hidden_units = 100


batch_size = 11
num_epochs = 50
num_units = 20
learning_rate = 0.0001
num_samples = X.shape[0]
num_batches = num_samples // batch_size

# original layers
l_in_org = LinearLayerRNN(num_inputs, num_units)
l_hid_org = LinearLayerRNN(num_units, num_units)
l_out_org = LinearLayerRNN(num_units, num_classes)
lossLayer = CrossEntropyLoss()


def create_timestep(num):
    num = str(num)
    step = {}
    step['l_in'] = LinearLayerRNN(
        num_inputs, num_units, W=l_in_org.W, b=l_in_org.b, name="IN"+num)
    step['l_hid'] = LinearLayerRNN(
        num_units, num_units, W=l_hid_org.W, b=l_hid_org.b, name="HID"+num)
    step['rnn_add'] = AddLayer()
    step['rnn_act'] = SigmoidActivationLayer()
    step['l_out'] = LinearLayerRNN(num_units, num_classes,
                                   W=l_out_org.W, b=l_out_org.b, name="OUT"+num)
    step['output_act'] = SoftmaxActivationLayer()
    step['loss'] = CrossEntropyLoss()
    return step


# create time steps
steps = [create_timestep(t) for t in range(seq_len)]

# fprop single step
def forward_step(x_t, hid_prev, step):
    if VERBOSE:
        print "fprop", step['l_in'], "and",
    x_to_hid = step['l_in'].fprop(x_t)

    if VERBOSE:
        print step['l_hid'], "->",
    hid_to_hid = step['l_hid'].fprop(hid_prev)

    if VERBOSE:
        print step['rnn_add'], "->",
    hid_pre = step['rnn_add'].fprop(x_to_hid, hid_to_hid)

    if VERBOSE:
        print ['rnn_act'], "->",
    hid = step['rnn_act'].fprop(hid_pre)

    if VERBOSE:
        print step['l_out'], "->",
    hid_to_out = step['l_out'].fprop(hid)

    if VERBOSE:
        print step['output_act']
    preds = step['output_act'].fprop(hid_to_out)

    return hid, preds

# forward prop over sequence
def forward(x_batch, steps):
    batch_size, num_steps, _ = x_batch.shape

    hid_prev = np.zeros((batch_size, num_units))
    for t in range(num_steps):
        x_t = x_batch[:, t]   # (batch_size, num_inputs)
        step = steps[t]
        hid_prev, preds = forward_step(x_t, hid_prev, step)

# bprop single step
def backward_step(targets_t, delta_hid_before, step):
    # set target to none if target is masked
    if targets_t is not None:
        probs = step['output_act'].a  # get output from softmax layer
        if VERBOSE:
            print step['loss'], "->",
        delta_loss = step['loss'].bprop(probs, targets_t)
        loss = step['loss'].fprop(probs, targets_t)

        if VERBOSE:
            print step['output_act'], "->",
        delta_out_act = step['output_act'].bprop(delta_loss)
        if VERBOSE:
            print step['l_out'], "->",
        delta_out = step['l_out'].bprop(delta_out_act)
    else:
        delta_out = np.zeros((batch_size, num_classes))
        step['l_out'].reset_grads()

    if VERBOSE:
        print step['rnn_act'], "->",
    delta_rnn_act = step['rnn_act'].bprop(delta_hid_before + delta_out)

    if VERBOSE:
        print step['rnn_add'], "->",
    delta_rnn_in, delta_rnn_hid = step['rnn_add'].bprop(delta_rnn_act)

    if VERBOSE:
        print step['l_in'], " and",
    delta_in = step['l_in'].bprop(delta_rnn_in)

    if VERBOSE:
        print step['l_hid']
    delta_hid = step['l_hid'].bprop(delta_rnn_hid)
    return delta_hid

# bprop sequence
def backward(targets_batch, steps):
    batch_size, num_steps, _ = targets_batch.shape
    delta_hid_before = np.zeros((batch_size, num_units))
    for t in reversed(range(num_steps)):
        delta_hid_before = backward_step(
            targets_batch[:, t],
            delta_hid_before,
            steps[t])

        # make sure that i backproped everything
        assert hasattr(steps[t]['l_in'], 'grad_W')
        assert hasattr(steps[t]['l_in'], 'grad_b')
        assert hasattr(steps[t]['l_hid'], 'grad_W')
        assert hasattr(steps[t]['l_hid'], 'grad_b')
        assert hasattr(steps[t]['l_out'], 'grad_W')
        assert hasattr(steps[t]['l_out'], 'grad_b')


def accumulate_grads_and_update(steps, lr):
    all_params = []
    all_params += steps[0]['l_out'].get_params()
    all_params += steps[0]['l_hid'].get_params()
    all_params += steps[0]['l_in'].get_params()
    all_acc = [np.zeros_like(param) for param in all_params]

    # accumulate over all time steps
    for step in steps:
        all_grads = []
        all_grads += step['l_out'].get_grads()
        all_grads += step['l_hid'].get_grads()
        all_grads += step['l_in'].get_grads()

        step['l_out'].reset_grads()
        step['l_hid'].reset_grads()
        step['l_in'].reset_grads()
        for acc, grad in zip(all_acc, all_grads):
            acc += grad

    # update parameters
    [W_out_grad, b_out_grad,
     W_hid_grad, b_hid_grad,
     W_in_grad, b_in_grad] = all_grads

    steps[0]['l_out'].update_params(grad_W=W_out_grad, grad_b=b_out_grad, lr=lr)
    steps[0]['l_hid'].update_params(grad_W=W_hid_grad, grad_b=b_hid_grad, lr=lr)
    steps[0]['l_in'].update_params(grad_W=W_in_grad, grad_b=b_in_grad, lr=lr)


def mean_loss(steps):
    return np.mean([step['loss'].a for step in steps])


for i in range(100):
    for j in range(num_samples//batch_size):
        idx = range(j*batch_size, (1+j)*batch_size)
        forward(X[idx], steps)
        backward(targets_train[idx], steps)
        accumulate_grads_and_update(steps, learning_rate)
    print mean_loss(steps)