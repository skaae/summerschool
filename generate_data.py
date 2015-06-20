"""
This file is a slightly modified data generator originally written by
Colin Raffel.

See the original file at

https://github.com/craffel/nntools/blob/recurrent/examples/recurrent.py
"""
import numpy as np


def gen_data(length, num_samples):
    '''
    Generate a batch of sequences for the "add" task, e.g. the target for the
    following

    ``| 0.5 | 0.7 | 0.3 | 0.1 | 0.2 | ... | 0.5 | 0.9 | ... | 0.8 | 0.2 |
      |  0  |  0  |  1  |  0  |  0  |     |  0  |  1  |     |  0  |  0  |``

    would be 0.3 + .9 = 1.2.  This task was proposed in [1]_ and explored in
    e.g. [2]_.

    Parameters
    ----------
    length : int
         sequence length.
    num_samples : int
        Number of samples to generate

    References
    ----------
    .. [1] Hochreiter, Sepp, and Jurgen Schmidhuber. "Long short-term memory."
    Neural computation 9.8 (1997): 1735-1780.

    .. [2] Sutskever, Ilya, et al. "On the importance of initialization and
    momentum in deep learning." Proceedings of the 30th international
    conference on machine learning (ICML-13). 2013.
    '''
    # Generate X - we'll fill the last dimension later
    X = np.concatenate([np.random.uniform(size=(num_samples, length, 1)),
                        np.zeros((num_samples, length, 1))],
                       axis=-1)
    mask = np.zeros((num_samples, length))
    y = np.zeros((num_samples,))
    # Compute masks and correct values
    for n in range(num_samples):
        # Set the second dimension to 1 at the indices to add
        X[n, np.random.randint(length/10), 1] = 1
        X[n, np.random.randint(length/2, length), 1] = 1
        # Multiply and sum the dimensions of X to get the target value
        y[n] = np.sum(X[n, :, 0]*X[n, :, 1])
    # Center the inputs and outputs
    X -= X.reshape(-1, 2).mean(axis=0)
    y -= y.mean()
    return X, y