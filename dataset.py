"""Dataset interface for pre-TF r1.4."""

import numpy as np

class Dataset:

    def __init__(self, x, y, validation_split=1.):
        """Initialize with the data and hyperparameters."""
        length = y.shape[0]

        # Split the data.
        split_ind = int(length * validation_split)
        self.x = {'train': x[:, split_ind:, :], 'val': x[:, :split_ind, :]}
        self.y = {'train': y[split_ind:, :], 'val': y[:split_ind, :]}

    def shuffle(self):
        """Shuffle the training data."""
        self.x['train'], self.y['train'] = self._shuffle(
            self.x['train'],
            self.y['train']
        )

    @staticmethod
    def _shuffle(x, y):
        """Shuffle the given data and return them."""
        ind = np.arange(y.shape[0])
        np.random.shuffle(ind)

        return x[:, ind, :], y[ind]
