"""Dataset iterator for pre-TF r1.4."""

import numpy as np

class Iterator:

    def __init__(self, dataset, total_epochs, batch_size):
        self._dataset = dataset
        self._total_epochs = total_epochs
        self._batch_size = batch_size
        self._i = 0 # Data index.
        self.epoch = 0 # Number of epochs completed.

    def __iter__(self):
        return self

    def __next__(self):
        """Get the next minibatch of training data."""
        if self.epoch >= self._total_epochs:
            raise StopIteration

        start = self._i # Starting index for this batch.
        self._i += self._batch_size # Ending index.

        # self._i can safely exceed the 0th dimension if at the end of the
        # epoch.
        result_x = self._dataset.x['train'][:, start:self._i, :]
        result_y = self._dataset.y['train'][start:self._i, :]
        self.new_epoch = False # True if we started a new epoch.
        length = self._dataset.y['train'].shape[0]

        # If we reached the end of the data, shuffle and sample more as needed.
        while self._i > length:
            self.epoch += 1
            self.new_epoch = True
            self._i -= length
            self._dataset.shuffle()

            result_x = np.concatenate(
                (result_x, self._dataset.x['train'][:, :self._i, :]),
                1
            )
            result_y = np.concatenate(
                (result_y, self._dataset.y['train'][:self._i, :])
            )

        return result_x, result_y
