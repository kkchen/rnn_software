"""TensorFlow RNN example."""

import string

import numpy as np
import tensorflow as tf

from discriminator import Discriminator
import text

class TFDiscriminator(Discriminator):

    """Class that holds several TensorFlow models for discriminating."""

    def __init__(self, args, dataset):
        super().__init__(args)
        self._dataset = dataset
        self._dataset.batch(self._batch_size)
        self._dataset.repeat(self._epochs)
        self._iterator = self._dataset.make_one_shot_iterator()

    def compile(self):
        pass

    def train(self, x, y):
        pass

    def label(self, words):
        pass

    def _baseline(self):
        """2-layer baseline model."""
        x, y = self._iterator.get_next()
        # Unlike Keras, TensorFlow needs RNN input to be arranged as (length,
        # batch, inputs).
        x = tf.transpose(x, [1, 0, 2], 'transposed_input')

        cell = tf.contrib.LSTMBlockFusedCell(self._n_state)
        lstms = tf.contrib.rnn.MultiRNNCell([cell for i in range(self._layers)])
        state = cell.zero_state(self._batch_size, tf.float32)
        lstm_output, new_state = lstm(x, state)
        dense_output = tf.layers.dense(
            tf.squeeze(lstm_output[-1, :]),
            1,
            tf.sigmoid,
            name='dense_output'
        )

    def _bidirectional(self):
        """Bidirectional model."""
        pass

    def _all_outputs(self):
        """All outputs from the 2nd layer go to the dense layer."""
        pass

    def _noise_dropout(self):
        """Model with input Gaussian noise, and dropout in all layers."""
        dropout = 0.3
        pass

    def _deep_output(self):
        """Model with a deep LSTM output."""
        pass


def get_dataset(x, y):
    """Get a TensorFlow dataset for the input data x and labels y."""
    return tf.data.Dataset.from_tensor_slices(
        {'x': tf.constant(x, name='encoded_words'),
         'y': tf.constant(y, name='labels')}
    )
