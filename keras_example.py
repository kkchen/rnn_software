"""Keras RNN example."""

import string

import numpy as np
import keras

from discriminator import Discriminator
import text

class KerasDiscriminator(Discriminator):

    """Class that holds several Keras models for discriminating."""

    def __init__(self, args):
        super().__init__(args)
        # Options to pass to RNNs.
        self._kw = {'recurrent_activation': 'sigmoid', 'unroll': True}

        self.model = keras.models.Sequential() # The Keras model.
        getattr(self, '_' + args.model)() # Build the Keras model’s layers.

    def compile(self):
        """Compile the Keras model."""
        self.model.summary() # Print the layers.
        self.model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    def train(self, x, y):
        """Train the RNN on input data x and binary labels y.

        x should have dimensions (# words) x max_length x (# chars + 1), and y
        should have dimension (# words).
        """
        self.model.fit(
            x,
            y,
            self._batch_size,
            self._epochs,
            validation_split=self._validation_split
        )

    def label(self, words):
        """Label the given words."""
        words_encoded = text.one_hot(words, self._max_length)
        labels = self.model.predict(words_encoded)

        # Print predictions.
        print('\nWord: P(German)')

        for word, label in zip(words, labels):
            print('{}: {}'.format(word, float(label)))

    def _baseline(self):
        """Multi-layer baseline model."""
        # LSTM layer 1.  Retain outputs across all time steps.  Only the first
        # layer needs to specify input_shape.
        self.model.add(
            keras.layers.LSTM(
                self._n_state,
                return_sequences=True,
                input_shape=(self._max_length, len(string.ascii_lowercase) + 1),
                **self._kw
            )
        )

        # Intermediate RNN layers.  Return outputs from all time steps.
        for i in range(self._layers - 2):
            self.model.add(
                keras.layers.LSTM(
                    self._n_state,
                    return_sequences=True,
                    **self._kw
                )
            )

        # Final RNN layer.  Only output the final time step’s output.
        self.model.add(keras.layers.LSTM(self._n_state, **self._kw))
        # Dense layer on the final output of the LSTM.
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))

    def _bidirectional(self):
        """Bidirectional model."""
        # First RNN.
        self.model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    self._n_state,
                    return_sequences=True,
                    **self._kw
                ),
                input_shape=(self._max_length, len(string.ascii_lowercase) + 1)
            )
        )

        # Intermediate RNNs.  Return outptus from all time steps.
        for i in range(self._layers - 2):
            self.model.add(
                keras.layers.Bidirectional(
                    keras.layers.LSTM(
                        self._n_state,
                        return_sequences=True,
                        **self._kw
                    )
                )
            )

        # Final RNN.
        self.model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(self._n_state, **self._kw)
            )
        )
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))

    def _all_outputs(self):
        """All outputs from the 2nd layer go to the dense layer."""
        # RNN layer 1.  Retain outputs across all time steps.
        self.model.add(
            keras.layers.LSTM(
                self._n_state,
                return_sequences=True,
                input_shape=(self._max_length, len(string.ascii_lowercase) + 1),
                **self._kw
            )
        )

        # All other RNNs.  Retain outputs across all time steps.
        for i in range(self._layers - 1):
            self.model.add(
                keras.layers.LSTM(
                    self._n_state,
                    return_sequences=True,
                    **self._kw
                )
            )

        # Flatten all the outputs across all time steps.
        self.model.add(keras.layers.Flatten())
        # Output a single scalar from all time steps.
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))

    def _noise_dropout(self):
        """Model with input Gaussian noise, and dropout in all layers."""
        dropout = 0.3

        # Add Gaussian noise at the input.
        self.model.add(
            keras.layers.GaussianNoise(
                0.1,
                input_shape=(self._max_length, len(string.ascii_lowercase) + 1)
            )
        )
        # First RNN layer.
        self.model.add(
            keras.layers.LSTM(
                self._n_state,
                return_sequences=True,
                dropout=dropout, # Input-to-state dropout.
                recurrent_dropout=dropout, # State-to-state dropout.
                input_shape=(self._max_length, len(string.ascii_lowercase) + 1),
                **self._kw
            )
        )

        # Intermediate RNN layers.  Keep outputs from all time steps.
        for i in range(self._layers - 2):
            self.model.add(
                keras.layers.LSTM(
                    self._n_state,
                    return_sequences=True,
                    dropout=dropout,
                    recurrent_dropout=dropout,
                    **self._kw
                )
            )

        # Final RNN layer.  Only keep the output from the final time step.
        self.model.add(
            keras.layers.LSTM(
                self._n_state,
                dropout=dropout,
                recurrent_dropout=dropout,
                **self._kw
            )
        )

        # Apply dropout and then the final dense layer.
        self.model.add(keras.layers.Dropout(dropout))
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))

    def _deep_output(self):
        """Model with a deep LSTM output."""
        # First RNN layer.
        self.model.add(
            keras.layers.LSTM(
                self._n_state,
                return_sequences=True,
                input_shape=(self._max_length, len(string.ascii_lowercase) + 1),
                **self._kw
            )
        )

        # All other RNN layers.
        for i in range(self._layers - 1):
            self.model.add(
                keras.layers.LSTM(
                    self._n_state,
                    return_sequences=True,
                    **self._kw
                )
            )

        # Every time step gets the same dense layer.
        self.model.add(
            keras.layers.TimeDistributed(
                keras.layers.Dense(self._n_state // 2, activation='relu')
            )
        )
        # Collect all outputs across time.
        self.model.add(keras.layers.Flatten())
        # Output a single scalar across time.
        self.model.add(keras.layers.Dense(1, activation='sigmoid'))
