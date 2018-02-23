"""Simple Keras RNN example.

Builds, trains, and tests an RNN discriminator that distinguishes between
English and German/French.
"""

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    warnings.simplefilter('ignore', FutureWarning)

    from tensorflow import keras

import text

def keras_example(args):
    """Build, train, and test the discriminator using the Keras frontend."""
    # Get one-hot encoded English and German/French words.
    x, y = text.get_data(args.length, args.language)
    implementation = 2 if args.gpu else 0, # Optimize matrix sizes.

    # The Keras neural network model.
    discriminator = keras.models.Sequential()
    # The first LSTM layer.
    discriminator.add(
        keras.layers.LSTM(
            args.n_state,
            return_sequences=True, # Keep outputs from all time steps.
            input_shape=(x.shape[1], x.shape[2]),
            implementation=implementation,
            recurrent_activation='sigmoid',
            unroll=True # Faster performance but more memory consumption.
        )
    )
    #The second LSTM layer.  Only keep the output from the final time step.
    discriminator.add(
        keras.layers.LSTM(
            args.n_state,
            return_sequences=False,
            implementation=implementation,
            recurrent_activation='sigmoid',
            unroll=True
        )
    )
    # Dense layer.
    discriminator.add(keras.layers.Dense(1, activation='sigmoid'))

    # Compile the Keras model.
    discriminator.summary()
    discriminator.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    # Train the Keras model.
    discriminator.fit(
        x,
        y,
        args.batch_size,
        args.n_epochs,
        validation_split=args.validation_split
    )

    # Save the model.
    discriminator.save('keras_example.h5')

    # Label test words.
    test_words = text.get_test_data(args.language)
    words_encoded = text.one_hot(test_words, args.length)
    test_labels = discriminator.predict(words_encoded)

    # Print predictions.
    print('\nWord: P({})'.format(args.language.capitalize()))

    for word, label in zip(test_words, test_labels):
        print('{}: {:.3f}'.format(word, float(label)))
