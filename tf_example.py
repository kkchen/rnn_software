"""TensorFlow RNN example."""

import warnings

import os
import string

with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    warnings.simplefilter('ignore', FutureWarning)

    import tensorflow as tf

from dataset import Dataset
from iterator import Iterator
import text

def tf_example(args):
    """Build, train, and test the discriminator using TensorFlow frontend."""
    dtype = tf.float32

    # Get one-hot encoded English and German/French words.
    words, labels = text.get_data(args.length, args.language, True)
    data = Dataset(words, labels, args.validation_split)
    iterator = Iterator(data, args.n_epochs, args.batch_size)

    # Model input and output.
    with tf.name_scope('io'):
        x = tf.placeholder(dtype, [words.shape[0], None, words.shape[2]], 'x')
        y = tf.placeholder(dtype, [None, 1], 'y')

    # The neural network.
    with tf.variable_scope('lstm0'):
        lstm0 = tf.contrib.rnn.LSTMBlockFusedCell(args.n_state)
        lstm0_output, lstm0_state = lstm0(x, dtype=dtype)

    with tf.variable_scope('lstm1'):
        lstm1 = tf.contrib.rnn.LSTMBlockFusedCell(args.n_state)
        lstm1_output, lstm1_state = lstm1(lstm0_output, dtype=dtype)

    with tf.variable_scope('dense'):
        # Apply the sigmoid in the loss, not in the dense layer.
        logits = tf.layers.dense(lstm1_output[-1, :, :], 1, name='logits')

    # The training loss.
    with tf.name_scope('loss'):
        loss = tf.losses.sigmoid_cross_entropy(y, logits)

    # Classification accuracy.  y = 1 iff logits > 0.
    with tf.name_scope('accuracy'):
        correct = tf.cast(
            tf.equal(tf.cast(y, tf.bool), tf.greater(logits, 0)),
            dtype
        )
        accuracy = tf.reduce_mean(correct)

    # The optimizer.
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer()
        global_step = tf.Variable(0, False, name='global_step')

    # The training operation.
    with tf.name_scope('train'):
        train_op = optimizer.minimize(loss, global_step)

    # Label inputs.
    with tf.name_scope('predict'):
        label = tf.sigmoid(logits, 'label')

    # One-hot encoded words to label.
    test_words = text.get_test_data(args.language)
    words_encoded = text.one_hot(test_words, args.length, True)

    # Run the training and testing.
    with tf.Session() as session:
        tf.global_variables_initializer().run()

        # Run through the minibatches.
        for data_x, data_y in iterator:
            # Run the training operation and get the loss.
            train_op.run({x: data_x, y: data_y})

            # Report diagnostics at every epoch.
            if iterator.new_epoch:
                # Report training loss and accuracy.
                train_loss, train_accuracy = session.run(
                    [loss, accuracy],
                    {x: data_x, y: data_y}
                )
                # Report validation loss and accuracy.
                val_loss, val_accuracy = session.run(
                    [loss, accuracy],
                    {x: data.x['val'], y: data.y['val']}
                )

                print('Epoch {}:'.format(iterator.epoch))
                print(
                    '    Train: loss = {:.6f}, accuracy = {:.6f}'
                    .format(train_loss, train_accuracy)
                )
                print(
                    '    Validation: loss = {:.6f}, accuracy = {:.6f}'
                    .format(val_loss, val_accuracy)
                )

        # Label words.
        test_labels = label.eval({x: words_encoded})

        # Save all variables.
        saver = tf.train.Saver()
        saver.save(session, os.getcwd() + '/tf_example.ckpt')

    # Print predictions.
    print('\nWord: P({})'.format(args.language.capitalize()))

    for word, label in zip(test_words, test_labels):
        print('{}: {:.3f}'.format(word, float(label)))
