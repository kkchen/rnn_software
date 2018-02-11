#!/usr/bin/env python3

"""Simple Keras RNN example.

Builds, trains, and tests an RNN discriminator that distinguishes between
English and German/French.  For the demonstration, two stacked LSTMs feed into a
dense node with a standard logistic activation.
"""

from keras_example import keras_example
from parser import parse
from tf_example import tf_example

def main():
    args = parse(__doc__)

    if args.frontend == 'keras':
        keras_example(args)
    elif args.frontend == 'tensorflow':
        tf_example(args)


if __name__ == '__main__':
    main()
