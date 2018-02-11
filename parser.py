"""Command-line parser for run_example.py."""

import argparse

def parse(doc):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-l',
        '--length',
        default=14,
        type=int,
        help='RNN length; ignore words longer than this'
    )
    parser.add_argument(
        '-n',
        '--n-state',
        default=128,
        type=int,
        help='RNN state size in each layer'
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        default=1028,
        type=int,
        help='minibatch size'
    )
    parser.add_argument(
        '-e',
        '--n_epochs',
        default=15,
        type=int,
        help='number of complete passes through all the training data'
    )
    parser.add_argument(
        '-v',
        '--validation-split',
        default=0.2,
        type=float,
        help='proportion of data to use for validation'
    )
    parser.add_argument(
        '-g',
        '--gpu',
        action='store_true',
        help='optimize Keras for GPU; does nothing for TensorFlow'
    )

    parser.add_argument(
        'frontend',
        choices=('keras', 'tensorflow'),
        help='which neural network frontend to use'
    )
    parser.add_argument(
        'language',
        choices=('german', 'french'),
        help='which language to discriminate against English'
    )

    return parser.parse_args()
