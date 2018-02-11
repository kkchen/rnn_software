"""Text-processing utilities."""

import string
import numpy as np

def get_data(max_length, language, tensorflow=False):
    """Return one-hot English and foreign words, along with language labels.

    Each character is one-hot encoded, ande words shorter than max_length are
    padded with end-of-word tokens.  Words longer than max_length are excluded.

    The data are returned as x, y.  For Keras, x is a (# words) x max_length x
    (# chars) array (where (# chars) = 27; the extra character is the
    end-of-string token).  For TensorFlow, x is a max_length x (# words) x (#
    chars) array.

    For Keras, y is a (# words)-long label vector, where a label of 0 indicates
    English and 1 indicates German/French.  For TensorFlow, y is (# words) x 1.
    """
    # Read lists of English and German words.
    with open('english.txt') as f:
        english = f.read().split()

    if language == 'german':
        foreign = _process_german()
    elif language == 'french':
        foreign = _process_french()
    else:
        raise ValueError('Invalid language "{}".'.format(language))

    # One-hot encode lists of words, up to a maximum length.
    english_encoded = one_hot(english, max_length)
    foreign_encoded = one_hot(foreign, max_length)

    # Combine the one-hot-encoded words.
    x = np.concatenate((english_encoded, foreign_encoded))

    # Shuffle the data.
    ind = np.arange(x.shape[0])
    np.random.shuffle(ind)
    x = x[ind, :, :]

    # Create labels, where a 0 value or index means English and 1 means
    # German/French, and shuffle.
    y = np.r_[np.zeros(len(english)), np.ones(len(foreign))]
    y = y[ind]

    if tensorflow:
        # TensorFlow needs the input data to be length x batch x chars, not
        # batch x length x chars.
        x = np.transpose(x, (1, 0, 2))
        # TensorFlow needs the output to be batch x 1.
        y = np.expand_dims(y, 1)

    return x, y


def get_test_data():
    """Return a pre-selected short list of test words."""
    words = ['informatic', 'armature', 'conditioner', 'baste', 'helical',
             'lamentation', 'surging', 'surjective']
    words += ['kostritzer', 'krombacher', 'erhard', 'janssen', 'hasseroder',
              'wernesgruner', 'franziskaner', 'oettingen', 'warstein']

    return words


def one_hot(words, max_length=None, transpose=False):
    """Take in a list of words with a-z chars, and return as one-hot array.

    The indices are [batch, word, letter].  Ignore words longer than max_length
    if max_length is not None.

    If transpose is True, then return as length x batch x chars rather than
    batch x length x chars.
    """
    if max_length is None:
        # Compute the maximum length.
        max_length = max(len(word) for word in words)
    else:
        # Truncate words that are too long.
        words = [word for word in words if len(word) <= max_length]

    # Check that only a-z letters are used.
    for word in words:
        for char in word:
            if char not in string.ascii_lowercase:
                raise RuntimeError(
                    'The character {} is not in a-z.'.format(char)
                )

    # Convert letter to index, e.g., a: 0, b: 1, etc.  The end-of-word token has
    # index 26 but is not in this dict.
    encoding = {char: num for num, char in enumerate(string.ascii_lowercase)}
    # Initialize the one-hot-encoded array of words.  Add one to the number of
    # letters to account for the end-of-word token.
    result = np.zeros((len(words), max_length, len(string.ascii_lowercase) + 1))

    for i, word in enumerate(words):
        for j, char in enumerate(word):
            result[i][j][encoding[char]] = 1 # Encode this character.

        for j in range(j + 1, max_length):
            result[i][j][-1] = 1 # Encode the end-of-word token.

    return np.transpose(result, (1, 0, 2)) if transpose else result


def _process_french():
    """Restrict French dictionary to just a-z letters, and deduplicate.

    The raw dictionary is read from french.txt.  Return the new list.
    """
    return _process('french.txt', ('ä', 'é', 'ü'), ('a', 'e', 'u'), ".-'")


def _process_german():
    """Restrict German dictionary to just a-z letters, and deduplicate.

    The raw dictionary is read from german.txt.  Return the new list.
    """
    return _process(
        'german.txt',
        ('Ä', 'Ö', 'Ü', 'ä', 'é', 'ö', 'ü', 'ß'),
        ('a', 'o', 'u', 'a', 'e', 'o', 'u', 'ss'),
        "'-./0123589"
    )


def _process(filename, chars, subs, delete):
    """Return the list of words with substitutions and deleted.

    The words in the file given by filename are loaded.  Characters in chars are
    substituted with those in subs, and characters in delete are deleted.  The
    deduplicated list of words is then returned.
    """
    with open(filename) as f:
        old = f.read().split()

    new = []

    # Make character substitutions.
    for word in old:
        for c, s in zip(chars, subs):
            word = word.replace(c, s)

        for d in delete:
            word = word.replace(d, '')

        if len(word) > 0:
            new.append(word.lower())

    # Deduplicate the list.
    return list(set(new))
