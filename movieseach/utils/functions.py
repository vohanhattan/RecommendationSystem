from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import pandas as pd

# utilities
ps = PorterStemmer()
table = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))
# words reverse index (word: idxs of the movies containing that word)


def preprocess_text(line):
    """
    Split tokens on white space.
    Remove all punctuation from words.
    Convert to lowercase
    Remove all words that are known stop words.
    Remove all words that have a length <= 1 character.
    Stems words
    """
    # split into tokens by white space
    tokens = line.split()
    # remove punctuation and lowercase
    tokens = [w.translate(table).lower() for w in tokens]
    # filter out short tokens, stop words and stem
    tokens = [ps.stem(w) for w in tokens if w not in stop_words and len(w) > 1]
    return tokens  # return a list of words


def drop_duplicates(l):
    """Removes duplicates from a list"""
    return list(dict.fromkeys(l))


def exists(value):
    if pd.isna(value) or pd.isnull(value) or value is None:
        return False
    return True


def preprocess_list(array):
    """Apply preprocess_text to a list of sentences"""
    tokens = []
    if array is None:
        return tokens
    for s in array:
        if s is not None:
            tokens += preprocess_text(s)
    return tokens


if __name__ == '__main__':
    pass
