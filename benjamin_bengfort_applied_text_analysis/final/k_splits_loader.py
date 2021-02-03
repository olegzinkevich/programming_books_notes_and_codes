import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as tts

# We consider it
# so important to applied text analytics that we start by creating a CorpusLoader object
# that wraps a CorpusReader in order to provide streaming access to k splits!

# We’ll construct the base class, CorpusLoader, which is instantiated with a Corpus
# Reader, the number of folds, and whether or not to shuffle the corpus, which is true
# by default

class CorpusLoader(object):

    def __init__(self, reader, folds=12, shuffle=True, categories=None):
        self.reader = reader
        self.folds  = KFold(n_splits=folds, shuffle=shuffle)
        self.files  = np.asarray(self.reader.fileids(categories=categories))

    def fileids(self, idx=None):
        if idx is None:
            return self.files
        return self.files[idx]

    # The documents() method returns a generator
# to provide memory-efficient access to the documents in our corpus, and yields a list
# of tagged tokens for each fileid in the split, one document at a time.
    def documents(self, idx=None):
        for fileid in self.fileids(idx):
            yield list(self.reader.docs(fileids=[fileid]))

    # The labels()
# method uses the corpus.categories() to look up the label from the corpus and
# returns a list of labels, one per document.
    def labels(self, idx=None):
        return [
            self.reader.categories(fileids=[fileid])[0]
            for fileid in self.fileids(idx)
        ]

    # Finally, we add a custom iterator method that calls KFold’s split() method, yielding # training and test splits for each fold:
    def __iter__(self):
        for train_index, test_index in self.folds.split(self.files):
            X_train = self.documents(train_index)
            y_train = self.labels(train_index)

            X_test = self.documents(test_index)
            y_test = self.labels(test_index)

            yield X_train, X_test, y_train, y_test


if __name__ == '__main__':
    from reader import PickledCorpusReader

    corpus = PickledCorpusReader('../corpus')
    loader = CorpusLoader(corpus, 12)
