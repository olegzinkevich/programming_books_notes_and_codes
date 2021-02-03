import bs4
import time
import nltk
import pickle
import logging
import os
import codecs
import sqlite3

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

# pip install readability-lxml
from readability.readability import Unparseable
from readability.readability import Document as Paper

from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

DOC_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.json'
PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
CAT_PATTERN = r'([a-z_\s]+)/.*'


# a corpus of roughly 300,000 HTML news
# articles, these preprocessing steps took over 12 hours. This is not something we will
# want to have to do every time we run our models or test out a new set of hyperparameters.
# In practice, we address this by adding two additional classes, a Preprocessor class
# that wraps our HTMLCorpusReader to wrangle the raw corpus to store an intermediate
# transformed corpus artifact, and a PickledCorpusReader that can stream the transformed
# documents from disk in a standardized fashion for downstream vectorization
# and analysis

# Once we have a compressed, preprocessed, pickled corpus, we can quickly access our
# corpus data without having to reapply tokenization methods or any string parsing—
# instead directly loading Python data structures and thus saving a significant amount
# of time and effort

# To read our corpus, we require a PickledCorpusReader class that uses
# pickle.load() to quickly retrieve the Python structures from one document at a
# time. This reader contains all the functionality of the HTMLCorpusReader (since it
# extends it), but since it isn’t working with raw text under the hood, it will be many
# times faster.

# When dealing with large corpora, the PickledCorpusReader makes things immensely
# easier. Although preprocessing and accessing data can be parallelized using the multi
# processing Python library (which we’ll see in Chapter 11), once the corpus is used to
# build models, a single sequential scan of all the documents before vectorization is
# required. Though this process can also be parallelized, it is not common to do so
# because of the experimental nature of exploratory modeling. Utilizing the pickle serialization
# speeds up the modeling and exploration process significantly!

class PickledCorpusReader(CategorizedCorpusReader, CorpusReader):

    def __init__(self, root, fileids=PKL_PATTERN, **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining arguments
        are passed to the ``CorpusReader`` constructor.
        """
        # Add the default category pattern if not passed into the class.
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids)

    def resolve(self, fileids, categories):
        """
        Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function. This primarily bubbles up to
        the high level ``docs`` method, but is implemented here similar to
        the nltk ``CategorizedPlaintextCorpusReader``.
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Returns the document loaded from a pickled object for every file in
        the corpus. Similar to the BaleenCorpusReader, this uses a generator
        to acheive memory safe iteration.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, enc, fileid in self.abspaths(fileids, True, True):
            with open(path, 'rb') as f:
                yield pickle.load(f)

    def paras(self, fileids=None, categories=None):
        """
        Returns a generator of paragraphs where each paragraph is a list of
        sentences, which is in turn a list of (token, tag) tuples.
        """
        for doc in self.docs(fileids, categories):
            for paragraph in doc:
                yield paragraph

    def sents(self, fileids=None, categories=None):
        """
        Returns a generator of sentences where each sentence is a list of
        (token, tag) tuples.
        """
        for paragraph in self.paras(fileids, categories):
            for sentence in paragraph:
                yield sentence

    # Sentences are lists of (token, tag) tuples, so we need two methods to access the
# ordered set of words that make up a document or documents. The first tagged()
# method returns the token and tag together, the second words() method returns only
# the token in question.
    def tagged(self, fileids=None, categories=None):
        for sent in self.sents(fileids, categories):
            for token in sent:
                yield token

    def words(self, fileids=None, categories=None):
        """
        Returns a generator of (token, tag) tuples.
        """
        for token in self.tagged(fileids, categories):
            yield token[0]


if __name__ == '__main__':

    corpus = PickledCorpusReader('C:/Users/810004/Desktop/Html_corpus/pickled_corpus')


    for category in corpus.categories():
        n_docs = len(corpus.fileids(categories=[category]))
        n_words = sum(1 for word in corpus.words(categories=[category]))

        print("- '{}' contains {:,} docs and {:,} words".format(category, n_docs, n_words))