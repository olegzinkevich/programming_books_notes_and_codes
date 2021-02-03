# We’ll begin with constants to define the start and end of the sentence as <s> and </s>
# (because English reads left to right, the left_pad_symbol and right_pad_symbol,
# respectively). In languages that read right to left, these could be reversed.
# The second part of the code creates a function nltk_ngrams that uses the partial
# function to wrap the nltk.ngrams function with our code-specific keyword arguments.
# This ensures that every time we call nltk_ngrams, we get our expected behavior,
# without managing the call signature everywhere in our code that we use it. Finally
# our newly redefined ngrams function takes as arguments a string containing our text
# and n-gram size. It then applies the sent_tokenize and word_tokenize functions to
# the text before passing them into nltk_ngrams to get our padded n-grams


import sys
import argparse

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import ngrams as nltk_ngrams
from functools import partial

LPAD_SYMBOL = "<s>"
RPAD_SYMBOL = "</s>"

nltk_ngrams = partial(nltk_ngrams,
    pad_right=True, pad_left=True,
    right_pad_symbol=RPAD_SYMBOL, left_pad_symbol=LPAD_SYMBOL
)


def ngrams(self, n=2, fileids=None, categories=None):
    for sent in self.sents(fileids=fileids, categories=categories):
        for ngram in nltk.ngrams(sent, n):
            yield ngram



