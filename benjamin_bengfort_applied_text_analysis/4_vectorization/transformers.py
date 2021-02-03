#!/usr/bin/env python3

import os
import nltk
import gensim
import unicodedata

from loader import CorpusLoader
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.matutils import sparse2full

# sklearn

# The BaseEstimator Interface
# The API itself is object-oriented and describes a hierarchy of interfaces for different
# machine learning tasks. The root of the hierarchy is an Estimator, broadly any object
# that can learn from data. The primary Estimator objects implement classifiers,
# regressors, or clustering algorithms. However, they can also include a wide array of
# data manipulation, from dimensionality reduction to feature extraction from raw
# data. The Estimator essentially serves as an interface, and classes that implement
# Estimator functionality must have two methods—fit and predict—as shown here:

from sklearn.base import BaseEstimator


class Estimator(BaseEstimator):

    # The Estimator.fit method sets the state of the estimator based on the training data,
    # X and y. The training data X is expected to be matrix-like—for example, a twodimensional
    # NumPy array of shape (n_samples, n_features) or a Pandas DataFrame
    # whose rows are the instances and whose columns are the features. Supervised estimators
    # are also fit with a one-dimensional NumPy array, y, that holds the correct labels.
    # The fitting process modifies the internal state of the estimator such that it is ready or
    # able to make predictions. This state is stored in instance variables that are usually
    # postfixed with an underscore (e.g., Estimator.coefs_). Because this method modifies
    # an internal state, it returns self so the method can be chained.
    def fit(self, X, y=None):
        """
        Accept input data, X, and optional target data, y. Returns self.
        """
        return self
    # The Estimator.predict method creates predictions using the internal, fitted state of
# the model on the new data, X. The input for the method must have the same number
# of columns as the training data passed to fit, and can have as many rows as predictions
# are required. This method returns a vector, yhat, which contains the predictions
# for each row in the input data.
    def predict(self, X):
        """
        Accept input data, X and return a vector of predictions for each row.
        """
        return yhat

# Estimator objects have parameters (also called hyperparameters) that define how the
# fitting process is conducted. These parameters are set when the Estimator is instantiated
# (and if not specified, they are set to reasonable defaults), and can be modified
# with the get_param and set_param methods that are also available from the BaseEsti
# mator super class.

# example creating of a model

# We engage the Scikit-Learn API by specifying the package and type of the estimator.
# Here we select the Naive Bayes model family, and a specific member of the family, a
# multinomial model (which is suitable for text classification). The model is defined
# when the class is instantiated and hyperparameters are passed in. Here we pass an
# alpha parameter that is used for additive smoothing, as well as prior probabilities for
# each of our two classes. The model is trained on specific data (documents and
# labels) and at that point becomes a fitted model. This basic usage is the same for
# every model (Estimator) in Scikit-Learn, from random forest decision tree ensembles
# to logistic regressions and beyond.

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB(alpha=0.0, class_prior=[0.4, 0.6])
model.fit(documents, labels)


# Transformer

# Scikit-Learn also specifies utilities for performing machine learning in a repeatable
# fashion. We could not discuss Scikit-Learn without also discussing the Transformer
# interface. A Transformer is a special type of Estimator that creates a new dataset
# from an old one based on rules that it has learned from the fitting process. The interface
# is as follows:

# As a result, we propose to use the sklearn API to create
# our own Transformer and Estimator objects that implement methods from
# NLTK and Gensim. For example, we can create topic modeling estimators that wrap
# Gensim’s LDA and LSA models (which are not currently included in Scikit-Learn) or
# create transformers that utilize NLTK’s part-of-speech tagging and named entity
# chunking methods.

from sklearn.base import TransformerMixin
class Transfomer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        """
        Learn how to transform data based on input data, X.
        """
        return self

    # The Transformer.transform method takes a dataset and returns a new dataset, X`,
# with new values based on the transformation process. There are several transformers
# included in Scikit-Learn, including transformers to normalize or scale features, handle
# missing values (imputation), perform dimensionality reduction, extract or select
# features, or perform mappings from one feature space to another.
    def transform(self, X):
        """
        Transform X into a new dataset, Xprime and return it.
        """
        return Xprime



# Creating a custom Gensim vectorization transformer based on sklearn Trasformer and Estimator

# Our GensimVectorizer transformer will wrap a Gensim Dictionary
# object generated during fit() and whose doc2bow method is used during
# transform(). The Dictionary object (like the TfidfModel) can be saved and loaded
# from disk, so our transformer utilizes that methodology by taking a path on instantiation.
# If a file exists at that path, it is loaded immediately. Additionally, a save()
# method allows us to write our Dictionary to disk, which we can do in fit().

# The fit() method constructs the Dictionary object by passing already tokenized
# and normalized documents to the Dictionary constructor. The Dictionary is then
# immediately saved to disk so that the transformer can be loaded without requiring a
# refit. The transform() method uses the Dictionary.doc2bow method, which returns
# a sparse representation of the document as a list of (token_id, frequency) tuples.
# This representation can present challenges with Scikit-Learn, however, so we utilize a
# Gensim helper function, sparse2full, to convert the sparse representation into a
# NumPy array.

class GensimVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, path=None):
        self.path = path
        self.id2word = None
        self.load()

    def load(self):
        if os.path.exists(self.path):
            self.id2word = gensim.corpora.Dictionary.load(self.path)

    def save(self):
        self.id2word.save(self.path)

    def fit(self, documents, labels=None):
        self.id2word = gensim.corpora.Dictionary(documents)
        self.save()

    def transform(self, documents):
        for document in documents:
            docvec = self.id2word.doc2bow(document)
            yield sparse2full(docvec, len(self.id2word))

# We will leave it to the reader to extend this example and investigate
# TF–IDF and distributed representation transformers that are implemented in the
# same fashion



# Creating a custom text normalization transformer

# Many model families suffer from “the curse of dimensionality”; as the feature space
# increases in dimensions, the data becomes more sparse and less informative to the
# underlying decision space. Text normalization reduces the number of dimensions,
# decreasing sparsity. Besides the simple filtering of tokens (removing punctuation and
# stopwords), there are two primary methods for text normalization: stemming and
# lemmatization.

# Stemming uses a series of rules (or a model) to slice a string to a smaller substring.
# The goal is to remove word affixes (particularly suffixes) that modify meaning.

# Lemmatization, on the other hand, uses a dictionary to look up every token
# and returns the canonical “head” word in the dictionary, called a lemma. Because it is
# looking up tokens from a ground truth, it can handle irregular cases as well as handle
# tokens with different parts of speech. For example, the verb 'gardening' should be
# lemmatized to 'to garden', while the nouns 'garden' and 'gardener' are both different
# lemmas. Stemming would capture all of these tokens into a single 'garden'
# token.

# Because it
# only requires us to splice word strings, stemming is faster. Lemmatization, on the
# other hand, requires a lookup to a dictionary or database, and uses part-of-speech
# tags to identify a word’s root lemma, making it noticeably slower than stemming, but
# also more effective.

# Because it
# only requires us to splice word strings, stemming is faster. Lemmatization, on the
# other hand, requires a lookup to a dictionary or database, and uses part-of-speech
# tags to identify a word’s root lemma, making it noticeably slower than stemming, but
# also more effective.

# # We could
# also customize the TextNormalizer to allow uses to choose between stemming and
# lemmatization, and pass the language into the SnowballStemmer.

class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='english'):
        self.stopwords  = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

#     # For filtering extraneous
# tokens, we create two methods. The first, is_punct(), checks if every character
# in the token has a Unicode category that starts with 'P' (for punctuation); the second,
# is_stopword() determines if the token is in our set of stopwords.
    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token):
        return token.lower() in self.stopwords

#     # We can then add a normalize() method that takes a single document composed of a
# list of paragraphs, which are lists of sentences, which are lists of (token, tag) tuples. This method applies the filtering functions to remove unwanted tokens and then lemmatizes them.
    def normalize(self, document):
        return [
            self.lemmatize(token, tag).lower()
            for paragraph in document
            for sentence in paragraph
            for (token, tag) in sentence
            if not self.is_punct(token) and not self.is_stopword(token)
        ]
    # The lemmatize() method first converts the Penn Treebank part-ofspeech
# tags that are the default tag set in the nltk.pos_tag function to WordNet tags,
# selecting nouns by default.
    def lemmatize(self, token, pos_tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    # Finally, we must add the Transformer interface, allowing us to add this class to a
# Scikit-Learn pipeline, which we’ll explore in the next section:
    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            yield self.normalize(document[0])

# Note that text normalization is only one methodology, and also utilizes NLTK very
# heavily, which may add unnecessary overhead to your application. Other options
# could include removing tokens that appear above or below a particular count threshold
# or removing stopwords and then only selecting the first five to ten thousand
# most common words. Yet another option is simply computing the cumulative frequency
# and only selecting words that contain 10%–50% of the cumulative frequency
# distribution. These methods would allow us to ignore both the very low frequency hapaxes (terms that appear only once) and the most common words, enabling us to identify the most potentially predictive terms in the corpus.

# The act of text normalization should be optional and applied carefully
# because the operation is destructive in that it removes information.
# Case, punctuation, stopwords, and varying word
# constructions are all critical to understanding language. Some
# models may require indicators such as case. For example, a named
# entity recognition classifier, because in English, proper nouns are
# capitalized.

# An alternative approach is to perform dimensionality reduction with Principal Component
# Analysis (PCA) or Singular Value Decomposition (SVD), to reduce the feature
# space to a specific dimensionality (e.g., five or ten thousand dimensions) based
# on word frequency. These transformers would have to be applied following a vectorizer
# transformer, and would have the effect of merging together words that are similar
# into the same vector space.


# if __name__ == '__main__':
#     from loader import CorpusLoader
#     from reader import PickledCorpusReader
#
#     corpus = PickledCorpusReader('../corpus')
#     loader = CorpusLoader(corpus, 12)
#
#     docs   = loader.documents(0, test=True)
#     labels = loader.labels(0, test=True)
#     # print(next(docs)[0][0][0])
#     normal = TextNormalizer()
#     normal.fit(docs, labels)
#
#     docs   = list(normal.transform(docs))
#
#     vect = GensimVectorizer('lexicon.pkl')
#     vect.fit(docs)
#     docs = vect.transform(docs)
#     print(next(docs))
