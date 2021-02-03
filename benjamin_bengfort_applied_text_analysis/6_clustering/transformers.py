#!/usr/bin/env python3

import os
import nltk
import gensim
import unicodedata

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from gensim.matutils import sparse2full
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim.sklearn_api import lsimodel, ldamodel

class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='english'):
        self.stopwords  = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def normalize(self, document):
        return [
            self.lemmatize(token, tag).lower()
            for paragraph in document
            for sentence in paragraph
            for (token, tag) in sentence
            if not self.is_punct(token) and not self.is_stopword(token)
        ]

    def lemmatize(self, token, pos_tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        return [
            self.normalize(document)
            for document in documents
        ]


#         # To use Gensim’s LdaTransformer, we need to create a custom Scikit-Learn wrapper
# for Gensim’s TfidfVectorizer so that it can function inside a Scikit-Learn Pipeline.
# GensimTfidfVectorizer will vectorize our documents ahead of LDA, as well as saving,
# holding, and loading a custom-fitted lexicon and vectorizer for later use.
class GensimTfidfVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, dirpath=".", tofull=False):
        """
        Pass in a directory that holds the lexicon in corpus.dict and the
        TFIDF model in tfidf.model (for now).

        Set tofull = True if the next thing is a Scikit-Learn estimator
        otherwise keep False if the next thing is a Gensim model.
        """
        self._lexicon_path = os.path.join(dirpath, "corpus.dict")
        self._tfidf_path = os.path.join(dirpath, "tfidf.model")

        self.lexicon = None
        self.tfidf = None
        self.tofull = tofull

        self.load()

#     # If the model has already been fit, we can initialize the GensimTfidfVectorizer with a
# lexicon and vectorizer that can be loaded from disk using the load method. We also
# implement a save() method, which we will call after fitting the vectorizer.
    def load(self):

        if os.path.exists(self._lexicon_path):
            self.lexicon = Dictionary.load(self._lexicon_path)

        if os.path.exists(self._tfidf_path):
            self.tfidf = TfidfModel().load(self._tfidf_path)

    def save(self):
        self.lexicon.save(self._lexicon_path)
        self.tfidf.save(self._tfidf_path)

#     # Next, we implement fit() by creating a Gensim Dictionary object, which takes as
# an argument a list of normalized documents. We instantiate a Gensim TfidfModel,
# passing in as an argument the list of documents, each of which have been passed
# through lexicon.doc2bow, and been transformed into bags of words. We then call
# the save method, which serializes our lexicon and vectorizer and saves them to disk.
    def fit(self, documents, labels=None):
        self.lexicon = Dictionary(documents)
        self.tfidf = TfidfModel([self.lexicon.doc2bow(doc) for doc in documents], id2word=self.lexicon)
        self.save()
        return self

    # We then implement our transform() method, which creates a generator that loops
# through each of our normalized documents and vectorizes them using the fitted
# model and their bag-of-words representation.
    def transform(self, documents):
        def generator():
            for document in documents:
                vec = self.tfidf[self.lexicon.doc2bow(document)]
                # Because the next step in our pipeline
# will be a Gensim model, we initialized our vectorizer to set tofull=False, so that it
# would output a sparse document format (a sequence of 2-tuples). However, if we
# were going to use a Scikit-Learn estimator next, we would want to initialize our
# GensimTfidfVectorizer with tofull=True, which here in our transform method would convert the sparse format into the needed dense representation for Scikit-Learn, an np array.
                if self.tofull:
                    yield sparse2full(vec)
                else:
                    yield vec
        return list(generator())


if __name__ == '__main__':
    from reader import PickledCorpusReader

    corpus = PickledCorpusReader('../corpus')
    docs = [
        list(corpus.docs(fileids=fileid))[0]
        for fileid in corpus.fileids()
    ]

    model = Pipeline([
        ('norm', TextNormalizer()),
        ('vect', GensimTfidfVectorizer()),
        ('lda', ldamodel.LdaTransformer())])

    model.fit_transform(docs)

    print(model.named_steps['norm'])
