#!/usr/bin/env python3

# не работает

# Now that we have organized our documents into piles, how should we go about labeling
# them and describing their contents? In this section, we’ll explore topic modeling,
# an unsupervised machine learning technique for abstracting topics from collections
# of documents.

# In the next section, we’ll compare three of
# these techniques: Latent Dirichlet Allocation (LDA), Latent Semantic Analysis (LSA),
# and Non-Negative Matrix Factorization (NNMF).


from reader import PickledCorpusReader
# returns a representation of documents as bags-of-words.
from transformers import TextNormalizer, GensimTfidfVectorizer

from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer

from gensim.sklearn_api import lsimodel, ldamodel

def identity(words):
    return words


# Gensim also exposes an implementation for Latent Dirichlet Allocation, which offers
# some convenient attributes over Scikit-Learn. Conveniently, Gensim (starting with
# version 2.2.0) provides a wrapper for its LDAModel, called ldamodel.LdaTransformer,
# which makes integration with a Scikit-Learn pipeline that much more convenient.
class GensimTopicModels(object):

    def __init__(self, n_topics=10, estimator='LSA'):
        """
        n_topics is the desired number of topics

        To use Latent Semantic Analysis, set estimator to 'LSA'
        otherwise defaults to Latent Dirichlet Allocation.
        """
        self.n_topics = n_topics

        if estimator == 'LSA':
            self.estimator = lsimodel.LsiTransformer(num_topics=self.n_topics)
        else:
            self.estimator = ldamodel.LdaTransformer(num_topics=self.n_topics)

        self.model = Pipeline([
            ('norm', TextNormalizer()),
            ('vect', GensimTfidfVectorizer()),
            ('model', self.estimator)
        ])

    def fit(self, documents):
        self.model.fit(documents)

        return self.model


if __name__ == '__main__':
    corpus = PickledCorpusReader('C:/Users/810004/Desktop/Html_corpus/')






#
    gmodel = GensimTopicModels(estimator='LSA')

    docs = [list(corpus.docs(fileids=fileid))[0]
            for fileid in corpus.fileids()]

    gmodel.fit(docs)

    # # retrieve the fitted lsa model from the named steps of the pipeline
    lsa = gmodel.model.named_steps['model'].gensim_model

    # # show the topics with the token-weights for the top 10 most influential tokens:
    print(lsa.print_topics(10))


    # # retrieve the fitted lda model from the named steps of the pipeline
    lda = gmodel.model.named_steps['model'].gensim_model
    #
    # # show the topics with the token-weights for the top 10 most influential tokens:
    print(lda.print_topics(10))

    corpus = [
        gmodel.model.named_steps['vect'].lexicon.doc2bow(doc)
        for doc in gmodel.model.named_steps['norm'].transform(docs)]

    id2token = gmodel.model.named_steps['vect'].lexicon.id2token

    for word_id, freq in next(iter(corpus)):
        print(id2token[word_id], freq)
#
#     # We can also define a function get_topics, which given the fitted LDAModel and vectorized
# # corpus, will retrieve the highest-weighted topic for each of the documents in
# # the corpus
#     # # get the highest weighted topic for each of the documents in the corpus
#     def get_topics(vectorized_corpus, model):
#         from operator import itemgetter
#
#         topics = [
#             max(model[doc], key=itemgetter(1))[0]
#             for doc in vectorized_corpus
#         ]
#
#         return topics
#
#     topics = get_topics(corpus,lda)
#
#     for topic, doc in zip(topics, docs):
#         print("Topic:{}".format(topic))
#         print(doc)
#
#     ## retreive the fitted vectorizer or the lexicon if needed
#     tfidf = gmodel.model.named_steps['vect'].tfidf
#     lexicon = gmodel.model.named_steps['vect'].lexicon
