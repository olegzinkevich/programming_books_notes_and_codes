#!/usr/bin/env python3

# работает

# So which topic modeling algorithm is best? Anecdotally,5 LSA is sometimes considered
# better for learning descriptive topics, which is helpful with longer documents
# and more diffuse corpora. Latent Dirichlet Allocation and non-negative matrix factorization,
# on the other hand, can be better for learning compact topics, which is useful
# for creating succinct labels from topics.

# Now that we have organized our documents into piles, how should we go about labeling
# them and describing their contents? In this section, we’ll explore topic modeling,
# an unsupervised machine learning technique for abstracting topics from collections
# of documents.

# In the next section, we’ll compare three of
# these techniques: Latent Dirichlet Allocation (LDA), Latent Semantic Analysis (LSA),
# and Non-Negative Matrix Factorization (NNMF).


from reader import PickledCorpusReader
# returns a representation of documents as bags-of-words.
from transformers import TextNormalizer

from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer

def identity(words):
    return words


class SklearnTopicModels(object):

    # We begin by creating a class, SklearnTopicModels. The __init__ function instantiates
# a pipeline with our TextNormalizer, CountVectorizer, and Scikit-Learn’s implementation
# of LatentDirichletAllocation.
    def __init__(self, n_topics=50, estimator='LDA'):
        """
        n_topics is the desired number of topics
        To use Latent Semantic Analysis, set estimator to 'LSA',
        To use Non-Negative Matrix Factorization, set estimator to 'NMF',
        otherwise, defaults to Latent Dirichlet Allocation ('LDA').
        """
        self.n_topics = n_topics

        if estimator == 'LSA':
            self.estimator = TruncatedSVD(n_components=self.n_topics)
        elif estimator == 'NMF':
            self.estimator = NMF(n_components=self.n_topics)
        else:
            self.estimator = LatentDirichletAllocation(n_topics=self.n_topics)

        # To use topic models in an application, we need a tunable pipeline that will extrapolate # topics from unstructured text data, and a method for storing the best model so it can
# be used on new, incoming data.

        self.model = Pipeline([
            ('norm', TextNormalizer()),
            ('tfidf', CountVectorizer(tokenizer=identity,
                                      preprocessor=None, lowercase=False)),
            ('model', self.estimator)
        ])

    # We then create a fit_transform method, which will call the internal fit and trans
# form methods of each step of our pipeline
    def fit_transform(self, documents):
        self.model.fit_transform(documents)

        return self.model

#     # Now that we have a way to create and fit the pipeline, we want some mechanism to
# inspect our topics. The topics aren’t labeled, and we don’t have a centroid with which
# to produce a label as we would with centroidal clustering. Instead, we will inspect
# each topic in terms of the words it has the highest probability of generating.
    def get_topics(self, n=25):
        """
        n is the number of top terms to show for each topic
        """
        # We create a get_topics method, which steps through our pipeline object to retrieve
# the fitted vectorizer and extracts the tokens from its get_feature_names() attribute.
# We loop through the components_ attribute of the LDA model, and for each of the
# topics and its corresponding index, we reverse-sort the numbered tokens by weight
# such that the 25 highest weighted terms are ranked first. We then retrieve the corresponding
# tokens from the feature names and store our topics as a dictionary where
# the key is the index of one of the 50 topics and the values are the top words associated
# with that topic.
        vectorizer = self.model.named_steps['tfidf']
        model = self.model.steps[-1][1]
        names = vectorizer.get_feature_names()
        topics = dict()

        for idx, topic in enumerate(model.components_):
            features = topic.argsort()[:-(n - 1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens


        return topics


if __name__ == '__main__':

    corpus = PickledCorpusReader('C:/Users/810004/Desktop/Html_corpus/')

    # We can now instantiate a SklearnTopicModels object, and fit and transform the pipeline
# on our corpus documents.
    # With Sklearn
    skmodel = SklearnTopicModels(estimator='LDA')
    documents   = corpus.docs()

    skmodel.fit_transform(documents)
    #         # We assign the result of the get_topics() attribute (a
# Python dictionary) to a topics variable and unpack the dictionary, printing out the
# corresponding topics and their most informative terms:
    topics = skmodel.get_topics()
    for topic, terms in topics.items():
        print("Topic #{}:".format(topic+1))
        print(terms)
