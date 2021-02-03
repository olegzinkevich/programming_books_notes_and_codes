#!/usr/bin/env python3

# # Frequency Vectors

# The simplest vector encoding model is to simply fill in the vector with the frequency
# of each word as it appears in the document. In this encoding scheme, each document
# is represented as the multiset of the tokens that compose it and the value for each
# word position in the vector is its count. This representation can either be a straight
# count (integer) encoding as shown in Figure 4-2 or a normalized encoding where
# each word is weighted by the total number of words in the document.

# look frequency_vectors.png


# vectorization techniques:

# The choice of a specific vectorization technique will be largely driven by the problem
# space. Similarly, our choice of implementation—whether NLTK, Scikit-Learn, or
# Gensim—should be dictated by the requirements of the application. For instance,
# NLTK offers many methods that are especially well-suited to text data, but is a big
# dependency. Scikit-Learn was not designed with text in mind, but does offer a robust
# API and many other conveniences (which we’ll explore later in this chapter) particularly
# useful in an applied context. Gensim can serialize dictionaries and references in
# matrix market format, making it more flexible for multiple platforms. However,
# unlike Scikit-Learn, Gensim doesn’t do any work on behalf of your documents for
# tokenization or stemming.

# To set this up, let’s create a list of our documents and tokenize them for the proceeding
# vectorization examples.


import nltk
import string


# The tokenize method performs some lightweight nor‐malization, stripping punctuation using the string.punctuation character set and setting the text to lowercase. This function also performs some feature reduction using the SnowballStemmer to remove affixes such as plurality (“bats” and “bat” are the same token). The examples in the next section will utilize this example corpus and some will use the tokenization method.


# Tokenization function
def tokenize(text):
    stem = nltk.stem.SnowballStemmer('english')
    text = text.lower()

    for token in nltk.word_tokenize(text):
        if token in string.punctuation: continue
        yield stem.stem(token)


# The corpus object
corpus = [
    "The elephant sneezed at the sight of potatoes.",
    "Bats can see via echolocation. See the bat sight sneeze!",
    "Wondering, she opened the door to the studio.",
]

# # NLTK expects features as a dict object whose keys are the names of the features and
# whose values are boolean or numeric. To encode our documents in this way, we’ll create
# a vectorize function that creates a dictionary whose keys are the tokens in the
# document and whose values are the number of times that token appears in the document.
def nltk_frequency_vectorize(corpus):
    # The NLTK frequency vectorize method
    from collections import defaultdict

    def vectorize(doc):
#         # The defaultdict object allows us to specify what the dictionary will return for a key
# that hasn’t been assigned to it yet. By setting defaultdict(int) we are specifying that
# a 0 should be returned, thus creating a simple counting dictionary.
        features = defaultdict(int)

        for token in tokenize(doc):
            features[token] += 1

        return features
    #We can map this function to every item in the corpus using the last line of code, creating an iterable of vectorized documents.
    return map(vectorize, corpus)


def sklearn_frequency_vectorize(corpus):
    # The Scikit-Learn frequency vectorize method
    from sklearn.feature_extraction.text import CountVectorizer
    # The CountVectorizer transformer from the sklearn.feature_extraction model
# has its own internal tokenization and normalization methods. The fit method of the
# vectorizer expects an iterable or list of strings or file objects, and creates a dictionary
# of the vocabulary on the corpus.
    vectorizer = CountVectorizer()
    # When transform is called, each individual document
# is transformed into a sparse array whose index tuple is the row (the document
# ID) and the token ID from the dictionary, and whose value is the count:
    return vectorizer.fit_transform(corpus)

# reccomend: Vectors can become extremely sparse, particularly as vocabularies
# get larger, which can have a significant impact on the speed and
# performance of machine learning models. For very large corpora, it
# is recommended to use the Scikit-Learn HashingVectorizer,
# which uses a hashing trick to find the token string name to feature
# index mapping. This means it uses very low memory and scales to
# large datasets as it does not need to store the entire vocabulary and
# it is faster to pickle and fit since there is no state. However, there is
# no inverse transform (from vector to text), there can be collisions,
# and there is no inverse document frequency weighting.


# Gensim’s frequency encoder is called doc2bow. To use doc2bow, we first create a Gensim
# Dictionary that maps tokens to indices based on observed order (eliminating the
# overhead of lexicographic sorting). The dictionary object can be loaded or saved to
# disk, and implements a doc2bow library that accepts a pretokenized document and
# returns a sparse matrix of (id, count) tuples where the id is the token’s id in the
# dictionary. Because the doc2bow method only takes a single document instance, we
# use the list comprehension to restore the entire corpus, loading the tokenized documents
# into memory so we don’t exhaust our generator:
def gensim_frequency_vectorize(corpus):
    # The Gensim frequency vectorize method
    import gensim

    tokenized_corpus = [list(tokenize(doc)) for doc in corpus]
    id2word = gensim.corpora.Dictionary(tokenized_corpus)
    return [id2word.doc2bow(doc) for doc in tokenized_corpus]


# Because they disregard grammar and the relative position of words in documents,
# frequency-based encoding methods suffer from the long tail, or Zipfian distribution,
# that characterizes natural language. As a result, tokens that occur very frequently are
# orders of magnitude more “significant” than other, less frequent ones. This can have a
# significant impact on some models (e.g., generalized linear models) that expect normally
# distributed features.
#
# # A solution to this problem is one-hot encoding, a boolean vector encoding method
# that marks a particular vector index with a value of true (1) if the token exists in the
# document and false (0) if it does not. In other words, each element of a one-hot enco ded vector reflects either the presence or absence of the token in the described text

# One-hot encoding reduces the imbalance issue of the distribution of tokens, simplifying
# a document to its constituent components. This reduction is most effective for
# very small documents (sentences, tweets) that don’t contain very many repeated elements

def nltk_one_hot_vectorize(corpus):
    # The NLTK one hot vectorize method
    def vectorize(doc):
        return {
            # In addition to the boolean dictionary values, it is also acceptable to use an integer value; 1 for present and 0 for absent.
            token: True
            for token in tokenize(doc)
        }

    return map(vectorize, corpus)


def sklearn_one_hot_vectorize(corpus):
    # The Sklearn one hot vectorize method

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import Binarizer

    # Note that we could also use CountVectorizer(binary=True) to achieve one-hot encoding in the above, obviating the Binarizer.
    freq = CountVectorizer()
    vectors = freq.fit_transform(corpus)

    print(len(vectors.toarray()[0]))
    # The Binarizer takes only numeric data, so the text data
# must be transformed into a numeric space using the CountVectorizer ahead of onehot
# encoding  The Binarizer class uses a threshold value (0 by default). All values of the vector that are less than or equal to the threshold are set to zero, while those that are greater than the threshold are set to 1. Therefore, by default, the Binarizer converts all frequency values to 1 while maintaining the zero-valued frequencies.

    # The OneHotEncoder treats each vector component (column)
# as an independent categorical variable, expanding the dimensionality
# of the vector for each observed value in each column. In this
# case, the component (sight, 0) and (sight, 1) would be treated
# as two categorical dimensions rather than as a single binary encoded
# vector component.
    onehot = Binarizer()
    vectors = onehot.fit_transform(vectors.toarray())

    print(len(vectors[0]))


def gensim_one_hot_vectorize(corpus):
    # The Gensim one hot vectorize method
    import gensim
    import numpy as np

    corpus = [list(tokenize(doc)) for doc in corpus]
    id2word = gensim.corpora.Dictionary(corpus)
    # One-hot encoding represents similarity and difference at the document level, but
# because all words are rendered equidistant, it is not able to encode per-word similarity.
# Moreover, because all words are equally distant, word form becomes incredibly important; the tokens “trying” and “try” will be equally distant from unrelated tokens like “red” or “bicycle”!
#     sp Normalizing tokens to a single word class, either through stemming or lemmatization, will help.

    # Extending the code from the Gensim
# frequency vectorization example in the previous section, we can one-hot encode our
# vectors with our id2word dictionary. To get our vectors, an inner list comprehension
# converts the list of tuples returned from the doc2bow method into a list of (token_id,
# 1) tuples and the outer comprehension applies that converter to all documents in the
# corpus
    corpus = np.array([
        [(token[0], 1) for token in id2word.doc2bow(doc)]
        for doc in corpus
    ])

    return corpus

# The bag-of-words representations that we have explored so far only describe a document
# in a standalone fashion, not taking into account the context of the corpus. A
# better approach would be to consider the relative frequency or rareness of tokens in
# the document against their frequency in other documents. The central insight is that
# meaning is most likely encoded in the more rare terms from a document. For example,
# in a corpus of sports text, tokens such as “umpire,” “base,” and “dugout” appear
# more frequently in documents that discuss baseball, while other tokens that appear
# frequently throughout the corpus, like “run,” “score,” and “play,” are less important.

# TF–IDF, term frequency–inverse document frequency, encoding normalizes the frequency
# of tokens in a document with respect to the rest of the corpus. This encoding
# approach accentuates terms that are very relevant to a specific instance,
# look tf-idf.jpg

# TF–IDF is computed on a per-term basis, such that the relevance of a token to a
# document is measured by the scaled frequency of the appearance of the term in the
# document, normalized by the inverse of the scaled frequency of the term in the entire
# corpus\

# We interpret the score to mean that the closer the
# TF–IDF score of a term is to 1, the more informative that term is to that document.
# The closer the score is to zero, the less informative that term is.

# One benefit of TF–IDF is that it naturally addresses the problem of stopwords, As a result TF–IDF is widely used for bag-of-words models, and is an excellent starting point for most text analytics

def nltk_tfidf_vectorize(corpus):
    from nltk.text import TextCollection
    # With NLTK To vectorize text in this way with NLTK, we use the TextCollection class, a wrapper # for a list of texts or a corpus consisting of one or more texts.
    corpus = [list(tokenize(doc)) for doc in corpus]
    texts = TextCollection(corpus)

    # Because TF–IDF requires the entire corpus, our new version of vectorize does not
# accept a single document, but rather all documents. After applying our tokenization
# function and creating the text collection, the function goes through each document in
# the corpus and yields a dictionary whose keys are the terms and whose values are the
# TF–IDF score for the term in that particular document.
    for doc in corpus:
        yield {
            term: texts.tf_idf(term, doc)
            for term in doc
        }


def sklearn_tfidf_vectorize(corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Under
# the hood, the TfidfVectorizer uses the CountVectorizer estimator we used to produce
# the bag-of-words encoding to count occurrences of tokens, followed by a Tfidf
# Transformer, which normalizes these occurrence counts by the inverse document
# frequency.

    # The input for a TfidfVectorizer is expected to be a sequence of filenames, file-like
# objects, or strings that contain a collection of raw documents, similar to that of the
# CountVectorizer. As a result, a default tokenization and preprocessing method is
# applied unless other functions are specified. The vectorizer returns a sparse matrix
# representation in the form of ((doc, term), tfidf) where each key is a document
# and term pair and the value is the TF–IDF score.

    tfidf = TfidfVectorizer()
    return tfidf.fit_transform(corpus)


def gensim_tfidf_vectorize(corpus):
    import gensim

    corpus = [list(tokenize(doc)) for doc in corpus]
    lexicon = gensim.corpora.Dictionary(corpus)

    tfidf = gensim.models.TfidfModel(dictionary=lexicon, normalize=True)
    vectors = [tfidf[lexicon.doc2bow(vector)] for vector in corpus]

#     # Gensim provides helper functionality to write dictionaries and models to disk in a
# compact format, meaning you can conveniently save both the TF–IDF model and the
# lexicon to disk in order to load them later to vectorize new documents.
    lexicon.save_as_text('test.txt')
    tfidf.save('tfidf.pkl')
    # This will save the lexicon as a text-delimited text file, sorted lexicographically, and the
# TF–IDF model as a pickled sparse matrix. Note that the Dictionary object can also
# be saved more compactly in a binary format using its save method, but
# save_as_text allows easy inspection of the dictionary for later work. To load the
# models from disk:
# lexicon = gensim.corpora.Dictionary.load_from_text('lexicon.txt')
# tfidf = gensim.models.TfidfModel.load('tfidf.pkl')

    return vectors



# Distributed representation

# When document similarity is important in the context of an application, we instead
# encode text along a continuous scale with a distributed representation, as shown in
# Figure 4-5. This means that the resulting document vector is not a simple mapping
# from token position to token score. Instead, the document is represented in a feature
# space that has been embedded to represent word similarity. The complexity of this
# space (and the resulting vector length) is the product of how the mapping to that representation
# is learned.

# Word2vec, created by a team of researchers at Google led by Tomáš Mikolov, implements
# a word embedding model that enables us to create these kinds of distributed
# representations. The word2vec algorithm trains word representations based on either
# a continuous bag-of-words (CBOW) or skip-gram model, such that words are
# embedded in space along with similar words based on their context.

# The doc2vec1 algorithm is an extension of word2vec. It proposes a paragraph vector—
# an unsupervised algorithm that learns fixed-length feature representations from variable
# length documents. This representation attempts to inherit the semantic properties
# of words such that “red” and “colorful” are more similar to each other than they
# are to “river” or “governance.” Moreover, the paragraph vector takes into consideration
# the ordering of words within a narrow context, similar to an n-gram model. The
# combined result is much more effective than a bag-of-words or bag-of-n-grams
# model because it generalizes better and has a lower dimensionality but still is of a
# fixed length so it can be used in common machine learning algorithms.

# Neither NLTK nor Scikit-Learn provide implementations of these kinds of word
# embeddings. Gensim’s implementation allows users to train both word2vec and
# doc2vec models on custom corpora and also conveniently comes with a model that is
# pretrained on the Google news corpus.

def gensim_doc2vec_vectorize(corpus):
    from gensim.models.doc2vec import TaggedDocument, Doc2Vec

    # We can train our own model as follows. First, we use a list comprehension to load our
# corpus into memory.
    corpus = [list(tokenize(doc)) for doc in corpus]
    # Next, we create a list of TaggedDocument objects, which
# extend the LabeledSentence, and in turn the distributed representation of word2vec.
# TaggedDocument objects consist of words and tags. We can instantiate the tagged
# document with the list of tokens along with a single tag, one that uniquely identifies
# the instance. In this example, we’ve labeled each document as "d{}".format(idx),
# e.g. d0, d1, d2 and so forth.
    docs = [
        TaggedDocument(words, ['d{}'.format(idx)])
        for idx, words in enumerate(corpus)
    ]
    # Once we have a list of tagged documents, we instantiate the Doc2Vec model and specify
# the size of the vector as well as the minimum count, which ignores all tokens that have a frequency less than that number. The size parameter is usually not as low a
# dimensionality as 5; we selected such a small number for demonstration purposes
# only. We also set the min_count parameter to zero to ensure we consider all tokens,
# but generally this is set between 3 and 5, depending on how much information the
# model needs to capture.
    model = Doc2Vec(docs, size=5, min_count=0)
    return model.docvecs
# Once instantiated, an unsupervised neural network is trained
# to learn the vector representations, which can then be accessed via the docvecs
# property
print(gensim_doc2vec_vectorize(corpus)[0])

# Distributed representations will dramatically improve results over TF–IDF models
# when used correctly. The model itself can be saved to disk and retrained in an active
# fashion, making it extremely flexible for a variety of use cases


# vAgain, the choice of vectorization technique (as well as the library implementation)
# tend to be use case- and application-specific,


# Later in this chapter we will explore the Scikit-Learn Pipeline object, which enables
# us to streamline vectorization together with later modeling phrases. As such, we often
# prefer to use vectorizers that conform to the Scikit-Learn API.