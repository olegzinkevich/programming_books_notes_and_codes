# First introduced by David Blei, Andrew Ng, and Michael Jordan in 2003, Latent
# Dirichlet Allocation (LDA) is a topic discovery technique. It belongs to the generative
# probabilistic model family, in which topics are represented as the probability that
# each of a given set of terms will occur. Documents can in turn be represented in
# terms of a mixture of these topics. A unique feature of LDA models is that topics are
# not required to be distinct, and words may occur in multiple topics; this allows for a
# kind of topical fuzziness that is useful for handling the flexibility of language

# Blei et al. (2003) found that the Dirichlet prior, a continuous mixture distribution (a
# way of measuring a distribution over distributions), is a convenient way of discovering
# topics that occur across a corpus and also manifest in different mixtures within
# each document in the corpus.1 In effect, with a Latent Dirichlet Allocation, we are
# given an observed word or token, from which we attempt to model the probability of
# topics, the distribution of words for each topic, and the mixture of topics within a
# document.


# LSA  Latent Semantic Analysis
#
# Latent Semantic Analysis (LSA) is a vector-based approach first suggested as a topic
# modeling technique by Deerwester et al in 1990.2
# While Latent Dirichlet Allocation works by abstracting topics from documents,
# which can then be used to score documents by their proportion of topical terms,
# Latent Semantic Analysis simply finds groups of documents with the same words.
# The LSA approach to topic modeling (also known as Latent Semantic Indexing) identifies
# themes within a corpus by creating a sparse term-document matrix, where each
# row is a token and each column is a document. Each value in the matrix corresponds
# to the frequency with which the given term appears in that document, and can be
# normalized using TF–IDF. Singular Value Decomposition (SVD) can then be applied
# to the matrix to factorize into matrices that represent the term-topics, the topic
# importances, and the topic-documents.
#
# Using the derived diagonal topic importance matrix, we can identify the topics that
# are the most significant in our corpus, and remove rows that correspond to less
# important topic terms. Of the remaining rows (terms) and columns (documents), we
# can assign topics based on their highest corresponding topic importance weights.


# Non-Negative Matrix Factorization

Another unsupervised technique that can be used for topic modeling is non-negative
matrix factorization (NNMF). First introduced by Pentti Paatero and Unto Tapper
(1994)3 and popularized in a Nature article by Daniel Lee and H. Sebastian Seung
(1999),4 NNMF has many applications, including spectral data analysis, collaborative
filtering for recommender systems, and topic extraction