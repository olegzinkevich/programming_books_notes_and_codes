This is, in fact, a task practiced in many disciplines, from medicine to law. At its core,
this sorting task relies on our ability to compare two documents and determine their
similarity. Documents that are similar to each other are grouped together and the
resulting groups broadly describe the overall themes, topics, and patterns inside the
corpus.

Unsupervised learning

Clustering algorithms aim to discover latent structure or themes in unlabeled data
using features to organize instances into meaningfully dissimilar groups.

# clustering pipeline

As we see in the pipeline presented in Figure 6-1, a corpus is
transformed into feature vectors and a clustering algorithm is employed to create
groups or topic clusters, using a distance metric such that documents that are closer
together in feature space are more similar. New incoming documents can then be
vectorized and assigned to the nearest cluster.

# process
- clusters extraction
- With the resulting clusters, we’ll experiment with using Gensim for topic modeling to describe and summarize our clusters.

# Manhattan distance, shown in Figure 6-3 as the three stepped paths, is similar, computed
as the sum of the absolute differences of the Cartesian coordinates. Minkowski
distance is a generalization of Euclidean and Manhattan distance, and defines the distance
between two points in a normalized vector space.
However, as the vocabulary of our corpus grows, so does its dimensionality—and
rarely in an evenly distributed way. For this reason, these distance measures are not
always a very effective measure, since they assume all data is symmetric and that distance
is the same in all dimensions

# By contrast, Mahalanobis distance, shown in Figure 6-4, is a multidimensional generalization
of the measurement of how many standard deviations away a particular
point is from a distribution of points. This has the effect of shifting and rescaling the
coordinates with respect to the distribution. As such, Mahalanobis distance gives us a
slightly more flexible way to define distances between documents; for instance, enabling
us to identify similarities between utterances of different lengths

# Jaccard distance defines similarity between finite sets as the quotient of their intersection
and their union, as shown in Figure 6-5. For instance, we could measure the Jaccard
distance between two documents A and B by dividing the number of unique
words that appear in both A and B by the total number of unique words that appear in
A and B. A value of 0 would indicate that the two documents have nothing in common,
a 1 that they were the same document, and values between 0 and 1 indicating
their relative degree of similarity.

# Edit distance measures the distance between two strings by the number of permutations
needed to convert one into the other. There are multiple implementations of
edit distance, all variations on Levenshtein distance, but with differing penalties for
insertions, deletions, and substitutions, as well as potentially increased penalties for
gaps and transpositions.

# It is also possible to measure distances between vectors. For example, we can define
two document vectors as similar by their TF–IDF distance; in other words, the magnitude
to which they share the same unique terms relative to the rest of the words in the
corpus.

# We can also
measure vector similarity with cosine distance, using the cosine of the angle between the two vectors to assess the degree to which they share the same orientation

# While Euclidean distance is often the default metric used in clustering model hyperparameters
(as we’ll see in the next sections), we frequently find the most success
using cosine distance.



# Partitive
clustering and agglomerative clustering are our two main approaches, and both separate
documents into groups whose members share maximum similarity as defined by
some distance metric. In this section, we will focus on partitive methods, which partition
instances into groups that are represented by a central vector (the centroid) or
described by a density of documents per cluster. Centroids represent an aggregated
value (e.g., mean or median) of all member documents and are a convenient way to
describe documents in that cluster.


# In this section, we will use clustering to establish subcategories within the “news” corpus, which might then be employed as target values for subsequent classification tasks.

k-means clustering

Because it has implementations in familiar libraries like NLTK and Scikit-Learn, kmeans
is a convenient place to start. A popular method for unsupervised learning
tasks, the k-means clustering algorithm starts with an arbitrarily chosen number of
clusters, k, and partitions the vectorized instances into clusters according to their
proximity to the centroids, which are computed to minimize the within-cluster sum
of squares


