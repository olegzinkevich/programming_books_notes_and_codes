# работает

# # After feature analysis and engineering, the next phase of the model selection triple
# workflow is model selection. In practice, we will select and compare multiple models,
# since it is generally very difficult to predict in advance which model will be most
# effective with a new corpus. Thus, our next task is to determine when our models are
# performing well or poorly. In a traditional machine learning context, we can rely on model performance scores
# —such as mean square error or coefficient of determination in the case of regression,
# and precision, accuracy, and F1 score for classification—to determine which models
# are strongest. These techniques can also be extended to the context of visual analytics
# Just as we looked for small-scale indications of separability and diffuseness using our
# frequency distribution plots, we should also investigate the degree of document similarity
# across all features. One very popular method for doing so is to use the nonlinear
# dimensionality reduction method t-distributed stochastic neighbor embedding,
# or t-SNE.
#
# Scikit-Learn implements the t-SNE decomposition method as the sklearn.mani
# fold.TSNE transformer. By decomposing high-dimensional document vectors into
# two dimensions using probability distributions from both the original dimensionality
# and the decomposed dimensionality, t-SNE is able to effectively cluster similar documents.
# By decomposing to two or three dimensions, the documents can be visualized
# with a scatterplot.
#
# Unfortunately, t-SNE is very computationally expensive, so typically a simpler
# decomposition method such as SVD or PCA is applied ahead of time.
# The Yellowbrick
#
# library exposes a TSNEVisualizer, which creates an inner transformer pipeline
# that applies such a decomposition first (SVD with 50 components by default), then
# performs the t-SNE embedding. The TSNEVisualizer expects document vectors, so
# we will use the TfidfVectorizer from Scikit-Learn in advance of passing the documents
# into the TSNEVisualizer fit method:


import os
from sklearn.datasets.base import Bunch
from yellowbrick.text import TSNEVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer

# The path to the test data sets
FIXTURES = os.path.join(os.getcwd(), "data")

# Corpus loading mechanisms
corpora = {
    "hobbies": os.path.join(FIXTURES, "hobbies")
}


def load_corpus(name, download=True):
    """
    Loads and wrangles the passed in text corpus by name.
    If download is specified, this method will download any missing files.

    Note: This function is slightly different to the `load_data` function
    used above to load pandas dataframes into memory.
    """

    # Get the path from the datasets
    path = corpora[name]

    # Check if the data exists, otherwise download or raise
    if not os.path.exists(path):
        raise ValueError((
            "'{}' dataset has not been downloaded, "
            "use the download.py module to fetch datasets"
        ).format(name))

    # Read the directories in the directory as the categories.
    categories = [
        cat for cat in os.listdir(path)
        if os.path.isdir(os.path.join(path, cat))
    ]

    files = []  # holds the file names relative to the root
    data = []  # holds the text read from the file
    target = []  # holds the string of the category

    # Load the data from the files in the corpus
    for cat in categories:
        for name in os.listdir(os.path.join(path, cat)):
            files.append(os.path.join(path, cat, name))
            target.append(cat)

            with open(os.path.join(path, cat, name), 'r', encoding='utf-8', errors='ignore') as f:
                data.append(f.read())

    # Return the data bunch for use similar to the newsgroups example
    return Bunch(
        categories=categories,
        files=files,
        data=data,
        target=target,
    )


# Load the data and create document vectors
corpus = load_corpus('hobbies')
# As mentioned before, the TSNEVisualizer expects vectorized text as input, and in this case, we have used TF–IDF
tfidf  = TfidfVectorizer()

docs   = tfidf.fit_transform(corpus.data)
labels = corpus.target

# Create a visualizer to simply see the vectors plotted in 2D
# To speed the rendering, the TSNEVisualizer also employs decomposition
# ahead of the stochastic neighbor embedding, defaulting to using a sparse method
# (TruncatedSVD); we might also experiment with a dense method like PCA, which we
# can do by passing decompose = "pca" into TSNEVisualizer() upon initialization.
tsne = TSNEVisualizer()
tsne.fit(docs)
tsne.poof()


# Create a visualizer to see how k-means clustering grouped the docs
from sklearn.cluster import KMeans

# When used in conjunction with a clustering algorithm, TSNEVisualizer can also be
# used for visualizing clusters. Used this way, the technique can help to assess the
# efficacy of one clustering method over another. Here, we’ll use sklearn.clus
# ter.KMeans, set the number of clusters to 5, and then pass the resulting
# cluster.labels_ attribute as y into the TSNEVisualizer fit() method:
clusters = KMeans(n_clusters=5)
clusters.fit(docs)

tsne = TSNEVisualizer()
tsne.fit(docs, ["c{}".format(c) for c in clusters.labels_])
tsne.poof()


# Create a visualizer to see how the classes are distributed. a version of the graph where the colors of points are associated with the categorical labels corresponding to the documents.
# If we were interested in exploring only a few of the categories
# within our corpus, this is as easy as passing in a classes parameter into the # TSNEVisualizer upon instantiation, with a list of the strings representing # the different subcategories (e.g., TSNEVisualizer(classes=['sports', 'cinema',
# 'gaming']))
tsne = TSNEVisualizer()
tsne.fit(docs, labels)
tsne.poof()

# What we’re looking for in such graphs are spatial similarities between the points
# (documents) and any other discernible patterns. Figure 8-11 displays a projection of
# the vectorized Baleen hobbies corpus in two dimensions using t-SNE. The result is a
# scatterplot of the vectorized corpus, where each point represents a document or utterance.
# The distance between two points in the visual space is embedded using the
# probability distribution of pairwise similarities in the higher dimensionality; thus our
# TSNEVisualizer shows clusters of similar documents in the hobbies corpus and the
# relationships between groups of documents