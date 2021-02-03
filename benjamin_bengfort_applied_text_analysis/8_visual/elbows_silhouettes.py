# работает

# Scikit-Learn models are often surprisingly successful with little to no modification of
# the default hyperparameters. Rather than a matter of luck, this is a signal of the substantial
# amount of experience and domain expertise that have been contributed to the
# library. Nonetheless, after we have arrived at the suite of models we find most successful
# for our problem, the next step of the process is to experiment with tuning the
# hyperparameters so that we can arrive at the most optimal settings for each model.

# In this section, we will demonstrate how to explore hyperparameters visually, specifically
# to steer k-selection for k-means clustering problems.

# As we saw in Chapter 6, k-means is a simple unsupervised machine learning algorithm
# that groups data into a specified number k of clusters. Because the user must
# specify in advance what k to choose, the algorithm is somewhat naive—it assigns all
# members to k clusters whether or not it is the right k for the dataset. The Yellowbrick
# library provides two mechanisms for selecting an optimal k parameter for centroidal
# clustering, silhouette scores and elbow curves, which we’ll explore in this section.

import os
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets.base import Bunch
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer

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


corpus = load_corpus('hobbies')
tfidf  = TfidfVectorizer(stop_words='english')
docs   = tfidf.fit_transform(corpus.data)


# The silhouette coefficient is used when the ground-truth about the dataset is
# unknown, instead computing the density of clusters produced by the model. A silhouette
# score can then be calculated by averaging the silhouette coefficient for each
# sample, computed as the difference between the average intracluster distance and the
# mean nearest-cluster distance for each sample, normalized by the maximum value.
# This produces a score between 1 and -1, where 1 is highly dense clusters, -1 is completely
# incorrect clustering, and values near zero indicate overlapping clusters. The
# higher the score the better, because the clusters are denser and more separate. Negative
# values imply that samples have been assigned to the wrong cluster, and positive
# values mean that there are discrete clusters. The scores can then be plotted to display
# a measure of how close each point in one cluster is to points in the neighboring clusters.
# Instantiate the clustering model and visualizer

# In order to create the visualization, we first
# train the clustering model, instantiate the visualizer, fit it on the corpus, and then call
# the visualizer’s poof() method:
visualizer = SilhouetteVisualizer(KMeans(n_clusters=6))
visualizer.fit(docs)
visualizer.poof()

# The SilhouetteVisualizer displays the silhouette coefficient for each sample on a
# per-cluster basis, visualizing which clusters are dense and which are not. The vertical
# thickness of the plotted cluster indicates its size, and the dashed red line is the global
# average.

# Another visual technique that can be used for k selection is the elbow method. The
# elbow method visualizes multiple clustering models with different values for k. Model
# selection is based on whether or not there is an “elbow” in the curve (i.e., if the curve
# looks like an arm with a clear change in angle from one part of the curve to another).
#
# the elbow
# method runs k-means clustering on the dataset for each value of k and computes the
# silhouette_score, the mean silhouette coefficient for all samples. When poof() is
# called, the silhouette score for each k is plotted:
#
# If the line chart looks like an arm, then the “elbow” (the point of inflection on the
# curve) is the best value of k; we want as small a k as possible such that the clusters do
# not overlap. If the data isn’t very clustered, the elbow method may not always work
# well, resulting either in a smooth curve or a very jumpy line

# Instantiate the clustering model and visualizer
visualizer = KElbowVisualizer(KMeans(), metric='silhouette', k=[4,10])
visualizer.fit(docs)
visualizer.poof()
