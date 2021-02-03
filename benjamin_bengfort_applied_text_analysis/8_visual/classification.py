# работает

# In traditional classification pipelines, fitted models can be optimized and then
# described with respect to their precision, recall, and F1 scores. We can visualize these
# measures using confusion matrices, classification heatmaps, and ROC-AUC curves.
#
# A classification report is a text summary of the main metrics for assessing the success
# of a classifier: precision, the ability not to label an instance positive that is actually
# negative; recall, the ability to find all positive instances; and f1 score, a weighted harmonic
# mean of precision and recall. While the Scikit-Learn metrics module does
# expose a classification_report method, we find that the Yellowbrick version,
# which integrates numerical scores with a color-coded heatmap, supports easier interpretation
# and problem detection.
#
# To use Yellowbrick to create a classification heatmap, we load our corpus as in “Loading
# Yellowbrick Datasets” on page 165, TF–IDF vectorize the documents and create
# train and test splits. We then instantiate a ClassificationReport, pass in the desired
# classifier, and the names of the classes, the call fit and score, which call the internal
# Scikit-Learn fitting and scoring mechanisms for the model. Finally, we call poof on
# the visualizer, which adds the requisite labeling and coloring to the plot and then calls
# Matplotlib’s draw:

import os

from sklearn.datasets.base import Bunch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression

from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError

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
tfidf  = TfidfVectorizer()

docs   = tfidf.fit_transform(corpus.data)
labels = corpus.target

X_train, X_test, y_train, y_test = train_test_split(docs.toarray(), labels, test_size=0.2, random_state=42)

visualizer = ClassificationReport(GaussianNB(), classes=corpus.categories)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.poof()

visualizer = ClassificationReport(SGDClassifier(), classes=corpus.categories)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.poof()

visualizer = ConfusionMatrix(LogisticRegression(), classes=corpus.categories)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.poof()

visualizer = ConfusionMatrix(MultinomialNB(), classes=corpus.categories)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.poof()
