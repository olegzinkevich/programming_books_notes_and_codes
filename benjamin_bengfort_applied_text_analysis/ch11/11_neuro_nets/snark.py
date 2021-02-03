#!/usr/bin/env python3

# работает, но нужно тренировать модель несколько часов на большем кол-ве материалов - запустить preprocess.py и сохранить все 12 000 reviews из sklite

# Our input data is a series of 18,000 reviews of albums from the website Pitchfork.
# com1; each review contains the text of the review, in which the music reviewer
# discusses the relative merits of the album and the band, as well as a floating-point
# numeric score between 0 and 10.
#
# We would like to predict the relative positivity or negativity of a review given the text.
# Scikit-Learn’s neural net module, sklearn.neural_network, enables us to train a
# multilayer perceptron to perform classification or regression using the now familiar
# fit and predict methods.
#
# We’ll attempt both a regression to predict the actual
# numeric score of an album and a classification to predict if the album is “terrible,”
# “okay,” “good,” or “amazing.”


import time
import numpy as np
from functools import wraps

from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start
    return wrapper

# # First, we create a function, documents, to retrieve the pickled, part-of-speech tagged
# documents from our corpus reader object, a continuous function to get the original
# numeric ratings of each album, and a categorical function that uses NumPy’s
# digitize method to bin the ratings into our four categories:

def documents(corpus):
    return list(corpus.reviews())

def continuous(corpus):
    return list(corpus.scores())

def make_categorical(corpus):
    """
    terrible : 0.0 < y <= 3.0
    okay     : 3.0 < y <= 5.0
    great    : 5.0 < y <= 7.0
    amazing  : 7.0 < y <= 10.1
    :param corpus:
    :return:
    """
    return np.digitize(continuous(corpus), [0.0, 3.0, 5.0, 7.0, 10.1])

# Next, we add a train_model function, which will take as input a path to the pickled
# corpus, a Scikit-Learn estimator, and keyword arguments for whether the labels are
# continuous, an optional path for storing the fitted model, and the number of folds to
# use in cross-validation.

# Our function instantiates a corpus reader, calls the documents function as well as
# either continuous or make_categorical to get the input values X and the target values
# y. We then calculate the cross-validated scores, fit and store the model using the
# joblib utility from Scikit-Learn, and return the scores:
@timeit
def train_model(path, model, continuous=True, saveto=None, cv=12):
    """
    Trains model from corpus at specified path; constructing cross-validation
    scores using the cv parameter, then fitting the model on the full data and
    writing it to disk at the saveto path if specified. Returns the scores.
    """
    # Load the corpus data and labels for classification
    corpus = PickledReviewsReader(path)
    X = documents(corpus)
    if continuous:
        y = continuous(corpus)
        scoring = 'r2'
    else:
        y = make_categorical(corpus)
        scoring = 'f1'

    # Compute cross validation scores
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    # Fit the model on entire data set
    model.fit(X, y)

    # Write to disk if specified
    if saveto:
        joblib.dump(model, saveto)

    # Return scores as well as training time via decorator
    return scores


if __name__ == '__main__':

    from sklearn.pipeline import Pipeline
    # As with other Scikit-Learn estimators, MLPRegressor and MLP
# Classifier expect NumPy arrays of floating-point values, and
# while arrays can be dense or sparse, it’s best to scale input vectors
# using one-hot encoding or a standardized frequency encoding.
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer

    from reader import PickledReviewsReader
    from transformer import TextNormalizer, KeyphraseExtractor

    # Path to postpreprocessed, part-of-speech tagged review corpus
    cpath = 'C:/Users/810004/PycharmProjects/Linguistic_parser/venv1/drafts/Benjamin_Bengfort_Applied_Text_Analysis/11_neuro_nets/ch12/review_corpus_proccessed'
    mpath = 'ann_cls.pkl'

    # Similar to choosing k for k-means clustering, selecting the best
# number and size of hidden layers in an initial neural network prototype
# is more art than science.
    # with regressor
    pipeline = Pipeline([
        ('norm', TextNormalizer()), # can use KeyphraseExtractor() instead
        ('tfidf', TfidfVectorizer()),
        ('ann', MLPClassifier(hidden_layer_sizes=[500,150], verbose=True))
    ])

    print("Starting training...")
    scores, delta = train_model(cpath, pipeline, continuous=False, saveto=mpath)

    print("Training complete.")
    for idx, score in enumerate(scores):
        print("Accuracy on slice #{}: {}.".format((idx+1), score))
    print("Total fit time: {:0.2f} seconds".format(delta))
    print("Model saved to {}.".format(mpath))


    # with classifier
    classifier = Pipeline([
    ('norm', TextNormalizer()),
    ('tfidf', TfidfVectorizer()),
    ('ann', MLPClassifier(hidden_layer_sizes=[500,150], verbose=True))
    ])
    classifer_scores = train_model(cpath, classifier, continuous=False)

    print("Starting training...")
    scores, delta = train_model(cpath, classifier, continuous=False, saveto=mpath)

    print("Training complete.")
    for idx, score in enumerate(scores):
        print("Accuracy on slice #{}: {}.".format((idx + 1), score))
    print("Total fit time: {:0.2f} seconds".format(delta))
    print("Model saved to {}.".format(mpath))