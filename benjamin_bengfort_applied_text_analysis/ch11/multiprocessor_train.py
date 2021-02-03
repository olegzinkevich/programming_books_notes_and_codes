# работает


# mp_train
# Parallel fit of models
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Sat Dec 16 08:04:57 2017 -0500
#
# ID: mp_train.py [] benjamin@bengfort.com $

# In order to illustrate how multiprocessing can help us perform machine learning on
# text, let’s consider an example where we would like to fit multiple models, crossvalidate
# them, and save them to disk. We will begin by writing three functions to generate
# a naive Bayes model, a logistic regression, and a multilayer perceptron. Each
# function in turn creates three different models, defined by Pipelines, that extract
# text from a corpus located at a specified path. Each task also determines a location to
# write the model to, and reports results using the logging module (more on this in a
# bit):

"""
Parallel fit of models
"""

##########################################################################
## Imports
##########################################################################

import time
import logging
import multiprocessing as mp

from functools import wraps

from reader import PickledCorpusReader
from transformers import TextNormalizer, identity

from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


# Python’s logging module is generally used to coordinate complex logging across multiple
# threads and modules. The logging configuration is at the top of the module,
# outside of any function, so it is executed when the code is imported. In the configuration,
# we can specify the %(processName)s directive, which allows us to determine
# which process is writing the log message. The logger is set to the module’s name so
# that different modules’ log statements can also be disambiguated:

# Logging is not multiprocess-safe for writing to a single file (though
# it is thread-safe). Generally speaking, writing to stdout or stderr
# should be fine, but more complex solutions exist to manage multiprocess
# logging in an application context. As a result, it is a good
# practice to start with logging (instead of print statements) to prepare
# for production environment.

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(processName)-10s %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# decorator  # a simple debugging
# decorator that we will use to compare performance times:
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start
    return wrapper


# The documents() and labels() are helper functions that read the data from the corpus
# reader into a list in memory as follows:

def documents(corpus):
    return [
        list(corpus.docs(fileids=fileid))
        for fileid in corpus.fileids()
    ]


def labels(corpus):
    return [
        corpus.categories(fileids=fileid)[0]
        for fileid in corpus.fileids()
    ]


# While each of our functions can be modified and customized individually, each must
# also share common code. This shared functionality is defined in the train_model()
# function, which creates a PickledCorpusReader from the specified path. The
# train_model() function uses this reader to create instances and labels, compute
# scores using the cross_val_score utility from Scikit-Learn, fit the model, write it to
# disk using joblib (a specialized pickle module used by Scikit-Learn), and return the
# scores:
@timeit
def train_model(path, model, saveto=None, cv=12):
    """
    Trains model from corpus at specified path; constructing cross-validation
    scores using the cv parameter, then fitting the model on the full data and
    writing it to disk at the saveto path if specified. Returns the scores.
    """

    # Note that our train_model function constructs the corpus reader
    # itself (rather than being passed a reader object). When considering
    # multiprocessing, all arguments to functions as well as return
    # objects must be serializable using the pickle module. If we imagine
    # that the CorpusReader is only created in the child processes,
    # there is no need to pickle it and send it back and forth. Complex
    # objects can be difficult to pickle, so while it is possible to pass a
    # CorpusReader to the function, it is sometimes more efficient and
    # simpler to pass only simple data such as strings.

    # Load the corpus data and labels for classification
    corpus = PickledCorpusReader(path)
    X = documents(corpus)
    y = labels(corpus)

    # Compute cross validation scores
    scores = cross_val_score(model, X, y, cv=cv)

    # Fit the model on entire data set
    model.fit(X, y)

    # Write to disk if specified
    if saveto:
        joblib.dump(model, saveto)

    # Return scores as well as training time via decorator
    return scores

# For simplicity, the pipelines for fit_naive_bayes, fit_logis
# tic_regression, and fit_multilayer_perceptron share the first
# two steps, using the text normalizer and vectorizer

def fit_naive_bayes(path, saveto=None, cv=12):

    model = Pipeline([
        ('norm', TextNormalizer()),
        ('tfidf', TfidfVectorizer(tokenizer=identity, lowercase=False)),
        ('clf', MultinomialNB())
    ])

    if saveto is None:
        saveto = "naive_bayes_{}.pkl".format(time.time())

    scores, delta = train_model(path, model, saveto, cv)
    logger.info((
        "naive bayes training took {:0.2f} seconds "
        "with an average score of {:0.3f}"
    ).format(delta, scores.mean()))


def fit_logistic_regression(path, saveto=None, cv=12):
    model = Pipeline([
        ('norm', TextNormalizer()),
        ('tfidf', TfidfVectorizer(tokenizer=identity, lowercase=False)),
        ('clf', LogisticRegression())
    ])

    if saveto is None:
        saveto = "logistic_regression_{}.pkl".format(time.time())

    scores, delta = train_model(path, model, saveto, cv)
    logger.info((
        "logistic regression training took {:0.2f} seconds "
        "with an average score of {:0.3f}"
    ).format(delta, scores.mean()))


def fit_multilayer_perceptron(path, saveto=None, cv=12):
    model = Pipeline([
        ('norm', TextNormalizer()),
        ('tfidf', TfidfVectorizer(tokenizer=identity, lowercase=False)),
        ('clf', MLPClassifier(hidden_layer_sizes=(10,10), early_stopping=True))
    ])

    if saveto is None:
        saveto = "multilayer_perceptron_{}.pkl".format(time.time())

    scores, delta = train_model(path, model, saveto, cv)
    logger.info((
        "multilayer perceptron training took {:0.2f} seconds "
        "with an average score of {:0.3f}"
    ).format(delta, scores.mean()))


@timeit
def sequential(path):
    #Run each fit one after the other
    fit_naive_bayes(path)
    fit_logistic_regression(path)
    fit_multilayer_perceptron(path)


# At long last, we’re ready to actually execute our code in parallel, with a run_parallel
# function. This function takes a path to the corpus as an argument, the argument that
# is shared by all tasks
@timeit
def run_parallel(path):
    tasks = [
        fit_naive_bayes, fit_logistic_regression, fit_multilayer_perceptron,
    ]

    # To keep track of the processes we append them to a procs list before starting the process.
    procs = []
    for task in tasks:
        # for each function in the task list, we
# create an mp.Process object whose name is the name of the task, target is the callable,
# and args and kwargs are specified as a tuple and a dictionary, respectively
        proc = mp.Process(name=task.__name__, target=task, args=(path,))
        procs.append(proc)
        proc.start()

    # At this point, if we did nothing, our main process would exit as the run_parallel
# function is complete, which could cause our child processes to exit prematurely or to
# be orphaned (i.e., never terminate). To prevent this, we loop through each of our
# procs and join them, rejoining each to the main process. This will cause the main
# function to block (wait) until the processes’ join method is called
    for proc in procs:
        proc.join()


if __name__ == '__main__':
    # path = "../corpus"
    run_parallel('C:/Users/810004/Desktop/Html_corpus/')

    # print("beginning sequential tasks")
    # _, delta = sequential(path)
    # print("total sequential fit time: {:0.2f} seconds".format(delta))

    # logger.info("beginning parallel tasks")
    # _, delta = parallel(path)
    # logger.info("total parallel fit time: {:0.2f} seconds".format(delta))
