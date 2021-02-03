# We can start with a pipeline
# that normalizes our text, vectorizes it, and then passes it directly into a classifier.
# This will allow us to compare different text classification models such as Naive Bayes,
# Logistic Regression, and Support Vector Machines. Finally, we can apply a feature
# reduction technique such as Singular Value Decomposition to see if that improves
# our modeling.  The end result is that we’ll be constructing six classification models: one for each of
# the three models and for the two pipeline combinations as shown in Figure 5-5. We
# will go ahead and use the default hyperparameters for each of these models initially so
# that we can start getting results.


from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

def identity(words):
    return words

def create_pipeline(estimator, reduction=False):
    steps = [
        ('normalize', TextNormalizer()),
        ('vectorize', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False))
    ]

    if reduction:
        steps.append((
            'reduction', TruncatedSVD(n_components=10000)
        ))

    # Add the estimator
    steps.append(('classifier', estimator))
    return Pipeline(steps)


# We can now quickly generate our models as follows:

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
models = []
for form in (LogisticRegression, MultinomialNB, SGDClassifier):
    models.append(create_pipeline(form(), True))
    models.append(create_pipeline(form(), False))

# # The models list now contains six model forms—instantiated pipelines that include
# specific vectorization and feature extraction methods (feature analysis), a specific
# algorithm, and specific hyperparameters (currently set to the Scikit-Learn defaults).

# Fitting the models given a training dataset of documents and their associated labels
# can be done as follows:
for model in models:
    model.fit(train_docs, train_labels)

# By calling the fit() method on each model, the documents and labels from the training
# dataset are sent into the beginning of each pipeline. The transformers have their
# fit() methods called, then the data is passed into their transform() method. The
# transformed data is then passed to the fit() of the next transformer for each step in
# the sequence. The final estimator, in this case one of our classification algorithms, will
# have its fit() method called on the completely transformed data



# evaluate models

# Let’s compare our models. We’ll use the CorpusLoader we created in “Streaming
# access to k splits” on page 88 to get our train_test_splits for our cross-validation
# folds. Then, for each fold, we will fit the model on the training data and the accompanying
# labels, then create a prediction vector off the test data. Next, we will pass the
# actual and the predicted labels for each fold to a score function and append the score
# to a list. Finally, we will average the results across all folds to get a single score for the
# model.

import numpy as np
from sklearn.metrics import accuracy_score

for model in models:
    scores = [] # Store a list of scores for each split

    for X_train, X_test, y_train, y_test in loader:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)
    print("Accuracy of {} is {:0.3f}".format(model, np.mean(scores)))


# Do certain models perform better for one class over another? Is there one poorly performing
# class that is bringing the global accuracy down? How often does the fitted
# classifier guess one class over another? In order to get insight into these factors, we
# need to look at a per-class evaluation: enter the confusion matrix.

from sklearn.metrics import classification_report

model = create_pipeline(SGDClassifier(), False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, labels=labels))

# 		precision recall f1-score support
# books     0.85    0.73    0.79    15
# cinema    0.63    0.60    0.62    20
# cooking   0.75    1.00    0.86    3
# gaming    0.85    0.79    0.81    28
# sports    0.93    1.00    0.96    26
# tech      0.77    0.82    0.79    33
# avg / total 0.81  0.81    0.81    125

# The precision of a class, A, is computed as the ratio between the number of correctly
# predicted As (true As) to the total number of predicted As (true As plus false As). Precision
# shows how accurately a model predicts a given class according to the number of
# times it labels that class as true.

# The recall of a class A is computed as the ratio between the number of predicted As
# (true As) to the total number of As (true As + false ¬As). Recall, also called sensitivity, is
# a measure of how often relevant classes are retrieved.

# The support of a class shows how many test instances were involved in computing the
# scores. As we can see in the classification report above, the cooking class is potentially
# under-represented in our sample, meaning there are not enough documents to
# inform its score.

# Finally, the F1 score is the harmonic mean of precision and recall and embeds more
# information than simple accuracy by taking into account how each class contributes
# to the overall score.

# In an application, we want to be able to retrain our models on some
# routine basis as new data is ingested. This training process will
# happen under the hood, and should result in updates to the
# deployed model depending on whichever model is currently most
# performant. As such, it is convenient to build these scoring mechanisms
# into the application’s logs, so that we can go back and examine
# # shifts in precision, recall, F1 score, and training time over time.
#
# We can tabulate all the model scores and sort by F1 score in order to select the best
# model through some minor iteration and score collection

import tabulate
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score

fields = ['model', 'precision', 'recall', 'accuracy', 'f1']
table = []

for model in models:
    scores = defaultdict(list) # storage for all our model metrics

    # k-fold cross-validation
    for X_train, X_test, y_train, y_test in loader:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Add scores to our scores
        scores['precision'].append(precision_score(y_test, y_pred))
        scores['recall'].append(recall_score(y_test, y_pred))
        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['f1'].append(f1_score(y_test, y_pred))

    # Aggregate our scores and add to the table.
    row = [str(model)]
    for field in fields[1:]
        row.append(np.mean(scores[field]))
    table.append(row)

# Sort the models by F1 score descending
table.sort(key=lambda row: row[-1], reverse=True)
print(tabulate.tabulate(table, headers=fields))


# Model Operationalization

# Now that we have identified the best performing model, it is time to save the model to
# disk in order to operationalize it. Machine learning techniques are tuned toward creating
# models that can make predictions on new data in real time, without verification.
# To employ models in applications, we first need to save them to disk so that they can
# be loaded and reused. For the most part, the best way to accomplish this is to use the
# pickle module:

import pickle
from datetime import datetime
time = datetime.now().strftime("%Y-%m-%d")

path = 'hobby-classifier-{}'.format(time)
# The model is saved along with the date that it was built.
with open(path, 'wb') as f:
    pickle.dump(model, f)

# Model fitting is a routine process, and generally speaking,
# models should be retrained at regular intervals appropriate to the
# velocity of your data. Graphing model performance over time and
# making determinations about data decay and model adjustments is
# a crucial part of machine learning applications

# Using model in real world

# To use the model in an application with new, incoming text, simply load the estimator
# from the pickle object, and use its predict() method.

import nltk

def preprocess(text):
    return [
        [
            list(nltk.pos_tag(nltk.word_tokenize(sent)))
            for sent in nltk.sent_tokenize(para)
        ] for para in text.split("\n\n")
    ]


# Because our vectorization process is embedded with our model via the Pipeline we
# need to ensure that the input to the pipeline is prepared in a manner identical to the
# training data input. Our training data was preprocessed text, so we need to include a
# function to preprocess strings into the same format. We can then open the pickle file,
# load the model, and use its predict() method to return labels.

with open(path, 'rb') as f:
    model = pickle.load(f)
    model.predict([preprocess(doc) for doc in newdocs])

