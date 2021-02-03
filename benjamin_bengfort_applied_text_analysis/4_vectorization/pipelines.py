# Pipeline objects enable us to integrate a series of transformers that combine normalization,
# vectorization, and feature analysis into a single, well-defined mechanism.
#
# As
# shown in Figure 4-6, Pipeline objects move data from a loader (an object that will
# wrap our CorpusReader from Chapter 2) into feature extraction mechanisms to
# finally an estimator object that implements our predictive models. Pipelines are directed
# acyclic graphs (DAGs) that can be simple linear chains of transformers to arbitrarily
# complex branching and joining paths.
#
# The purpose of a Pipeline is to chain together multiple estimators representing a
# fixed sequence of steps into a single unit. All estimators in the pipeline, except the last
# one, must be transformers—that is, implement the transform method, while the last
# estimator can be of any type, including predictive estimators.
#
# Pipelines provide convenience;
# fit and transform can be called for single inputs across multiple objects at
# once. Pipelines also provide a single interface for grid search of multiple estimators at
# once.
#
# Pipelines are constructed by describing a list of (key, value) pairs where the key is a
# string that names the step and the value is the estimator object. Pipelines can be created
# either by using the make_pipeline helper function, which automatically determines
# the names of the steps, or by specifying them directly.
#
# Pipeline objects are a Scikit-Learn specific utility, but they are also the critical integration
# point with NLTK and Gensim. Here is an example that joins the TextNormal
# izer and GensimVectorizer we created in the last section together in advance of a
# Bayesian model. By using the Transformer API as discussed earlier in the chapter, we
# can use TextNormalizer to wrap NLTK CorpusReader objects and perform preprocessing
# and linguistic feature extraction. Our GensimVectorizer is responsible for
# vectorization, and Scikit-Learn is responsible for the integration via Pipelines, utilities
# like cross-validation, and the many models we will use, from Naive Bayes to Logistic
# Regression.
#

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
model = Pipeline([
    ('normalizer', TextNormalizer()),
    ('vectorizer', GensimVectorizer()),
    ('bayes', MultinomialNB()),
])
#
# The Pipeline can then be used as a single instance of a complete model. Calling
# model.fit is the same as calling fit on each estimator in sequence, transforming the
# input and passing it on to the next step. Other methods like fit_transform behave
# similarly.
# The pipeline will also have all the methods the last estimator in the pipeline
# has. If the last estimator is a transformer, so too is the pipeline. If the last estimator is
# a classifier (MultinominalNB), as in the example above, then the pipeline will also have predict and # score methods so that the entire model can be used as a classifier.

# The estimators in the pipeline are stored as a list, and can be accessed by index. For
# example, model.steps[1] returns the tuple ('vectorizer', GensimVectorizer
# (path=None)). However, common usage is to identify estimators by their names
# using the named_steps dictionary property of the Pipeline object. The easiest way to
# access the predictive model is to use model.named_steps["bayes"] and fetch the
# estimator directly.



# Grid Search for Hyperparameter Optimization

# Grid search can be implemented to modify the parameters
# of all estimators in the Pipeline as though it were a single object. In order to
# access the attributes of estimators, you would use the set_params or get_params
# pipeline methods with a dunderscore representation of the estimator and parameter
# names as follows: estimator__parameter

# Let’s say that we want to one-hot encode only the terms that appear at least three
# times in the corpus; we could modify the Binarizer as follows:
# model.set_params(onehot__threshold=3.0)

# Using this principle, we could execute a grid search by defining the search parameters
# grid using the dunderscore parameter syntax. Consider the following grid search to
# determine the best one-hot encoded Bayesian text classification model:

from sklearn.model_selection import GridSearchCV
search = GridSearchCV(model, param_grid={
    'count__analyzer': ['word', 'char', 'char_wb'],
    'count__ngram_range': [(1,1), (1,2), (1,3), (1,4), (1,5), (2,3)],
    'onehot__threshold': [0.0, 1.0, 2.0, 3.0],
    'bayes__alpha': [0.0, 1.0],
})

# Enriching Feature Extraction with Feature Unions

# Pipelines do not have to be simple linear sequences of steps; in fact, they can be arbitrarily
# complex through the implementation of feature unions. The FeatureUnion
# object combines several transformer objects into a new, single transformer similar to
# the Pipline object
# However, instead of fitting and transforming data in sequence
# through each transformer, they are instead evaluated independently and the results
# are concatenated into a composite vector.

# Consider the example shown in Figure 4-7. We might imagine an HTML parser
# transformer that uses BeautifulSoup or an XML library to parse the HTML and
# return the body of each document. We then perform a feature engineering step,
# where entities and keyphrases are each extracted from the documents and the results
# passed into the feature union. Using frequency encoding on the entities is more sensible
# since they are relatively small, but TF–IDF makes more sense for the keyphrases.
# The feature union then concatenates the two resulting vectors such that our decision
# space ahead of the logistic regression separates word dimensions in the title from
# word dimensions in the body.

# FeatureUnion objects are similarly instantiated as Pipeline objects with a list of
# (key, value) pairs where the key is the name of the transformer, and the value is
# the transformer object Estimator parameters can also be
# accessed in the same fashion, and to implement a search on a feature union, simply
# nest the dunderscore for each transformer in the feature union.


# The feature union is fit in
# sequence with respect to the rest of the pipeline, but each transformer within the feature
# union is fit independently, meaning that each transformer sees the same data as
# # the input to the feature union. During transformation, each transformer is applied in
# parallel and the vectors that they output are concatenated together into a single larger
# vector, which can be optionally weighted,

from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
model = Pipeline([
    ('parser', HTMLParser()),

    ('text_union', FeatureUnion(
    transformer_list = [
    ('entity_feature', Pipeline([
    ('entity_extractor', EntityExtractor()),
    ('entity_vect', CountVectorizer()),
    ])),
    ('keyphrase_feature', Pipeline([
    ('keyphrase_extractor', KeyphraseExtractor()),
    ('keyphrase_vect', TfidfVectorizer()),
    ])),
    ],

    transformer_weights= {
    'entity_feature': 0.6,
    'keyphrase_feature': 0.2,
    }
    )),
    ('clf', LogisticRegression()),
])