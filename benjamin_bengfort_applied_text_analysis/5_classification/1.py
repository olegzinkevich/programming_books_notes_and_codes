# The collection of a corpus of both
# spam and ham emails allowed the construction of a Naive Bayes model—a model that
# uses a uniform prior to predict the probabilities of a word’s presence in both spam
# and ham emails based on its frequency.
#
# Because the target is given ahead of
# time, classification is said to be supervised machine learning because a model can be
# trained to minimize error between predicted and actual categories in the training
# data. Once a classification model is fit, it assigns categorical labels to new instances
# based on the patterns detected during training.
#
# This simple premise gives the opportunity for a huge number of possible applications,
# so long as the application problem can be formulated to identify either a yes/no
# (binary classification) or discrete buckets (multiclass classification).
#
# For example, a
# recommendation system such as the one shown in Figure 5-1 may have classifiers
# that identify a product’s target age (e.g., a youth versus an adult bicycle), gender
# (women’s versus men’s clothing), or category (e.g., electronics versus movies) by classifying
# the product’s description or other attributes. Product reviews may then be
# classified to detect quality or to determine similar products.
#
# Another real-world application
# is the automatic topic classification of text: by using blogs that publish content
# in a single domain (e.g., a cooking blog doesn’t generally discuss cinema), it is possible
# to create classifiers that can detect topics in uncategorized sources such as news
# articles.
#
# #  Naive Baises
#
# The nice thing about the Naive Bayesian method used in the classic spam identification
# problem is that both the construction of the model (requiring only a single pass
# through the corpus) and predictions (computation of a probability via the product of
# an input vector with the underlying truth table) are extremely fast. The performance
# of Naive Bayes meant a machine learning model that could keep up with email-sized
# applications. Accuracy could be further improved by adding nontext features like the
# IP or email address of the sender, the number of included images, the use of numbers
# in spelling "v14gr4", etc.
#
# Naive Bayes is an online model, meaning that it can be updated in real time without
# retraining from scratch (simply update the underlying truth table and token probabilities).
# This meant that email service providers could keep up with spammer reactions
# by simply allowing the user to mark offending emails as spam—updating the underlying
# model for everyone.
#
#     # Workflow
#
# # The classification workflow occurs in two phases: a build phase and an operational
# phase as shown in Figure 5-2. In the build phase, a corpus of documents is transformed
# into feature vectors. The document features, along with their annotated labels
# (the category or class we want the model to learn), are then passed into a classification
# algorithm that defines its internal state along with the learned patterns. Once
# trained or fitted, a new document can be vectorized into the same space as the training
# data and passed to the predictive algorithm, which returns the assigned class label
# for the document.
#
# # Multiclass classification
#
# # Binary classifiers have two classes whose relationship is important: only the two
# classes are possible and one class is the opposite of the other (e.g., on/off, yes/no,
# etc.). In probabilistic terms, a binary classifier with classes A and B assumes that P(B)
# = 1 - P(A). However, this is frequently not the case. Consider sentiment analysis; if a
# document is not positive, is it necessarily negative? If some documents are neutral
# (which is often the case), adding a third class to our classifier may significantly
# increase its ability to identify the positive and negative documents. This then
# becomes a multiclass problem with multiple binary classes—for example, A and ¬A
# (not A) and B and ¬B (not B).
#
