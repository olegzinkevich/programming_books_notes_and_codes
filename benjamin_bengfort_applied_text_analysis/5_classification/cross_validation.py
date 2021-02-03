# One of the biggest challenges of applied machine learning is defining a stopping criterion
# upfront—how do we know when our model is good enough to deploy? When
# is it time to stop tuning? Which model is the best for our use case? Cross-validation is
# an essential tool for scoping these kinds of applications, since it will allow us to compare
# models using training and test splits and estimate in advance which model will be
# most performant for our use case.

# The trick is to walk the line between underfitting and overfitting. An underfit model
# has low variance, generally making the same predictions every time, but with
# extremely high bias, because the model deviates from the correct answer by a significant
# amount. Underfitting is symptomatic of not having enough data points, or not
# training a complex enough model. An overfit model, on the other hand, has memorized
# the training data and is completely accurate on data it has seen before, but
# varies widely on unseen data.
#
# Complexity
# increases with the number of features, parameters, depth, training epochs, etc. As
# complexity increases and the model overfits, the error on the training data decreases,
# but the error on test data increases,
#
# The goal is therefore to find the optimal point with enough model complexity so as to
# avoid underfit (decreasing the bias) without injecting error due to variance. To find
# that optimal point, we need to evaluate our model on data that it was not trained on.
# The solution is cross-validation: a multiround experimental method that partitions
# the data such that part of the data is reserved for testing and not fit upon to reduce
# error due to overfit.
#
# Cross-validation starts by shuffling the data (to prevent any unintentional ordering
# errors) and splitting it into k folds as shown
#
# A common question is what k should be chosen for k-fold crossvalidation.
# We typically use 12-fold cross-validation
#
# A
# higher k provides a more accurate estimate of model error on
# unseen data, but takes longer to fit

