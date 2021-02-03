# The traditional method used for feature engineering is One Hot encoding.
# If anyone knows the basics of machine learning, One Hot encoding is
# something they should have come across for sure at some point of time or
# maybe most of the time. It is a process of converting categorical variables
# into features or columns and coding one or zero for the presence of that
# particular category.

# One Hot Encoding will basically convert characters or words into binary
# numbers as shown below.

#                   I  love NLP is future
# I love NLP        1  1    1   0   0
# NLP is future     0  0    1   1   1

Text = "I am learning NLP"

# Importing the library

import pandas as pd

# Generating the features with Pandas Dummies
print(pd.get_dummies(Text.split()))

# has a disadvantage. It does not take the
# frequency of the word occurring into consideration. If a particular word
# is appearing multiple times, there is a chance of missing the information
# if it is not included in the analysis. A count vectorizer will solve that
# problem.

# Count vectorizer is almost similar to One Hot encoding. The only
# difference is instead of checking whether the particular word is present or
# not, it will count the words that are present in the document.

# #####
# Converting Text to Features Using Count Vectorizing

#importing the function

from sklearn.feature_extraction.text import CountVectorizer

# Text

text = ["I love NLP and I will learn NLP in 2month "]

# create the transform

vectorizer = CountVectorizer()

# tokenizing

vectorizer.fit(text)

# encode document

vector = vectorizer.transform(text)

# summarize & generating output

print(vectorizer.vocabulary_)
print(vector.toarray())

#  output
# {'love': 4, 'nlp': 5, 'and': 1, 'will': 6, 'learn': 3, 'in': 2,
# '2month': 0}
# [[1 1 1 1 1 2 1]]
# The fifth token 'nlp' has appeared twice in the document.

#  next - 2_ngrams.py

# # If you observe the above methods, each word is considered as a feature.
# There is a drawback to this method.
# It does not consider the previous and the next words, to see if that
# would give a proper and complete meaning to the words.