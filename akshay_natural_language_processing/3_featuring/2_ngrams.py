# # If you observe the above methods, each word is considered as a feature.
# There is a drawback to this method.
# It does not consider the previous and the next words, to see if that
# would give a proper and complete meaning to the words.

# For example: consider the word “not bad.” If this is split into individual
# words, then it will lose out on conveying “good” – which is what this word
# actually means.

# N-grams are the fusion of multiple letters or multiple words. They are
# formed in such a way that even the previous and next words are captured.
# • Unigrams are the unique words present in the sentence.
# • Bigram is the combination of 2 words.
# • Trigram is 3 words and so on.

# For example,
# “I am learning NLP”
# Unigrams: “I”, “am”, “ learning”, “NLP”
# Bigrams: “I am”, “am learning”, “learning NLP”
# Trigrams: “I am learning”, “am learning NLP”


#  generating n-grams with TextBlob

Text = "I am learning NLP"

#Import textblob
from textblob import TextBlob

#For unigram : Use n = 1

TextBlob(Text).ngrams(1)

#For Bigram : For bigrams, use n = 2

print(TextBlob(Text).ngrams(2))

# ################## With Count Vectorizer

#importing the function

from sklearn.feature_extraction.text import CountVectorizer

# Text

text = ["I love NLP and I will learn NLP in 2month "]

# create the transform

vectorizer = CountVectorizer(ngram_range=(2,2))

# tokenizing

vectorizer.fit(text)

# encode document

vector = vectorizer.transform(text)

# summarize & generating output
print(vectorizer.vocabulary_)
print(vector.toarray())