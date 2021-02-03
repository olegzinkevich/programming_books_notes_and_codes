# • Let’s say a particular word is appearing in all the documents
# of the corpus, then it will achieve higher importance in
# our previous methods. That’s bad for our analysis.
# • The whole idea of having TF-IDF is to reflect on how
# important a word is to a document in a collection, and
# hence normalizing words appeared frequently in all the
# documents.

# Term frequency (TF): Term frequency is simply the ratio of the count of a
# word present in a sentence, to the length of the sentence.
# TF is basically capturing the importance of the word irrespective of the
# length of the document. For example, a word with the frequency of 3 with
# the length of sentence being 10 is not the same as when the word length of
# sentence is 100 words. It should get more importance in the first scenario;
# that is what TF does.
#
# Inverse Document Frequency (IDF): IDF of each word is the log of
# the ratio of the total number of rows to the number of rows in a particular
# document in which that word is present.
#
# IDF = log(N/n), where N is the total number of rows and n is the
# number of rows in which the word was present
#
# IDF will measure the rareness of a term. Words like “a,” and “the” show
# up in all the documents of the corpus, but rare words will not be there
# in all the documents. So, if a word is appearing in almost all documents,
# then that word is of no use to us since it is not helping to classify or in
# information retrieval. IDF will nullify this problem.
# TF-IDF is the simple product of TF and IDF so that both of the
# drawbacks are addressed, which makes predictions and information
# retrieval relevant.

Text = ["The quick brown fox jumped over the lazy dog.",
"The dog.",
"The fox"]

#Import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

#Create the transform

vectorizer = TfidfVectorizer()

#Tokenize and build vocab

vectorizer.fit(Text)

#Summarize

print(vectorizer.vocabulary_)
print(vectorizer.idf_)

#  output

# {'the': 7, 'quick': 6, 'brown': 0, 'fox': 2, 'jumped': 3, 'over': 5, 'lazy': 4, 'dog': 1}
# [1.69314718 1.28768207 1.28768207 1.69314718 1.69314718 1.69314718
#  1.69314718 1.        ]

# If you observe, “the” is appearing in all the 3 documents and it does
# not add much value, and hence the vector value is 1 (position 7), which is less than all
# the other vector representations of the tokens.

# All these methods or techniques we have looked into so far are based
# on frequency and hence called frequency-based embeddings or features.
# And in the next recipe, let us look at prediction-based embeddings,
# typically called word embeddings.