# Part of speech (POS) tagging is another crucial part of natural language
# processing that involves labeling the words with a part of speech such as
# noun, verb, adjective, etc. POS is the base for Named Entity Resolution,
# Sentiment Analysis, Question Answering, and Word Sense Disambiguation.

# There are 2 ways a tagger can be built:

# • Rule based - Rules created manually, which tag a word
# belonging to a particular POS.

# • RStochastic based - These algorithms capture the
# sequence of the words and tag the probability of the
# sequence using hidden Markov models.

text  =  "I love NLP and I will learn NLP in 2 month"

# Importing necessary packages and stopwords
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
stop_words = set(stopwords.words('english'))

# Tokenize the text
tokens = sent_tokenize(text)

#Generate tagging for all the tokens using loop
for i in tokens:
    words = nltk.word_tokenize(i)
    words = [w for w in words if not w in stop_words]
    #  POS-tagger.
    tags = nltk.pos_tag(words)

print(tags)