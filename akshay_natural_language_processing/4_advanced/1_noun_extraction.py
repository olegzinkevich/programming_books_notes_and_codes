# Noun Phrase extraction is important when you want to analyze the “who” # in a sentence. Let’s see an example below using TextBlob

#Import libraries
import nltk
from textblob import TextBlob

#Extract noun
blob = TextBlob("John is learning natural language processing")
for np in blob.noun_phrases:
    print(np)

