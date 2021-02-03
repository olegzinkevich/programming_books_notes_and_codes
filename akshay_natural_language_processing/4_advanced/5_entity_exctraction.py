# In this recipe, we are going to discuss how to identify and extract entities
# from the text, called Named Entity Recognition. There are multiple
# libraries to perform this task like NLTK chunker, StanfordNER, SpaCy,
# opennlp, and NeuroNER; and there are a lot of APIs also like WatsonNLU,
# AlchemyAPI, NERD, Google Cloud NLP API, and many more.

# The simplest way to do this is by using the ne_chunk from NLTK or SpaCy.

# For example, suppose we are building a search engine for an
# e-commerce giant. Below are entities that we can train the model on:
# • Gender
# • Color
# • Brand
# • Product Category
# • Product Type
# • Price
# • Size

# Also, we can build named entity disambiguation using deep
# learning frameworks like RNN and LSTM. This is very important for the
# entities extractor to understand the content in which the entities are
# used. For example, pink can be a color or a brand. NED helps in such
# disambiguation.


sent = "John is studying at Stanford University in California"

#  with NLTK

#import libraries
import nltk
from nltk import ne_chunk
from nltk import word_tokenize

#NER
print(ne_chunk(nltk.pos_tag(word_tokenize(sent)), binary=False))


#  using SPACY

#  using SPACY

# !!!!! to make spacy work, also after pip install spacy,
#  write in cmd: python -m spacy download en
# it will download model

import spacy
# nlp = spacy.load('en')  or if en (may need exact path) doesnt work, use:
nlp = spacy.load('en_core_web_sm')

# Read/create a sentence
doc = nlp(u'Apple is ready to launch new phone worth $10000 in New york time square on 5th of 2021. A news by Alex Walfstein')

for ent in doc.ents:
   print(ent.text, ent.label_)
