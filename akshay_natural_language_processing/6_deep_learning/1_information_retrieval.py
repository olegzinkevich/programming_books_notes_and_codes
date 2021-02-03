# An information retrieval (IR) system allows users to efficiently
# search documents and retrieve meaningful information based on a
# search text/query.

# Information retrieval using word embeddings
#
# There are multiple ways to do Information retrieval. But we will see how to
# do it using word embeddings, which is very effective since it takes context
# also into consideration. We discussed how word embeddings are built in
# Chapter 3. We will just use the pretrained word2vec in this case.
#
# Let’s take a simple example and see how to build a document retrieval
# using query input. Let’s say we have 4 documents in our database as
# below. (Just showcasing how it works. We will have too many documents
# in a real-world application.)


# As mentioned earlier, we are going to use the word embeddings to solve
# this problem. Download word2vec from the below link:

# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

import gensim
from gensim.models import Word2Vec
import numpy as np
import nltk
import itertools
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import scipy
from scipy import spatial
from nltk.tokenize.toktok import ToktokTokenizer
import re
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')


# Let’s take a simple example and see how to build a document retrieval
# using query input. Let’s say we have 4 documents in our database as
# below. (Just showcasing how it works. We will have too many documents
# in a real-world application.)

# Randomly taking sentences from internet

Doc1 = [
    "With the Union cabinet approving the amendments to the Motor Vehicles Act, 2016, those caught for drunken driving will have to have really deep pockets, as the fine payable in court has been enhanced to Rs 10,000 for first-time offenders."]

Doc2 = [
    "Natural language processing (NLP) is an area of computer science and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data."]

Doc3 = [
    "He points out that public transport is very good in Mumbai and New Delhi, where there is a good network of suburban and metro rail systems."]

Doc4 = [
    "But the man behind the wickets at the other end was watching just as keenly. With an affirmative nod from Dhoni, India captain Rohit Sharma promptly asked for a review. Sure enough, the ball would have clipped the top of middle and leg."]

# Put all the documents in one list

fin = Doc1 + Doc2 + Doc3 + Doc4

# Assume we have numerous documents like this. And you want to retrieve
# the most relevant once for the query “cricket.” Let’s see how to build it.

query = "cricket"

#https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

#load the model

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

#Preprocessing

def remove_stopwords(text, is_lower_case=False):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', ''.join(text))
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# Function to get the embedding vector for n dimension, we have used "300"

def get_embedding(word):
    if word in model.wv.vocab:
        return model[word]
    else:
        return np.zeros(300)

# For every document, we will get a lot of vectors based on the number of
# words present. We need to calculate the average vector for the document
# through taking a mean of all the word vectors.

# Getting average vector for each document
out_dict =  {}
for sen in fin:
    average_vector = (np.mean(np.array([get_embedding(x) for x in nltk.word_tokenize(remove_stopwords(sen))]), axis=0))
    dict = { sen : (average_vector) }
    out_dict.update(dict)

# Function to calculate the similarity between the query vector and document vector

def get_sim(query_embedding, average_vector_doc):
    sim = [(1 - scipy.spatial.distance.cosine(query_embedding, average_vector_doc))]
    return sim

# Rank all the documents based on the similarity to get relevant docs

def Ranked_documents(query):
    query_words =  (np.mean(np.array([get_embedding(x) for x in nltk.word_tokenize(query.lower())],dtype=float), axis=0))
    rank = []
    for k,v in out_dict.items():
        rank.append((k, get_sim(query_words, v)))
    rank = sorted(rank,key=lambda t: t[1], reverse=True)
    print('Ranked Documents :')
    return rank

# Let’s see how the information retrieval system we built is working with a
# couple of examples.

# Call the IR function with a query
Ranked_documents("cricket")

# If you see, doc4 (on top in result), this will be most relevant for the
# query “cricket” even though the word “cricket” is not even mentioned once
# with the similarity of 0.449.

# Let’s take one more example as may be driving.
Ranked_documents("driving")

# Again, since driving is connected to transport and the Motor Vehicles
# Act, it pulls out the most relevant documents on top. The first 2 documents
# are relevant to the query

# We can use the same approach and scale it up for as many documents
# as possible.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# For more accuracy, we can build our own embeddings, as we
# learned in Chapter 3, for specific industries since the one we are using is
# generalized.

# This is the fundamental approach that can be used for many
# applications like the following:
# • Search engines
# • Document retrieval
# • Passage retrieval
# • Question and answer

# It’s been proven that results will be good when queries are longer and
# the result length is shorter. That’s the reason we don’t get great results in
# search engines when the search query has lesser number of words.