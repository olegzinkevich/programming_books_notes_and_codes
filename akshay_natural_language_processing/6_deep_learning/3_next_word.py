# todo пересмотреть этот файл!

# Autofill/showing what could be the potential sequence of words saves a # lot of time while writing emails and makes users happy to use it in any # product.

# Problem
# You want to build a model to predict/suggest the next word based on a
# previous sequence of words using Email Data.
#
# In this section, we will build an LSTM model to learn sequences of words from email data. We will use this model to predict the next word.

import pandas as pd

file_content = pd.read_csv('spam.csv', encoding = "ISO-8859-1")

# Just selecting emails and connverting it into list
Email_Data = file_content[[ 'v2']]

list_data = Email_Data.values.tolist()

import numpy as np
import random
import pandas as pd
import sys
import os
import time
import codecs
import collections
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from nltk.tokenize import sent_tokenize, word_tokenize
import scipy
from scipy import spatial
from nltk.tokenize.toktok import ToktokTokenizer
import re
tokenizer = ToktokTokenizer()



#Converting list to string
from collections import Iterable


def flatten(items):
    """Yield items from any nested iterable"""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


TextData=list(flatten(list_data))
TextData = ''.join(TextData)

# Remove unwanted lines and converting into lower case
TextData = TextData.replace('\n','')
TextData = TextData.lower()

pattern = r'[^a-zA-z0-9\s]'
TextData = re.sub(pattern, '', ''.join(TextData))

# Tokenizing

tokens = tokenizer.tokenize(TextData)
tokens = [token.strip() for token in tokens]

# get the distinct words and sort it

word_counts = collections.Counter(tokens)
word_c = len(word_counts)
print(word_c)

distinct_words = [x[0] for x in word_counts.most_common()]
distinct_words_sorted = list(sorted(distinct_words))


# Generate indexing for all words

word_index = {x: i for i, x in enumerate(distinct_words_sorted)}


# decide on sentence lenght

sentence_length = 25


#prepare the dataset of input to output pairs encoded as integers
# Generate the data for the model

#input = the input sentence to the model with index
#output = output of the model with index

InputData = []
OutputData = []

for i in range(0, word_c - sentence_length, 1):
    X = tokens[i:i + sentence_length]
    Y = tokens[i + sentence_length]
    InputData.append([word_index[char] for char in X])
    OutputData.append(word_index[Y])

print (InputData[:1])
print ("\n")
print(OutputData[:1])


# Generate  X
X = numpy.reshape(InputData, (len(InputData), sentence_length, 1))


# One hot encode the output variable
Y = np_utils.to_categorical(OutputData)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
file_name_path = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(file_name_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint]



# We can now fit the model to the data. Here we use 5 epochs and a
# batch size of 128 patterns. For better results, you can use more epochs like
# 50 or 100. And of course, you can use them on more data.
# fit the model
model.fit(X, Y, epochs=5, batch_size=128, callbacks=callbacks)


# load the network weights
file_name = "weights-improvement-05-6.8213.hdf5"
model.load_weights(file_name)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Generating random sequence
start = numpy.random.randint(0, len(InputData))
input_sent = InputData[start]

# Generate index of the next word of the email

X = numpy.reshape(input_sent, (1, len(input_sent), 1))
predict_word = model.predict(X, verbose=0)
index = numpy.argmax(predict_word)

print(input_sent)
print ("\n")
print(index)

# Convert these indexes back to words

word_index_rev = dict((i, c) for i, c in enumerate(tokens))
result = word_index_rev[index]
sent_in = [word_index_rev[value] for value in input_sent]

print(sent_in)
print ("\n")
print(result)