# We want to build a text classification model using CNN, RNN, and LSTM.

# Email classification (spam or ham). We need to classify spam or ham email
# based on email content.

import pandas as pd

#read file
file_content = pd.read_csv('spam.csv', encoding = "ISO-8859-1")

#check sample content in the email
file_content['v2'][1]

#Import library
from nltk.corpus import stopwords
from nltk import *
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.tokenize.toktok import ToktokTokenizer
from keras.preprocessing.text import Tokenizer

# Remove stop words
stop = stopwords.words('english')
file_content['v2'] = file_content['v2'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# Delete unwanted columns
Email_Data = file_content[['v1', 'v2']]

# Rename column names
Email_Data = Email_Data.rename(columns={"v1":"Target", "v2":"Email"})
Email_Data.head()

#Delete punctuations, convert text in lower case and delete the double space

Email_Data['Email'] = Email_Data['Email'].apply(lambda x: re.sub('[!@#$:).;,?&]', '', x.lower()))
Email_Data['Email'] = Email_Data['Email'].apply(lambda x: re.sub(' ', ' ', x))
Email_Data['Email'].head(5)

#Separating text(input) and target classes

list_sentences_rawdata = Email_Data["Email"].fillna("_na_").values
list_classes = ["Target"]
target = Email_Data[list_classes].values


To_Process=Email_Data[['Email', 'Target']]

# Data preparation for model building

# We are building the models using different deep learning approaches
# like CNN, RNN, LSTM, and Bidirectional LSTM and comparing the
# performance of each model using different accuracy metrics.

# We can now define our CNN model.
# Here we define a single hidden layer with 128 memory units. The
# network uses a dropout with a probability of 0.5. The output layer is a
# dense layer using the softmax activation function to output a probability
# prediction.

# Train and test split with 80:20 ratio
train, test = train_test_split(To_Process, test_size=0.2)

# Define the sequence lengths, max number of words and embedding dimensions
# Sequence length of each sentence. If more, truncate. If less, pad with zeros

MAX_SEQUENCE_LENGTH = 300

# Top 20000 frequently occurring words
MAX_NB_WORDS = 20000

# Get the frequently occurring words
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train.Email)
train_sequences = tokenizer.texts_to_sequences(train.Email)
test_sequences = tokenizer.texts_to_sequences(test.Email)

# dictionary containing words and their index
word_index = tokenizer.word_index
# print(tokenizer.word_index)
# total words in the corpus
print('Found %s unique tokens.' % len(word_index))

# get only the top frequent words on train
train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# get only the top frequent words on test
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

print(train_data.shape)
print(test_data.shape)

train_labels = train['Target']
test_labels = test['Target']

#import library

from sklearn.preprocessing import LabelEncoder
# converts the character array to numeric array. Assigns levels to unique labels.

le = LabelEncoder()
le.fit(train_labels)
train_labels = le.transform(train_labels)
test_labels = le.transform(test_labels)

print(le.classes_)
print(np.unique(train_labels, return_counts=True))
print(np.unique(test_labels, return_counts=True))

# changing data types
labels_train = to_categorical(np.asarray(train_labels))
labels_test = to_categorical(np.asarray(test_labels))
print('Shape of data tensor:', train_data.shape)
print('Shape of label tensor:', labels_train.shape)
print('Shape of label tensor:', labels_test.shape)

EMBEDDING_DIM = 100
print(MAX_SEQUENCE_LENGTH)

# We can now define our RNN model.

# Import Libraries
import sys, os, re, csv, codecs, numpy as np, pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D, Conv1D, SimpleRNN
from keras.models import Model
from keras.models import Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential

print('Training CNN 1D model.')

model = Sequential()
model.add(Embedding(MAX_NB_WORDS,
 EMBEDDING_DIM,
 input_length=MAX_SEQUENCE_LENGTH
 ))
model.add(Dropout(0.5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))


model.compile(loss='categorical_crossentropy',
 optimizer='rmsprop',
 metrics=['acc'])

model.fit(train_data, labels_train,
 batch_size=64,
 epochs=5,
 validation_data=(test_data, labels_test))

#predictions on test data

predicted=model.predict(test_data)
predicted

#model evaluation

import sklearn
from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(labels_test, predicted.round())

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

print("############################")

print(sklearn.metrics.classification_report(labels_test, predicted.round()))


#import library
from keras.layers.recurrent import SimpleRNN

#model training

print('Training SIMPLERNN model.')

model = Sequential()
model.add(Embedding(MAX_NB_WORDS,
 EMBEDDING_DIM,
 input_length=MAX_SEQUENCE_LENGTH
 ))
model.add(SimpleRNN(2, input_shape=(None,1)))

model.add(Dense(2,activation='softmax'))

model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])

model.fit(train_data, labels_train,
 batch_size=16,
 epochs=5,
 validation_data=(test_data, labels_test))


# prediction on test data
predicted_Srnn=model.predict(test_data)
predicted_Srnn

#model evaluation

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(labels_test, predicted_Srnn.round())

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

print("############################")

print(sklearn.metrics.classification_report(labels_test, predicted_Srnn.round()))

#model training

print('Training LSTM model.')

model = Sequential()
model.add(Embedding(MAX_NB_WORDS,
 EMBEDDING_DIM,
 input_length=MAX_SEQUENCE_LENGTH
 ))
model.add(LSTM(output_dim=16, activation='relu', inner_activation='hard_sigmoid',return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(2,activation='softmax'))

model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])

model.fit(train_data, labels_train,
 batch_size=16,
 epochs=5,
 validation_data=(test_data, labels_test))


#prediction on text data
predicted_lstm=model.predict(test_data)
predicted_lstm

#model evaluation

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(labels_test, predicted_lstm.round())

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

print("############################")

print(sklearn.metrics.classification_report(labels_test, predicted_lstm.round()))



#model training

print('Training Bidirectional LSTM model.')

model = Sequential()
model.add(Embedding(MAX_NB_WORDS,
 EMBEDDING_DIM,
 input_length=MAX_SEQUENCE_LENGTH
 ))
model.add(Bidirectional(LSTM(16, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
model.add(Conv1D(16, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform"))
model.add(GlobalMaxPool1D())
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.1))

model.add(Dense(2,activation='softmax'))

model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])

model.fit(train_data, labels_train,
 batch_size=16,
 epochs=3,
 validation_data=(test_data, labels_test))



# prediction on test data

predicted_blstm=model.predict(test_data)
predicted_blstm

#model evaluation

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(labels_test, predicted_blstm.round())

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

print("############################")

print(sklearn.metrics.classification_report(labels_test, predicted_blstm.round()))