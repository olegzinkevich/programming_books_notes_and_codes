# Classifying Text
# Text classification – The aim of text classification is to automatically classify
# the text documents based on pretrained categories.

# Applications:

# • Sentiment Analysis
# • Document classification
# • Spam – ham mail classification
# • Resume shortlisting
# • Document summarization

#  examples spam detection

# Please download data from the below link and save it in your working
# directory:

# https://www.kaggle.com/uciml/sms-spam-collection-dataset#
# spam.csv

import pandas as pd
#Read the data
Email_Data = pd.read_csv("spam.csv",encoding ='latin1')

#Data undestanding
Email_Data.columns

Email_Data = Email_Data[['v1', 'v2']]
Email_Data = Email_Data.rename(columns={"v1":"Target", "v2":"Email"})

print(Email_Data.head())

# Text processing and feature engineering

#import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import sklearn.feature_extraction.text as text
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

#pre processing steps like lower case, stemming and lemmatization

Email_Data['Email'] = Email_Data['Email'].apply(lambda x: " ".join(x.lower() for x in x.split()))
stop = stopwords.words('english')
Email_Data['Email'] = Email_Data['Email'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
st = PorterStemmer()
Email_Data['Email'] = Email_Data['Email'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
Email_Data['Email'] =Email_Data['Email'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

print(Email_Data.head())



#Splitting data into train and validation

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(Email_Data['Email'], Email_Data['Target'])

# TFIDF feature generation for a maximum of 5000 features

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(Email_Data['Email'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

print(xtrain_tfidf.data)

# This is the generalized function for training any given model:

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    return metrics.accuracy_score(predictions, valid_y)

# Naive Bayes trainig
accuracy = train_model(naive_bayes.MultinomialNB(alpha=0.2), xtrain_tfidf, train_y, xvalid_tfidf)
print ("Accuracy: ", accuracy)

# Naive Bayes is giving better results than the linear classifier. We can try
# many more classifiers and then choose the best one.

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("Accuracy: ", accuracy)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#  how to use it on the new incoming data?