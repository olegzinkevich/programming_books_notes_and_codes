# Clustering Documents
# Document clustering, also called text clustering, is a cluster analysis
# on textual documents. One of the typical usages would be document
# management.
#
# Clustering or grouping the documents based on the patterns and
# similarities.
#
# Document clustering yet again includes similar steps, so let’s have a look at
# them:
# 1. Tokenization
# 2. Stemming and lemmatization
# 3. Removing stop words and punctuation
# 4. Computing term frequencies or TF-IDF
# 5. Clustering: K-means/Hierarchical; we can then use
# any of the clustering algorithms to cluster different
# documents based on the features we have generated
# 6. Evaluation and visualization: Finally, the clustering
# results can be visualized by plotting the clusters into
# a two-dimensional space

#  pip install mpdl3  - http://mpld3.github.io/    matplot + d3js in browser

import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from sklearn.metrics.pairwise import cosine_similarity
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS


#Lets use the same complaint dataset we use for classification
Data = pd.read_csv("consumer_complaints.csv",encoding='latin-1')

#selecting required columns and rows
Data = Data[['consumer_complaint_narrative']]
Data = Data[pd.notnull(Data['consumer_complaint_narrative'])]

# lets do the clustering for just 200 documents. Its easier to interpret.
Data_sample=Data.sample(800)

# Preprocessing:

# Remove unwanted symbol

Data_sample['consumer_complaint_narrative'] = Data_sample['consumer_complaint_narrative'].str.replace('XXXX','')

# Convert dataframe to list
complaints = Data_sample['consumer_complaint_narrative'].tolist()

# create the rank of documents – we will use it later

ranks = []
for i in range(1, len(complaints)+1):
    ranks.append(i)

# Stop Words
stopwords = nltk.corpus.stopwords.words('english')

# Load 'stemmer'
stemmer = SnowballStemmer("english")

# Functions for sentence tokenizer, to remove numeric tokens and raw #punctuation

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

# TF-IDF feature engineering

from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf vectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

#fit the vectorizer to data

tfidf_matrix = tfidf_vectorizer.fit_transform(complaints)
terms = tfidf_vectorizer.get_feature_names()
print(tfidf_matrix.shape)

# Clustering using K-means

from sklearn.cluster import KMeans

# Define number of clusters
num_clusters = 6

#Running clustering algorithm
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

#final clusters
clusters = km.labels_.tolist()
complaints_data = { 'rank': ranks, 'complaints': complaints, 'cluster': clusters }
frame = pd.DataFrame(complaints_data, index = [clusters] , columns = ['rank', 'cluster'])

#number of docs per cluster
print(frame['cluster'].value_counts())


# Identify cluster behavior
# Identify which are the top 5 words that are nearest to the cluster centroid

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in complaints:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)

# sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :6]:
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()

#  output
# Cluster 0 words: b'loan', b'payment', b'months', b'time', b'any', b'make',
# Cluster 1 words: b'account', b'credited', b'payment', b'paid', b'report', b'year',
# Cluster 2 words: b'collect', b'debt', b'company', b'year', b'letter', b'received',
# Cluster 3 words: b'report', b'credited', b'credited', b'account', b'information', b'company',
# Cluster 4 words: b'bank', b'account', b'because', b'payment', b"'s", b'days',
# Cluster 5 words: b'pay', b'number', b"n't", b'phone', b'time', b"'s",


# Finally, we plot the clusters:

similarity_distance = 1 - cosine_similarity(tfidf_matrix)

# Convert two components as we're plotting points in a two-dimensional plane
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(similarity_distance)  # shape (n_components, n_samples)
xs, ys = pos[:, 0], pos[:, 1]

# Set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#D2691E'}


# set up cluster names using a dict
cluster_names = {0: 'property, based, assist',
                 1: 'business, card',
                 2: 'authorized, approved, believe',
                 3: 'agreement, application, business',
                 4: 'closed, applied, additional',
                 5: 'applied, card'}

# Finally plot it


# Create data frame that has the result of the MDS and the cluster
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters))
groups = df.groupby('label')

# Set up plot
fig, ax = plt.subplots(figsize=(17, 9))  # set size

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=20,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params( \
        axis='x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')
    ax.tick_params( \
        axis='y',
        which='both',
        left='off',
        top='off',
        labelleft='off')

ax.legend(numpoints=1)
plt.show()

# It basically clusters similar kinds of complaints to 6
# buckets using TF-IDF. We can also use the word embeddings and solve this
# to achieve better clusters.