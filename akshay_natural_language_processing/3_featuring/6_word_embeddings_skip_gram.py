# Even though all previous methods solve most of the problems, once
# we get into more complicated problems where we want to capture the
# semantic relation between the words, these methods fail to perform.

# All these techniques fail to capture the context and
# meaning of the words. All the methods discussed so
# far basically depend on the appearance or frequency
# of the words. But we need to look at how to capture the
# context or semantic relations: that is, how frequently
# the words are appearing close by.
#
# a. I am eating an apple.
# b. I am using apple.
# If you observe the above example, Apple gives different meanings
# when it is used with different (close by) adjacent words, eating and using.

# The answer to the above questions lies in creating a representation
# for words that capture their meanings, semantic relationships, and the
# different types of contexts they are used in.
# The above challenges are addressed by Word Embeddings

# Word embedding is the feature learning technique where words from
# the vocabulary are mapped to vectors of real numbers capturing the
# contextual hierarchy.

# If you observe the below table, every word is represented with 4
# numbers called vectors. Using the word embeddings technique, we are
# going to derive those vectors for each and every word so that we can use it
# in future analysis. In the below example, the dimension is 4. But we usually
# use a dimension greater than 100.
#
# Word embeddings are prediction based, and they use shallow neural
# networks to train the model that will lead to learning the weight and using
# them as a vector representation.
#
# word2vec: word2vec is the deep learning Google framework to train
# word embeddings. It will use all the words of the whole corpus and predict
# the nearby words. It will create a vector for all the words present in the
# corpus in a way so that the context is captured. It also outperforms any
# other methodologies in the space of word similarity and word analogies.
#
# There are mainly 2 types in word2vec.
#
# • Skip-Gram
# • Continuous Bag of Words (CBOW)

# Skip-Gram
#
# The skip-gram model (Mikolov et al., 2013)1 is used to predict the
# probabilities of a word given the context of word or words.
#
# Let us take a small sentence and understand how it actually works.
# Each sentence will generate a target word and context, which are the words
# nearby. The number of words to be considered around the target variable
# is called the window size. The table below shows all the possible target
# and context variables for window size 2. Window size needs to be selected
# based on data and the resources at your disposal. The larger the window
# size, the higher the computing power.
#
# look skip_gram.jpg

#
# building skip gram

# As mentioned in Chapter 3, import the text corpus and break it into
# sentences. Perform some cleaning and preprocessing like the removal of
# punctuation and digits, and split the sentences into words or tokens, etc.

sentences = [['I', 'love', 'nlp'],
			['I', 'will', 'learn', 'nlp', 'in', '2','months'],
			['nlp', 'is', 'future'],
			['nlp', 'saves', 'time', 'and', 'solves', 'lot', 'of', 'industry', 'problems'],
			['nlp', 'uses', 'machine', 'learning']]



import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

# training the model
skipgram = Word2Vec(sentences, size =50, window = 3, min_count=1,sg = 1)
print(skipgram)

# access vector for one word
print(skipgram['nlp'])

# Since our vector size parameter was 50, the model gives a vector of size
# 50 for each word.

# save model
skipgram.save('skipgram.bin')

# load model
skipgram = Word2Vec.load('skipgram.bin')

#  plotting

# T – SNE plot

X = skipgram[skipgram.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# create a scatter plot of the projection

pyplot.scatter(result[:, 0], result[:, 1])
words = list(skipgram.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()