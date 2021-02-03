# 1. Preprocessing
# Whenever the user enters the search query, it is passed on to the NLP
# preprocessing pipeline:
#
# 1. Removal of noise and stop words
# 2. Tokenization
# 3. Stemming
# 4. Lemmatization
#
# 2. The entity extraction model
# We can build the customized entity recognition model by using any of the
# libraries like StanfordNER or NLTK. Also, we can build named entity disambiguation using deep
# learning frameworks like RNN and LSTM
#
# Ways to train/build NERD model:
# • Named Entity Recognition and Disambiguation
# • Stanford NER with customization
# • Recurrent Neural Network (RNN) – LSTM (Long Short-Term
# Memory) to use context for disambiguation
# • Joint Named Entity Recognition and Disambiguation
#
# 3. Query enhancement/expansion
#
# Say, for example, men’s shoes can also be called as male shoes, men’s sports shoes,
# men’s formal shoes, men’s loafers, men’s sneakers. Use locally-trained word embedding (using Word2Vec / GloVe Model ) to achieve this.
#
# 4. Use a search platform
# Search platforms such as Solr or Elastic Search have major features that
# include full-text search hit highlighting, faceted search, real-time indexing,
# dynamic clustering, and database integration.
#
# 5. Learning to rank
# Once the search results are fetched from Solr or Elastic Search, they should
# be ranked based on the user preferences using the past behaviors
#
#
#
#
#
