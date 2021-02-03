# A count vectorizer and co-occurrence matrix have one limitation though.
# In these methods, the vocabulary can become very large and cause
# memory/computation issues.
# One of the ways to solve this problem is a Hash Vectorizer.

# Hash Vectorizer is memory efficient and instead of storing the tokens
# as strings, the vectorizer applies the hashing trick to encode them as
# numerical indexes. The downside is that it’s one way and once vectorized,
# the features cannot be retrieved.


from sklearn.feature_extraction.text import HashingVectorizer

# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]

# Transform
vectorizer = HashingVectorizer(n_features=10)

# create the hashing vector
vector = vectorizer.transform(text)

# summarize the vector
print(vector.shape)
print(vector.toarray())

# It created vector of size 10 and now this can be used for any
# supervised/unsupervised tasks.