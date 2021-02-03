# # Our
# # next step will be to prepare our preprocessed text data for machine learning by
# # encoding it as vectors. We’ll weigh several techniques for vector encoding, and discuss
# # how to wrap that encoding process in a pipeline to allow for systematic loading,
# # normalization, and feature extraction. Finally, we’ll discuss how to reunite the extracted
# # features to allow for more complex analysis and more sophisticated modeling.
# # These steps will leave us poised to extract meaningful patterns from our corpus and
# # to use those patterns to make predictions about new, as-yet unseen data.
#
# # Text Vectorization and Transformation Pipelines
#
# # Machine learning algorithms operate on a numeric feature space, expecting input as a
# # two-dimensional array where rows are instances and columns are features. In order
# # to perform machine learning on text, we need to transform our documents into vector
# # representations such that we can apply numeric machine learning. This process is
# # called feature extraction or more simply, vectorization, and is an essential first step
# # toward language-aware analysis.
#
# # In
# text analysis, instances are entire documents or utterances, which can vary in length
# from quotes or tweets to entire books, but whose vectors are always of a uniform
# length. Each property of the vector representation is a feature. For text, features represent
# attributes and properties of documents—including its content as well as meta
# attributes, such as document length, author, source, and publication date. When considered
# together, the features of a document describe a multidimensional feature
# space on which machine learning methods can be applied
#
# For this reason, we must now make a critical shift in how we think about language—
# from a sequence of words to points that occupy a high-dimensional semantic space.
# Points in space can be close together or far apart, tightly clustered or evenly distributed.
# Semantic space is therefore mapped in such a way where documents with
# similar meanings are closer together and those that are different are farther apart. By
# encoding similarity as distance, we can begin to derive the primary components of
# documents and draw decision boundaries in our semantic space.
#
# # bag-of-words model
#
# The simplest encoding of semantic space is the bag-of-words model, whose primary
# insight is that meaning and similarity are encoded in vocabulary.
#
# To vectorize a corpus with a bag-of-words (BOW) approach, we represent every
# document from the corpus as a vector whose length is equal to the vocabulary of the
# corpus. We can simplify the computation by sorting token positions of the vector into
# alphabetical order,
#
# look vector.png
#
# What should each element in the document vector be? In the next few sections, we
# will explore several choices, each of which extends or modifies the base bag-of-words
# model to describe semantic space. We will look at four types of vector encoding—frequency,
# one-hot, TF–IDF, and distributed representations—


In the context of text data, feature analysis amounts to building an understanding of
what is in the corpus. For instance, how long are our documents and how big is our
vocabulary? What patterns or combinations of n-grams tell us the most about our
documents? For that matter, how grammatical is our text? Is it highly technical, composed
of many domain-specific compound noun phrases? Has it been translated from
another language? Is punctuation used in a predictable way?