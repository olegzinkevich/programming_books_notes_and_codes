# In this recipe, we are going to discuss how to find the similarity between
# two documents or text. There are many similarity metrics like Euclidian,
# cosine, Jaccard, etc. Applications of text similarity can be found in areas

# like spelling correction and data deduplication.
# Here are a few of the similarity measures:

# Cosine similarity: Calculates the cosine of the angle (cocinus of an angle)
# between the two vectors.

# Jaccard similarity: The score is calculated using the
# intersection or union of words.

# Jaccard Index = (the number in both sets) / (the
# number in either set) * 100.

# Levenshtein distance: Minimal number of
# insertions, deletions, and replacements required for
# transforming string “a” into string “b.”

# Hamming distance: Number of positions with the
# same symbol in both strings. But it can be defined
# only for strings with equal length.

# The simplest way to do this is by using cosine similarity from the sklearn
# library.

documents = ["I like NLP",
             "I like NLP",
             "I am a beginner in NLP",
             "I want to learn NLP",
             "I like advanced NLP"
             ]

#Import libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Compute tfidf : feature engineering(refer previous chapter – Recipe 3-4)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print(tfidf_matrix.shape)

#output
# (5, 10)

#compute similarity for first sentence with rest of the sentences
similarity_array = cosine_similarity(tfidf_matrix[0:1],tfidf_matrix)
print(cosine_similarity(tfidf_matrix[0:1],tfidf_matrix))
values = similarity_array.tolist()

# >> [[1.         0.17682765 0.14284054 0.13489366 0.68374784]]
# If we clearly observe, the first sentence and last sentence have higher
# similarity compared to the rest of the sentences.

#  converting list of list in values into flat list
final_values = []
for sublist in values:
    for item in sublist:
        final_values.append(item)
print(final_values)

import pandas as pd

df = pd.DataFrame()
df['values'] = final_values
df['documents'] = documents

#  sorting sentences based on similarity with descending order
df_sorted = df.sort_values(by=['values'], ascending=False)
print(df_sorted)
