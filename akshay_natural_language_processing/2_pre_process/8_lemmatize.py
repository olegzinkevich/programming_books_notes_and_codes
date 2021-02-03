# Lemmatization is a process of # extracting a root word by considering the vocabulary. For example, “good,” # “better,” or “best” is lemmatized into good.

# Lemmatization handles matching “car” to “cars” along with matching “car” to “automobile.”
# • Stemming handles matching “car” to “cars.”

text=['I like fishing','I eat fish','There are many fishes in pound', 'leaves and leaf']

#convert list to dataframe

import pandas as pd

df = pd.DataFrame({'tweet':text})

#Import library
from textblob import Word

#Code for lemmatize
df['tweet'] = df['tweet'].apply(lambda x: " ".join([Word(word).
lemmatize() for word in x.split()]))

df['tweet']

