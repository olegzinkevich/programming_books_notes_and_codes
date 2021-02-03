#  stemming - extacting root of the word


text=['I like fishing','I eat fish','There are many fishes in pound']

#convert list to dataframe
import pandas as pd

df = pd.DataFrame({'tweet':text})
print(df)

from nltk.stem import PorterStemmer

st = PorterStemmer()
print(df['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()])))
