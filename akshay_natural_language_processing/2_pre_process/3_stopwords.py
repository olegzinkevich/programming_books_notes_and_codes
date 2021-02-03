text=['This is introduction to NLP','It is likely to be useful, to people ','Machine learning is the new electrcity', 'There would be less hype around AI and more action going forward','python is the best tool!','R is good langauage','I like this book','I want more books like this']

#convert list to data frame
import pandas as pd

df = pd.DataFrame({'tweet':text})
print(df)

from nltk.corpus import stopwords
#remove stop words
stop = stopwords.words('english')
df['tweet'] = df['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
print(df['tweet'])