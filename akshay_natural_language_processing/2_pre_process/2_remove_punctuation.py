# This step is very important as punctuation doesn’t add any extra
# information or value. Hence removal of all such instances will help reduce the size of the data and increase computational efficiency.

# The simplest way to do this is by using the regex and replace() function in Python.

text=['This is introduction to NLP','It is likely to be useful, to people ','Machine learning is the new electrcity','There would be less hype around AI and more action going forward','python is the best tool!','R is good langauage','I like this book','I want more books like this']

#convert list to dataframe
import pandas as pd

df = pd.DataFrame({'tweet':text})
print(df)

import re
# Using the regex and replace() function, we can remove the punctuation as shown below:
s = "I. like. This book!"
s1 = re.sub(r'[^\w\s]','',s)
print(s1)

# or

import string

s = "I. like. This book!"
for c in string.punctuation:
s= s.replace(c,"")

# or with pandas

df['tweet'] = df['tweet'].str.replace('[^\w\s]','')
print(df['tweet'])

