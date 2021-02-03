text=['Introduction to NLP','It is likely to be useful, to people ','Machine learning is the new electrcity', 'R is good langauage','I like this book','I want more books like this']

#convert list to dataframe
import pandas as pd

df = pd.DataFrame({'tweet':text})
print(df)



# !pip install autocorrect

from autocorrect import spell

print(spell(u'mussage'))
print(spell(u'sirvice'))

