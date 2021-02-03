# этот скрипт можно использовать для анализа

import pandas as pd

#Read the data
df = pd.read_csv('amazon_reviews.csv', nrows=50)
# Look at the top 5 rows of the data
print(df.head(5))

# Understand the data types of the columns
print(df.info())

# Looking at the summary of the reviews.
print(df.Summary.head(5))

# Looking at the description of the reviews
print(df.Text.head(5))


# Sentiment Analysis: Pretrained model takes the input from the text
# description and outputs the sentiment score ranging from -1 to +1 for each  sentence.

#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import sys
import ast
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

plt.style.use('fivethirtyeight')
# Function for getting the sentiment
cp = sns.color_palette()

analyzer = SentimentIntensityAnalyzer()

# Generating sentiment for all the sentence present in the dataset
emptyline=[]
for row in df['Text']:
    vs=analyzer.polarity_scores(row)
    emptyline.append(vs)

# Creating new dataframe with sentiments
df_sentiments=pd.DataFrame(emptyline)
print('Sentiments:', df_sentiments.head(5))

# Merging the sentiments back to reviews dataframe
df_c = pd.concat([df.reset_index(drop=True), df], axis=1)
print('---------------------------------', df_c.head(3))

# Convert scores into positive and negetive sentiments using some threshold
df_c['Sentiment'] = np.where(df_sentiments['compound'] >= 0 , 'Positive', 'Negative')
print(df_c.head(5))


# Let’s see how the overall sentiment is using the sentiment we generated
result=df_c['Sentiment'].value_counts()
result.plot(kind='bar', rot=0)
plt.show()

result[['Negative','Positive']].plot(kind='bar', rot=0)
plt.show()

# We can also group by-products, that is, sentiments by-products to
# understand the high-level customer feedback against products.

# Similarly, we can analyze sentiments by month using the time column
# and many other such attributes