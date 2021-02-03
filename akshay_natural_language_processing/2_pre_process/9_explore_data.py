#Execute below code to download dataset, if you haven’t already nltk.download().

#Importing data
import nltk
from nltk.corpus import webtext
# nltk.download('webtext')
wt_sentences = webtext.sents('firefox.txt')
wt_words = webtext.words('firefox.txt')

# Import Library for computing frequency
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string

# Count the number of words
len(wt_sentences)
len(wt_words)

frequency_dist = nltk.FreqDist(wt_words)
print(frequency_dist)

sorted_frequency_dist =sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)
print(sorted_frequency_dist)

# Let’s take the words only if their frequency is greater than 3.
large_words = dict([(k,v) for k,v in frequency_dist.items() if len(k)>3])

import matplotlib.pyplot as plt
frequency_dist = nltk.FreqDist(large_words)
print(frequency_dist)
frequency_dist.plot(50,cumulative=False)
plt.show()

#install library
# !pip install wordcloud

#build wordcloud
from wordcloud import WordCloud
wcloud = WordCloud().generate_from_frequencies(frequency_dist)

#plotting the wordcloud

plt.imshow(wcloud, interpolation='bilinear')
plt.axis("off")
plt.show()