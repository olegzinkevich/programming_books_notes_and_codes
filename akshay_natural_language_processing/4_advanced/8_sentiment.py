# The simplest way to do this by using a TextBlob or vedar library.

# Let’s follow the steps in this section to do sentiment analysis using
# TextBlob. It will basically give 2 metrics.
# • Polarity = Polarity lies in the range of [-1,1] where 1
# means a positive statement and -1 means a negative
# statement.
# • Subjectivity = Subjectivity refers that mostly it is a
# public opinion and not factual information [0,1].

review = "I like this phone. screen quality and camera clarity is really good."
review2 = "This tv is not good. Bad quality, no clarity, worst experience"

# preprocess data, here it's not done

#import libraries
from textblob import TextBlob

#TextBlob has a pre trained sentiment prediction model
blob = TextBlob(review)
print(blob.sentiment)

#now lets look at the sentiment of review2
blob = TextBlob(review2)
print(blob.sentiment)

# #output
# Review 1: It seems like a very positive review.
# Sentiment(polarity=0.7, subjectivity=0.6000000000000001)
#
# Review 2: Negative -0.68
# Sentiment(polarity=-0.6833333333333332,
# subjectivity=0.7555555555555555)

