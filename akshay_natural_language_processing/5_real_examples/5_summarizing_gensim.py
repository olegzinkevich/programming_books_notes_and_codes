# This # can be done using different techniques like the following:
# • TextRank: A graph-based ranking algorithm
# • Feature-based text summarization
# • LexRank: TF-IDF with a graph-based algorithm
# • Topic based
# • Using sentence embeddings
# • Encoder-Decoder Model: Deep learning techniques
#
# TextRank is the graph-based ranking algorithm for NLP. It is basically
# inspired by PageRank, which is used in the Google search engine but
# particularly designed for text. It will extract the topics, create nodes out of
# them, and capture the relation between nodes to summarize the text.
# Let’s see how to do it using the Python package Gensim. “Summarize”
# is the function used.

# Import BeautifulSoup and urllib libraries to fetch data from Wikipedia.
from bs4 import BeautifulSoup
from urllib.request import urlopen

# Function to get data from Wikipedia

def get_only_text(url):
    page = urlopen(url)
    soup = BeautifulSoup(page)
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    print (text)
    return soup.title.text, text

# Mention the Wikipedia url
url='https://en.wikipedia.org/wiki/Natural_language_processing'

# Call the function created above
text = get_only_text(url)

# Count the number of letters
len(''.join(text))
# Lets see first 1000 letters from the text
print(text[:1000])


# Import summarize from gensim
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

# Convert text to string format
text = str(text)

#Summarize the text with ratio 0.1 (10% of the total words.)
print(summarize(text, ratio=0.1))

#keywords
print(keywords(text, ratio=0.1))

#  it can be improved - look next - 6_summarizing_luhn.py