import re
import requests

#url you want to extract
url = 'https://www.gutenberg.org/files/2638/2638-0.txt'
raw = requests.get(url).text

def preprocess(sentence):
    return re.sub('[^A-Za-z0-9.]+' , ' ', sentence).lower()

processed_book = preprocess(raw)

# Count number of times "the" is appeared in the book
print(len(re.findall(r'the', processed_book)))
