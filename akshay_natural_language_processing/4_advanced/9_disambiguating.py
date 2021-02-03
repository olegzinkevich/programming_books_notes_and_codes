# There is ambiguity that arises due to a different meaning of words in a # different context.

# For example,
# Text1 = 'I went to the bank to deposit my money'
# Text2 = 'The river bank was full of dead fishes'
# In the above texts, the word “bank” has different meanings based on
# the context of the sentence.

# The Lesk algorithm is one of the best algorithms for word sense
# disambiguation. Let’s see how to solve using the package pywsd and nltk.

# pip install pywsd

#Import functions

from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from itertools import chain
from pywsd.lesk import simple_lesk

# Sentences

bank_sents = ['I went to the bank to deposit my money',
'The river bank was full of dead fishes']

# calling the lesk function and printing results for both the sentences

print ("Context-1:", bank_sents[0])
answer = simple_lesk(bank_sents[0],'bank')
print ("Sense:", answer)
print ("Definition : ", answer.definition())


print ("Context-2:", bank_sents[1])
answer = simple_lesk(bank_sents[1],'bank','n')
print ("Sense:", answer)
print ("Definition : ", answer.definition())