# Most of the text data is in the form of either customer reviews, blogs, or tweets,
# where there is a high chance of people using short words and abbreviations to
# represent the same meaning.

# We can write our own custom dictionary to look for short words and
# abbreviations.

lookup_dict = {'nlp':'natural language processing',
'ur':'your', "wbu" : "what about you"}

import re

def text_std(input_text):
    words = input_text.split()
    new_words = []
    for word in words:
        word = re.sub(r'[^\w\s]','', word)
        if word.lower() in lookup_dict:
            word = lookup_dict[word.lower()]
            new_words.append(word)
            new_text = " ".join(new_words)
    return new_text

# it will change nlp > Natural language processing
print(text_std("I like nlp it's ur choice"))