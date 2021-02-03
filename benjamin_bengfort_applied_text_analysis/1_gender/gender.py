#!/usr/bin/env python3

# gender analysis in text

#  можно использовать для анализа, какому бренду в тексте принадлежит большая роль.
# либо создать словари под эмоциональный уровень текста

import nltk
from collections import Counter

MALE = 'male'
FEMALE = 'female'
UNKNOWN = 'unknown'
BOTH = 'both'

MALE_WORDS = set(['guy','spokesman','chairman',"men's",'men','him',"he's",'his','boy','boyfriend','boyfriends','boys','brother','brothers','dad','dads','dude','father','fathers','fiance','gentleman','gentlemen','god','grandfather','grandpa','grandson','groom','he','himself','husband','husbands','king','male','man','mr','nephew','nephews','priest','prince','son','sons','uncle','uncles','waiter','widower','widowers' ])

FEMALE_WORDS = set([
    'heroine','spokeswoman','chairwoman',"women's",'actress','women',
    "she's",'her','aunt','aunts','bride','daughter','daughters','female',
    'fiancee','girl','girlfriend','girlfriends','girls','goddess',
    'granddaughter','grandma','grandmother','herself','ladies','lady',
    'mom','moms','mother','mothers','mrs','ms','niece','nieces',
    'priestess','princess','queens','she','sister','sisters','waitress',
    'widow','widows','wife','wives','woman'
])


def genderize(words):

    mwlen = len(MALE_WORDS.intersection(words))
    fwlen = len(FEMALE_WORDS.intersection(words))

    if mwlen > 0 and fwlen == 0:
        return MALE
    elif mwlen == 0 and fwlen > 0:
        return FEMALE
    elif mwlen > 0 and fwlen > 0:
        return BOTH
    else:
        return UNKNOWN


def count_gender(sentences):
    # We need a method for counting the frequency of gendered words and sentences
# within the complete text of an article, which we can do with the collections.Coun
# ters class, a built-in Python class. The count_gender function takes a list of sentences
# and applies the genderize function to evaluate the total number of gendered
# words and gendered sentences.
    sents = Counter()
    words = Counter()

    for sentence in sentences:
        gender = genderize(sentence)
        sents[gender] += 1
        words[gender] += len(sentence)

    return sents, words


def parse_gender(text):

    sentences = [
        [word.lower() for word in nltk.word_tokenize(sentence)]
        for sentence in nltk.sent_tokenize(text)
    ]

    sents, words = count_gender(sentences)
    total = sum(words.values())

    for gender, count in words.items():
        pcent = (count / total) * 100
        nsents = sents[gender]

        print(
            "{:0.3f}% {} ({} sentences)".format(pcent, gender, nsents)
        )

if __name__ == '__main__':
    with open('ballet.txt', 'r', encoding='UTF-8') as f:
        parse_gender(f.read())

# Each sentence’s gender is counted and all words in the sentence are also considered as belonging to that gender


# One might assume that sentiment analysis can
# be conducted with a technique similar to the gender analysis of the previous section:
# gather lists of positive words (“awesome,” “good,” “stupendous”) and negative words
# (“horrible,” “tasteless,” “bland”) and compute the relative frequencies of these tokens
# in their context. Unfortunately, this technique is naive and often produces highly
# inaccurate results.
# Sentiment analysis is fundamentally different from gender classification because sentiment
# is not a language feature, but instead dependent on word sense; for example,
# “that kick flip was sick” is positive whereas “the chowder made me sick” is negative