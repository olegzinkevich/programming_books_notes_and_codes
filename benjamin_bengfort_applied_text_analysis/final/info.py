# получаем базовую информацию о корпусе

from corpus_reader import HTMLCorpusReader
from pickled_corpus_reader import PickledCorpusReader

# html_corpus = HTMLCorpusReader('C:/Users/810004/Desktop/Html_corpus/')
# print(html_corpus.resolve(None, None))
# a = html_corpus.docs()
# for f in a:
#     print(f)
# # tokenize = html_corpus.tokenize()
# print(html_corpus.describe())

#
#
corpus = PickledCorpusReader('C:/Users/810004/Desktop/Html_corpus/')


for category in corpus.categories():
    n_docs = len(corpus.fileids(categories=[category]))
    n_words = sum(1 for word in corpus.words(categories=[category]))

    print("- '{}' contains {:,} docs and {:,} words".format(category, n_docs, n_words))