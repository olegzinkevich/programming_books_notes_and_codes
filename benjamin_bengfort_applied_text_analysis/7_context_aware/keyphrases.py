# KeyphraseExtrac tor class that will transform documents into a bag-of-keyphrase representation. Keyphrase extraction
# consists of identifying and isolating phrases of a dynamic size to capture as many nuances in the topics of documents as possible.

# работает

from nltk import ne_chunk
from itertools import groupby
from nltk.corpus import wordnet as wn
from nltk.chunk import tree2conlltags
from nltk.probability import FreqDist
from nltk.chunk.regexp import RegexpParser
from unicodedata import category as unicat
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

# The first step in keyphrase extraction is to identify candidates for phrases (e.g., which # words or phrases could best convey the topic or relationships of documents). We’ll # define our KeyphraseExtractor with a grammar and chunker to identify just the
# noun phrases using part-of-speech tagged text.

GRAMMAR = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
GOODTAGS = frozenset(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])
GOODLABELS = frozenset(['PERSON', 'ORGANIZATION', 'FACILITY', 'GPE', 'GSP'])


class KeyphraseExtractor(BaseEstimator, TransformerMixin):
    """
    Wraps a PickledCorpusReader consisting of pos-tagged documents.
    """
    def __init__(self, grammar=GRAMMAR):
        self.grammar = GRAMMAR
        self.chunker = RegexpParser(self.grammar)

    # Since we imagine that this KeyphraseExtractor will be the first step in a pipeline # after tokenization, we’ll add anormalize() method that performs some lightweight# text normalization, removing any punctuation and ensuring that all words are lowercase:
    def normalize(self, sent):
        """
        Removes punctuation from a tokenized/tagged sentence and
        lowercases words.
        """
        is_punct = lambda word: all(unicat(char).startswith('P') for char in word)
        sent = filter(lambda t: not is_punct(t[0]), sent)
        sent = map(lambda t: (t[0].lower(), t[1]), sent)
        return list(sent)

    def extract_keyphrases(self, document):
        """
        For a document, parse sentences using our chunker created by
        our grammar, converting the parse tree into a tagged sequence.
        Yields extracted phrases.
        """
#         # Given a document, this
# method will first normalize the text and then use our chunker to parse it. The output
# of a parser is a tree with only some branches of interest (the keyphrases!). To get the
# phrases of interest, we use the tree2conlltags function to convert the tree into the
# CoNLL IOB tag format, a list containing (word, tag, IOB-tag) tuples.
        for sents in document:
            for sent in sents:
                sent = self.normalize(sent)
                if not sent: continue
                chunks = tree2conlltags(self.chunker.parse(sent))
                # An IOB tag tells you how a term is functioning in the context of the phrase; the term # will either begin a keyphrase (B-KT), be inside a keyphrase (I-KT), or be outside a keyphrase # (O). Since we’re only interested in the terms that are part of a keyphrase, we’ll
# use the groupby() function from the itertools package in the standard library to write
# a lambda function that continues to group terms so long as they are not O:
                phrases = [
                    " ".join(word for word, pos, chunk in group).lower()
                    for key, group in groupby(
                        chunks, lambda term: term[-1] != 'O'
                    ) if key
                ]
                for phrase in phrases:
                    yield phrase

#     # Since our class is a transformer, we finish by adding a no-op fit method and a trans
# form method that calls extract_keyphrases() on each document in the corpus:
    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            yield list(self.extract_keyphrases(document))



# Similarly to our KeyphraseExtractor, we can create a custom feature extractor to
# transform documents into bags-of-entities. To do this we will make use of NLTK’s
# named entity recognition utility, ne_chunk, which produces a nested parse tree structure
# containing the syntactic categories as well as the part-of-speech tags contained in
# each sentence.

class EntityExtractor(BaseEstimator, TransformerMixin):
    # We begin by creating an EntityExtractor class that is initialized with a set of entity
# labels.
    def __init__(self, labels=GOODLABELS, **kwargs):
        self.labels = labels
#     # We then add a get_entities method that uses ne_chunk to get a syntactic
# parse tree for a given document. The method then navigates through the subtrees in
# the parse tree, extracting entities whose labels match our set (consisting of people’s
# names, organizations, facilities, geopolitical entities, and geosocial political entities).
    def get_entities(self, document):
        entities = []
        for paragraph in document:
            for sentence in paragraph:
                trees = ne_chunk(sentence)
                for tree in trees:
                    if hasattr(tree, 'label'):
                        if tree.label() in self.labels:
                            entities.append(
                                ' '.join([child[0].lower() for child in tree])
                                )
        return entities

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        for document in documents:
            yield self.get_entities(document)


if __name__ == '__main__':
    from reader import PickledCorpusReader

    corpus = PickledCorpusReader('C:/Users/810004/Desktop/Html_corpus/')
    docs = corpus.docs()

    phrase_extractor = KeyphraseExtractor()
    keyphrases = list(phrase_extractor.fit_transform(docs))
    print(keyphrases[0])

    # I believe you need to re-create the docs variable since it is a generator that gets exhausted in the KeyPhraseExtractor section previous
    docs = corpus.docs()
    entity_extractor = EntityExtractor()
    entities = list(entity_extractor.fit_transform(docs))
    print(entities[0])

# Unfortunately, grammar-based approaches, while very effective, do not always work.
# For one thing, they rely heavily on the success of part-of-speech tagging, meaning we
# must be confident that our tagger is correctly labeling nouns, verbs, adjectives, and
# other parts of speech.

# Grammar-based feature extraction is also somewhat inflexible, because we must
# begin by defining a grammar. It is often very difficult to know in advance which
# grammar pattern will most effectively capture the high-signal terms and phrases
# within a text.

# look next n-gram_extraction.py