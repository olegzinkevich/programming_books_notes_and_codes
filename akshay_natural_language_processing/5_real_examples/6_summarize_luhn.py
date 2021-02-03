# Feature-based text summarization

# Your feature-based text summarization methods will extract a feature from
# the sentence and check the importance to rank it. Position, length, term
# frequency, named entity, and many other features are used to calculate the
# score.

# Luhn’s Algorithm is one of the feature-based algorithms, and we will
# see how to implement it using the sumy library.

# Import the packages

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.luhn import LuhnSummarizer

# Extracting and summarizing
LANGUAGE = "english"
SENTENCES_COUNT = 10

url = "https://en.wikipedia.org/wiki/Natural_language_processing"

parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
summarizer = LsaSummarizer()
summarizer = LsaSummarizer(Stemmer(LANGUAGE))
summarizer.stop_words = get_stop_words(LANGUAGE)
for sentence in summarizer(parser.document, SENTENCES_COUNT):
    print(sentence)