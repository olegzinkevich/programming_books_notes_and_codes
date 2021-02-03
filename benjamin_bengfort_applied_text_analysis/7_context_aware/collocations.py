# записывает 4-х ngrams  в файл

# For now, let’s explore the discovery of significant quadgrams. Because finding and
# ranking n-grams for a large corpus can take a lot of time, it is a good practice to write
# the results to a file on disk. We’ll create a rank_quadgrams function that takes as input
# a corpus to read words from, as well as a metric from the QuadgramAssocMeasures,
# finds and ranks quadgrams, then writes the results as a tab-delimited file to disk

from nltk.collocations import QuadgramCollocationFinder
from nltk.metrics.association import QuadgramAssocMeasures

def rank_quadgrams(corpus, metric, path=None):
    """
    Find and rank quadgrams from the supplied corpus using the given
    association metric. Write the quadgrams out to the given path if
    supplied otherwise return the list in memory.
    """

    # Create a collocation ranking utility from corpus words.
    ngrams = QuadgramCollocationFinder.from_words(corpus.words())

    # Rank collocations by an association metric
    scored = ngrams.score_ngrams(metric)

    if path:
        with open(path, 'w') as f:
            f.write("Collocation\tScore ({})\n".format(metric.__name__))
            for ngram, score in scored:
                f.write("{}\t{}\n".format(repr(ngram), score))
    else:
        return scored


if __name__ == '__main__':

    from reader import PickledCorpusReader

    # NLTK’s QuadgramAssocMeasures class exposes a number of significance
# testing tools such as the student T test, Pearson’s Chi-square
# test, pointwise mutual information, the Poisson–Stirling measure,
# or even a Jaccard index.
    corpus = PickledCorpusReader('C:/Users/810004/Desktop/Html_corpus/')

    rank_quadgrams(
        corpus, QuadgramAssocMeasures.likelihood_ratio, "quadgrams.txt"
    )

    # # Group quadgrams by first word
    # prefixed = defaultdict(list)
    # for key, score in scored:
    #     prefixed[key[0]].append((key[1:], scores))
    #
    # # Sort keyed quadgrams by strongest association
    # for key in prefixed:
    #     prefixed[key].sort(key=itemgetter(1), reverse=True)
