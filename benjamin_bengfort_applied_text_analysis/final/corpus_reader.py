#!/usr/bin/env python3

# look also - preprocess.py  in the folder 3_corpus_preprocess -
# Preprocessor that takes our HTMLCorpusReader, executes
# the preprocessing steps, and writes out a new text corpus to disk

import os
import codecs
import sqlite3

# Our custom corpus reader now knows how to deal with individual documents in the
# corpus, one document at a time, allowing us to filter and seek to different places in
# the corpus. It can handle fileids and categories, and has all the tools imported from
# NLTK to make disk access easier.


import bs4
import time
import nltk
import pickle
import logging

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader

# pip install readability-lxml
from readability.readability import Unparseable
from readability.readability import Document as Paper

from nltk import pos_tag, sent_tokenize, wordpunct_tokenize

log = logging.getLogger("readability.readability")
log.setLevel('WARNING')

DOC_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.json'
PKL_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.pickle'
CAT_PATTERN = r'([a-z_\s]+)/.*'

TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']


# Our HTMLCorpusReader class extends both the CategorizedCorpusReader and the CorpusReader
class HTMLCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
    A corpus reader for raw HTML documents to enable preprocessing.
    """

    def __init__(self, root, fileids=DOC_PATTERN, encoding='utf8',
                 tags=TAGS, **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining
        arguments are passed to the ``CorpusReader`` constructor.
        """
        # Add the default category pattern if not passed into the class.
        if not any(key.startswith('cat_') for key in kwargs.keys()):
            kwargs['cat_pattern'] = CAT_PATTERN

        # Multiple inheritance can by tricky, so the bulk of the code in the
# __init__ function simply figures out which arguments to pass to which class. In particular,
# the CategorizedCorpusReader takes in generic keyword arguments, and the
# CorpusReader will be initialized with the root directory of the corpus, as well as the
# fileids and the HTML encoding scheme.
        # Initialize the NLTK corpus reader objects
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids, encoding)

        # Save the tags that we specifically want to extract.
        self.tags = tags

    # The next step is to augment the HTMLCorpusReader with a method that will allow us
# to filter how we read text data from disk, either by specifying a list of categories, or a
# list of filenames:
    def resolve(self, fileids, categories):
        """
        Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function. Implemented similarly to
        the NLTK ``CategorizedPlaintextCorpusReader``.
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        # Note that categories can either be a single category or a list of
# categories. Otherwise, we will simply return the fileids—if this is None, the Corpus
# Reader will automatically read every single document in the corpus without filtering
        if categories is not None:
            return self.fileids(categories)
        return fileids

    # we will want to parse one HTML document at a time, so the following method gives
# us access to the text on a document-by-document basis:

    def docs(self, fileids=None, categories=None):
        """
        Returns the complete text of an HTML document, closing the document
        after we are done reading it and yielding it in a memory safe fashion.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=encoding) as f:
                yield f.read()


    # corpus monitoring section
    # To begin, we should consider what specific kinds of information we would like to monitor, such as the dates and sources of ingestion. Given the massive size of the corpora with which we will be working, we should at the very least, keep track of the size of each file on disk.  The above sizes method is in
# part a reaction to these kinds of experiences with real-world corpora, and will help us
# to perform diagnostics and identify individual files within the corpus that are much
# larger than expected (e.g., images and video that have been encoded as text).This
# method will enable us to compute the complete size of the corpus, to track over time,
# and see how it is growing and changing.

    def sizes(self, fileids=None, categories=None):
        """
        Returns a list of tuples, the fileid and size on disk of the file.
        This function is used to detect oddly large files in the corpus.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, getting every path and computing filesize
        for path in self.abspaths(fileids):
            yield os.path.getsize(path)

   # The html method iterates over each file and uses the summary method from the readability.Document class to remove any nontext content as well as script and stylistic tags. It also corrects any of the most commonly misused tags (e.g., <div> and
# <br>), only throwing an exception if the original HTML is found to be unparseable. # The most likely reason for such an exception is if the function is passed an empty # document, which has nothing to parse:

    # Note that the above method may generate warnings about the readability logger;
# you can adjust the level of verbosity according to your taste by adding:
# import logging
# log = logging.getLogger("readability.readability")
# log.setLevel('WARNING')

    def html(self, fileids=None, categories=None):
        """
        Returns the HTML content of each document, cleaning it using
        the readability-lxml library.
        """
        for doc in self.docs(fileids, categories):
            try:
                yield Paper(doc).summary()
            except Unparseable as e:
                print("Could not parse HTML: {}".format(e))
                continue


    # Our text, however, is not plain text, so we will need to create a method that extracts
# the paragraphs from HTML. This means we can isolate content that appears within paragraphs
# by searching for <p> tags, the element that formally defines an HTML paragraph.
# Because content can also appear in other ways (e.g., embedded inside other
# structures within the document like headings and lists), we will search broadly
# through the text using BeautifulSoup.  We will define a paras() method to iterate through each fileid and pass each HTML document to the BeautifulSoup constructor, specifying that the HTML should be parsed using the lxml HTML parser. The resulting soup is a nested tree structure
# that we can navigate using the original HTML tags and elements. For each of our
# document soups, we then iterate through each of the tags from our predefined set
# and yield the text from within that tag.

    # If passed a specific fileid, paras will return the paragraphs from that file only.
    def paras(self, fileids=None, categories=None):
        """
        Uses BeautifulSoup to parse the paragraphs from the HTML.
        """
        for html in self.html(fileids, categories):
            soup = bs4.BeautifulSoup(html, 'lxml')
            for element in soup.find_all(self.tags):
                # The result of our paras() method is a generator with the raw text paragraphs from every document, from first to last, with no document boundaries.
                yield element.text
            #     We can then call BeautifulSoup’s decompose  method to destroy the tree when we’re done working with each file to free up memory.
            soup.decompose()

    # In this section, we’ll perform segmentation to parse our text into sentences, which will
    # facilitate the part-of-speech tagging methods. It uses method (above) paras() and returns a generator (an iterator) yielding each sentence from every paragraph.
    def sents(self, fileids=None, categories=None):
        """
        Uses the built in sentence tokenizer to extract sentences from the
        paragraphs. Note that this method uses BeautifulSoup to parse HTML.
        """
        for paragraph in self.paras(fileids, categories):
            # NLTK’s PunktSentenceTokenizer is trained on English text, and it works well for
            # most European languages. It performs well when provided standard paragraphs. However, punctuation marks can be ambiguous; while periods frequently signal the
            # end of a sentence, they can also appear in floats, abbreviations, and ellipses. In other
            # words, identifying the boundaries between sentences can be tricky. As a result, you
            # may find that using PunktSentenceTokenizer on nonstandard text will not always
            # produce usable results. NLTK does provide alternative sentences tokenizers (e.g., for tweets), which are worth exploring.
            for sentence in sent_tokenize(paragraph):
                yield sentence

    # Tokenization is the process by which we’ll arrive at those tokens, and we’ll use Word
    # PunctTokenizer, a regular expression–based tokenizer that splits text on both whitespace
    # and punctuation and returns a list of alphabetic and nonalphabetic characters
    def words(self, fileids=None, categories=None):
        """
        Uses the built in word tokenizer to extract tokens from sentences.
        Note that this method uses BeautifulSoup to parse HTML content.
        """
        for sentence in self.sents(fileids, categories):
            # We can select different tokenizers depending on our responses to these questions. Of # the many word tokenizers available in NLTK (e.g., TreebankWordTokenizer, Word
            # PunctTokenize, PunktWordTokenizer, etc.), a common choice for tokenization is
            # word_tokenize, which invokes the Treebank tokenizer and uses regular expressions
            # to tokenize text as in Penn Treebank. This includes splitting standard contractions
            # (e.g., “wouldn’t” becomes “would” and “n’t”) and treating punctuation marks (like
            # commas, single quotes, and periods followed by whitespace) as separate tokens. By
            # contrast, WordPunctTokenizer is based on the RegexpTokenizer class, which splits
            # strings using the regular expression \w+|[^\w\s]+ , matching either tokens or separators
            # between tokens and resulting in a sequence of alphabetic and nonalphabetic
            # characters. You can also use the RegexpTokenizer class to create your own custom
            # tokenizer
            for token in wordpunct_tokenize(sentence):
                yield token

    # The tokenize method returns a generator that can give us a list of lists containing
    # paragraphs, which are lists of sentences, which in turn are lists of part-of-speech tagged
    # tokens. The tagged tokens are represented as (tag, token) tuples. List of tags: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    def tokenize(self, fileids=None, categories=None):
        """
        Segments, tokenizes, and tags a document in the corpus.
        """
        for paragraph in self.paras(fileids=fileids):
            yield [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(paragraph)
            ]

    # We can now add a new method, describe(), which
    # will allow us to perform intermediate corpus analytics on its changing categories,
    # vocabulary, and complexity. First, describe() will start the clock and initialize two frequency distributions: the first, counts, to hold counts of the document substructures, and the second, tokens, to contain the vocabulary. We’ll keep a count of each paragraph, sentence, and word, and we’ll also store each unique token in our vocabulary. We then compute
    # the number of files and categories in our corpus, and return a dictionary with a statistical
    # summary of our corpus—its total number of files and categories; the total number
    # of paragraph, sentences, and words; the number of unique terms; the lexical
    # diversity, which is the ratio of unique terms to total words; the average number of
    # paragraphs per document; the average number of sentences per paragraph; and the
    # total processing time

    # As our corpus grows through ingestion, preprocessing, and compression, describe()
    # allows us to recompute these metrics to see how they change over time.
    def describe(self, fileids=None, categories=None):
        """
        Performs a single pass of the corpus and
        returns a dictionary with a variety of metrics
        concerning the state of the corpus.
        """
        started = time.time()

        # Structures to perform counting.
        counts = nltk.FreqDist()
        tokens = nltk.FreqDist()

        # Perform single pass over paragraphs, tokenize and count
        for para in self.paras(fileids, categories):
            counts['paras'] += 1

            for sent in sent_tokenize(para):
                counts['sents'] += 1

                for word in wordpunct_tokenize(sent):
                    counts['words'] += 1
                    tokens[word] += 1

        # Compute the number of files and categories in the corpus
        n_fileids = len(self.resolve(fileids, categories) or self.fileids())
        n_topics = len(self.categories(self.resolve(fileids, categories)))

        # Return data structure with information
        return {
            'files': n_fileids,
            'topics': n_topics,
            'paras': counts['paras'],
            'sents': counts['sents'],
            'words': counts['words'],
            'vocab': len(tokens),
            'lexdiv': float(counts['words']) / float(len(tokens)),
            'ppdoc': float(counts['paras']) / float(n_fileids),
            'sppar': float(counts['sents']) / float(counts['paras']),
            'secs': time.time() - started,
        }




# We want our SqliteCorpusReader to be able to fetch results from the database in a
# memory safe fashion; as with the HTMLCorpusReader from the previous section, we
# need to be able access one record at a time to perform wrangling, normalization, and
# transformation

# Instead, our ids(), scores(), and texts() methods each make use of
# fetchone(), a good alternative in this case, though with a larger database, batch-wise
# fetching (e.g., with fetchmany()) would be more performant.

class SqliteCorpusReader(object):

    def __init__(self, path):
        self._cur = sqlite3.connect(path).cursor()

    def scores(self):
        """
        Returns the review score
        """
        self._cur.execute("SELECT score FROM reviews")
        for score in iter(self._cur.fetchone, None):
            yield score

    def texts(self):
        """
        Returns the full review texts
        """
        self._cur.execute("SELECT content FROM content")
        for text in iter(self._cur.fetchone, None):
            yield text

    def ids(self):
        """
        Returns the review ids
        """
        self._cur.execute("SELECT reviewid FROM content")
        for idx in iter(self._cur.fetchone, None):
            yield idx


# In this section we’ll write a Preprocessor that takes our HTMLCorpusReader, executes
# the preprocessing steps, and writes out a new text corpus to disk,
#
# # We begin by defining a new class, Preprocessor, which will wrap our corpus reader
# and manage the stateful tokenization and part-of-speech tagging of our documents.
#
# The objects will be initialized with a corpus, the path to the raw corpus, and target,
# the path to the directory where we want to store the postprocessed corpus. The
# fileids() method will provide convenient access to the fileids of the HTMLCorpus
# Reader object, and abspath() will returns the absolute path to the target fileid for
# each raw corpus fileid


import os
import nltk
import pickle

from nltk import pos_tag, sent_tokenize, wordpunct_tokenize


class Preprocessor(object):
    """
    The preprocessor wraps a corpus object (usually a `HTMLCorpusReader`)
    and manages the stateful tokenization and part of speech tagging into a
    directory that is stored in a format that can be read by the
    `HTMLPickledCorpusReader`. This format is more compact and necessarily
    removes a variety of fields from the document that are stored in the JSON
    representation dumped from the Mongo database. This format however is more
    easily accessed for common parsing activity.
    """

    def __init__(self, corpus, target=None, **kwargs):
        """
        The corpus is the `HTMLCorpusReader` to preprocess and pickle.
        The target is the directory on disk to output the pickled corpus to.
        """
        self.corpus = corpus
        self.target = target

    def fileids(self, fileids=None, categories=None):
        """
        Helper function access the fileids of the corpus
        """
        fileids = self.corpus.resolve(fileids, categories)
        if fileids:
            return fileids
        return self.corpus.fileids()

    def abspath(self, fileid):
        """
        Returns the absolute path to the target fileid from the corpus fileid.
        """
        # Find the directory, relative from the corpus root.
        parent = os.path.relpath(
            os.path.dirname(self.corpus.abspath(fileid)), self.corpus.root
        )

        # Compute the name parts to reconstruct
        basename  = os.path.basename(fileid)
        name, ext = os.path.splitext(basename)

        # Create the pickle file extension
        basename  = name + '.pickle'

        # Return the path to the file relative to the target.
        return os.path.normpath(os.path.join(self.target, parent, basename))


    # Next, we add a tokenize() method to our Preprocessor, which, given a raw document, # will perform segmentation, tokenization, and part-of-speech tagging using the NLTK methods. This method will return a generator of paragraphs for each document that contains a list of sentences, which are in turn lists of part-of-speech tagged tokens:
    def tokenize(self, fileid):
        """
        Segments, tokenizes, and tags a document in the corpus. Returns a
        generator of paragraphs, which are lists of sentences, which in turn
        are lists of part of speech tagged words.
        """
        for paragraph in self.corpus.paras(fileids=fileid):
            yield [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(paragraph)
            ]

# we are adding much more content (tokens, pos info, etc) to the original text than we are removing.
    # For this reason, we should be prepared to apply a compression method
# to keep disk storage under control. Pickle
# There are several options for transforming and saving a preprocessed corpus, but our
# preferred method is using pickle. With this approach we write an iterator that loads
# one document into memory at a time, converts it into the target data structure, and
# dumps a string representation of that structure to a small file on disk. While the
# resulting string representation is not human readable, it is compressed, easier to load,
# serialize and deserialize, and thus fairly efficient.

    # Once we have
# established a place on disk to retrieve the original files and to store their processed,
# pickled, compressed counterparts, we create a temporary document variable that creates
# our list of lists of lists of tuples data structure.
    def process(self, fileid):
        """
        For a single file does the following preprocessing work:
            1. Checks the location on disk to make sure no errors occur.
            2. Gets all paragraphs for the given text.
            3. Segements the paragraphs with the sent_tokenizer
            4. Tokenizes the sentences with the wordpunct_tokenizer
            5. Tags the sentences using the default pos_tagger
            6. Writes the document as a pickle to the target location.
        This method is called multiple times from the transform runner.
        """
        # Compute the outpath to write the file to.
        target = self.abspath(fileid)
        parent = os.path.dirname(target)

        # Make sure the directory exists
        if not os.path.exists(parent):
            os.makedirs(parent)

        # Make sure that the parent is a directory and not a file
        if not os.path.isdir(parent):
            raise ValueError(
                "Please supply a directory to write preprocessed data to."
            )

        # Create a data structure for the pickle
        document = list(self.tokenize(fileid))

        # Open and serialize the pickle to disk
        with open(target, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)

        # Then, after we serialize the document
# and write it to disk using the highest compression option, we delete it (document) before
# moving on to the next file to ensure that we are not holding extraneous content in
# memory:
        # Clean up the document
        del document

        # Return the target fileid
        return target


    # Our preprocess() method will be called multiple times by the following trans
# form() runner:
    def transform(self, fileids=None, categories=None):
        """
        Transform the wrapped corpus, writing out the segmented, tokenized,
        and part of speech tagged corpus as a pickle to the target directory.
        This method will also directly copy files that are in the corpus.root
        directory that are not matched by the corpus.fileids().
        """
        # Make the target directory if it doesn't already exist
        if not os.path.exists(self.target):
            os.makedirs(self.target)

        # Resolve the fileids to start processing and return the list of
        # target file ids to pass to downstream transformers.
        return [
            self.process(fileid)
            for fileid in self.fileids(fileids, categories)
        ]




if __name__ == '__main__':

    from collections import Counter
    from preprocess import Preprocessor

    # corpus = PickledCorpusReader('C:/Users/810004/Desktop/Baleen_corpus')
    # words  = Counter(corpus.words())

    #
    # print("{:,} vocabulary {:,} word count".format(len(words.keys()), sum(words.values())))

    # принимает файлы в формате json, сохраненные из html.
    # развания файлов должы быть захешированы!
    create_corpus = HTMLCorpusReader('C:/Users/810004/Desktop/Html_corpus/raw_html')
    # print(create_corpus.resolve(None, 'fashion'))
    a = create_corpus.docs()
    for f in a:
        print(f)
    tokenize = create_corpus.tokenize()
    print(create_corpus.describe())

    preprocess = Preprocessor(create_corpus, 'C:/Users/810004/Desktop/Html_corpus/raw_pickled_corpus')
    preprocess.transform()





