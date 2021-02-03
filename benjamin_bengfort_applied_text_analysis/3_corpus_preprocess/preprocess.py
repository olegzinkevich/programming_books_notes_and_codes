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

preprocess = Preprocessor('C:/Users/810004/Desktop/Html_corpus/', 'C:/Users/810004/Desktop/Html_corpus/pickled_corpus')