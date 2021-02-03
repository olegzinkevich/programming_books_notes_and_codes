In this chapter, we propose a multipurpose preprocessing framework that can be used
to systematically transform our raw ingested text into a form that is ready for computation
and modeling. Our framework includes the five key stages shown in
Figure 3-1: content extraction, paragraph blocking, sentence segmentation, word
tokenization, and part-of-speech tagging.

# In the previous chapter, we began constructing a custom HTMLCorpusReader, providing
it with methods for filtering, accessing, and counting our documents (resolve(),
docs(), and sizes()). Because it inherits from NLTK’s CorpusReader object, our
custom corpus reader also implements a standard preprocessing API that also exposes
the following methods:

raw()
Provides access to the raw text without preprocessing

sents()
A generator of individual sentences in the text

words()
Tokenizes the text into individual words


# We’ve defined sentences as the units of discourse and paragraphs as the units of document
structure. In this section, we will isolate tokens, the syntactic units of language
that encode semantic information within sequences of characters.



# The end result of the steps described in
Chapters 2 and 3 is a collection of files stored in a structured manner
on disk—one document to a file, stored in directories named
after their class. Each document is a pickled Python object composed
of several nested list objects—for example, the document is
a list of paragraphs, each paragraph is a list of sentences, and
each sentence is a list of (token, tag) tuples.