# The readability-lxml library is an excellent resource for grappling with the high
# degree of variability in documents collected from the web. Readability-lxml is a
# Python wrapper for the JavaScript Readability experiment by Arc90. like Safari and Chrome offer a reading mode, Readability removes distractions from
# the content of the page, leaving just the text
#
# Given an HTML document, Readability employs a series of regular expressions to
# remove navigation bars, advertisements, page script tags, and CSS, then builds a new
# Document Object Model (DOM) tree, extracts the text from the original tree, and
# reconstructs the text within the newly restructured tree. In the following example,
# which extends our HTMLCorpusReader, we import two readability modules,
# Unparseable and Document, which we can use to extract and clean the raw HTML
# text for the first phase of our preprocessing workflow.


#  look /2_corpus_mngm/1_corpus_reader.py   - readability functions are added there

