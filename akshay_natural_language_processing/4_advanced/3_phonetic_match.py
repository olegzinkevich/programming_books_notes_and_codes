# Phonetic matching
# The next version of similarity checking is phonetic matching, which roughly
# matches the two words or sentences and also creates an alphanumeric
# string as an encoded version of the text or word. It is very useful for searching
# large text corpora, correcting spelling errors, and matching relevant names.
# Soundex and Metaphone are two main phonetic algorithms used for this
# purpose. The simplest way to do this is by using the fuzzy library.

# pip install fuzzy

import fuzzy

# Run the Soundex function
soundex = fuzzy.Soundex(4)

# Generate the phonetic form
print(soundex('natural'))
print(soundex('natuaral'))
# Soundex is treating “natural” and “natuaral” as the same, and the
# phonetic code for both of the strings is “N364.”