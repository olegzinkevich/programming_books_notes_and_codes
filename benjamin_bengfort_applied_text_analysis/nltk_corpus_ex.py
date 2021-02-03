# corpus
# ├── LICENSE
# ├── README
# └── Star Trek
# | ├── Star Trek - Balance of Terror.txt
# | ├── Star Trek - First Contact.txt
# | ├── Star Trek - Generations.txt
# | ├── Star Trek - Nemesis.txt
# | ├── Star Trek - The Motion Picture.txt
# | ├── Star Trek 2 - The Wrath of Khan.txt
# | └── Star Trek.txt
# └── Star Wars
# | ├── Star Wars Episode 1.txt
# | ├── Star Wars Episode 2.txt
# | ├── Star Wars Episode 3.txt
# | ├── Star Wars Episode 4.txt
# | ├── Star Wars Episode 5.txt
# | ├── Star Wars Episode 6.txt
# | └── Star Wars Episode 7.txt
# └── citation.bib
#

from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader

DOC_PATTERN = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.txt'
CAT_PATTERN = r'([\w_\s]+)/.*'

corpus = CategorizedPlaintextCorpusReader('/path/to/corpus/root', DOC_PATTERN, cat_pattern=CAT_PATTERN)

corpus.categories()
# ['Star Trek', 'Star Wars']
corpus.fileids()
# ['Star Trek/Star Trek - Balance of Terror.txt',
# 'Star Trek/Star Trek - First Contact.txt', ...]
