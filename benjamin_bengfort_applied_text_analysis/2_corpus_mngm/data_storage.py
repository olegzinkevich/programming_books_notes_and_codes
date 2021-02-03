Data products often employ write-once, read-many (WORM) storage as an intermediate
data management layer between ingestion and preprocessing as shown in
Figure 2-2. WORM stores (sometimes referred to as data lakes) provide streaming
read accesses to raw data in a repeatable and scalable fashion, addressing the requirement
for performance computing. Moreover, by keeping data in a WORM store, preprocessed
data can be reanalyzed without reingestion, allowing new hypotheses to be
easily explored on the raw data format.

Relational database management systems are great for transactions
that operate on a small collection of rows at a time, particularly
when those rows are updated frequently. Machine learning on a
text corpus has a different computational profile: many sequential
reads of the entire dataset. As a result, storing corpora on disk (or
in a document database) is often preferred.

For text data management, the best choice is often to store data in a NoSQL document
storage database that allows streaming reads of the documents with minimal overhead, or to simply write each document to disk

# Text is also the most compressible format, making Zip files, which leverage directory
structures on disk, an ideal distribution and storage format. Finally, corpora stored
on disk are generally static and treated as a whole, fulfilling the requirement for
WORM storage presented in the previous section.

Tweets are generally small JSON data
structures that include not just the text of the tweet but other metadata like user or
location. The typical way to store multiple tweets is in newline-delimited JSON,
sometimes called the JSON lines format. This format makes it easy to read one tweet
at a time by parsing only a single line at a time, but also to seek to different tweets in
the file. A single file of tweets can be large, so organizing tweets in files by user, location,
or day can reduce overall file sizes and create a meaningful disk structure of
multiple files.

# If the documents are
categorized by sentiment, as positive or negative, each type of document can be grouped
together into their own category subdirectory. If there are multiple users in a system
that generate their own subcorpora of user-specific writing, such as reviews or
tweets, then each user can have their own subdirectory.


# The Baleen corpus ingestion engine writes an HTML corpus to disk as follows: corpus

├── citation.bib
├── feeds.json
├── LICENSE.md
├── manifest.json
├── README.md
└── books
| ├── 56d629e7c1808113ffb87eaf.html
| ├── 56d629e7c1808113ffb87eb3.html
| └── 56d629ebc1808113ffb87ed0.html

# There are a few important things to note here. First, all documents are stored as
HTML files, named according to their MD5 hash (to prevent duplication), and each
stored in its own category subdirectory

# In terms of meta information, a citation.bib (look above) file provides attribution for the corpus
and the LICENSE.md file specifies the rights others have to use this corpus. While
these two pieces of information are usually reserved for public corpora, it is helpful to
include them so that it is clear how the corpus can be used—for the same reason that
you would add this type of information to a private software repository

Of these files, citation.bib, LICENSE.md, and README.md are special files because
they can be automatically read from an NLTK CorpusReader object with the
citation(), license(), and readme() methods

A structured approach to corpus management and storage means that applied text
analytics follows a scientific process of reproducibility, a method that encourages the
interpretability of analytics as well as confidence in their results.





