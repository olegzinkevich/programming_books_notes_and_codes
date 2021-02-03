# Storing many small files on a cluster can be complex because of data transfer issues
# and namespace management, not to mention the space savings that compressing
# larger files can bring. As a result, it is generally better to store data in fewer, larger
# files rather than in many, smaller files.
# One common storage method for concatenating data into larger files is called JSON
# lines (also known as JSONL) where each line of the file is a serialized JSON object
# rather than the entire file. JSON lines can be loaded and parsed into an RDD as
# follows:

import json
corpus = sc.wholeTextFiles("corpus/*.jsonl")
corpus = corpus.flatMap(
    lambda d: [json.loads(line)
               for line in d[1].split("\n") if line ]
)

# todo соединить с corpus reader

# In this case, we map a function that returns an array of JSON objects, parsed from
# each line in the document. By using flatMap, the list of lists is flattened into a single
# list; therefore the RDD is a collection of Python dictionaries, each parsed from a line
# in every file in the dataset