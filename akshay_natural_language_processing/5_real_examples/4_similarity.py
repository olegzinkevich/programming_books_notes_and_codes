# • Customer information scattered across multiple tables
# and systems.
# • No global key to link them all together.
# • A lot of variations in names and addresses.

# This can be solved by applying text similarity functions on the
# demographic’s columns like the first name, last name, address, etc. And
# based on the similarity score on a few common columns, we can decide
# either the record pair is a match or not a match.
#
# Huge records that need to be linked/stitched/ deduplicated.
#
# pip install recordlinkage
#
# The term record linkage is used to indicate the procedure of bringing together information from two or more records that are believed to belong to the same entity. Record linkage is used to link data from multiple data sources or to find duplicates in a single data source. In computer science, record linkage is also known as data matching or deduplication (in case of search duplicate records within a single file).

import recordlinkage

#For this demo let us use the inbuilt dataset from recordlinkage library

#import data set
from recordlinkage.datasets import load_febrl1

#create a dataframe - dfa
dfA = load_febrl1()
print(dfA.head())


# PArt 2 Blocking

# Here we reduce the comparison window and create record pairs.
# Why?
# • Suppose there are huge records say, 100M records
# means (100M choose 2) ≈ 10^16 possible pairs
# • Need heuristic to quickly cut that 10^16 down without
# losing many matches

# This can be accomplished by extracting a “blocking key” How?
# Example:

# • Record: first name: John, last name: Roberts, address:
# 20 Main St Plainville MA 01111
# • Blocking key: first name - John
# • Will be paired with: John Ray … 011
# • Won’t be paired with: Frank Sinatra … 07030
# • Generate pairs only for records in the same block

# Below is the blocking example at a glance: here blocking is done on the
# “Sndx-SN,” column

# There are many advanced blocking techniques, also, like the following:
# • Standard blocking
# • Single column
# • Multiple columns
# • Sorted neighborhood
# • Q-gram: fuzzy blocking
# • LSH
# • Canopy clustering

# but for now, let’s build the pairs using the first name as the blocking index.

indexer = recordlinkage.index.Block(left_on='given_name')
pairs = indexer.index(dfA)

print(len(pairs))

# Part 3 - similarity matching
# Here we compute the similarity scores on the columns like given name,
# surname, and address between the record pairs generated in the previous
# step.

# This cell can take some time to compute.
compare_cl = recordlinkage.Compare()

#  using method - Jarowinkler, but you can use some other

compare_cl.string('given_name', 'given_name',method='jarowinkler', label='given_name')
compare_cl.string('surname', 'surname', method='jarowinkler', label='surname')
compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
compare_cl.exact('suburb', 'suburb', label='suburb')
compare_cl.exact('state', 'state', label='state')
compare_cl.string('address_1', 'address_1',method='jarowinkler', label='address_1')

features = compare_cl.compute(pairs, dfA)
print(features.sample(5))

# So here record “rec-115-dup-0” is compared with “rec-120-dup-0.”
# Since their first name (blocking column) is matching, similarity scores are
# calculated on the common columns for these pairs.

# part 4:
# Predicting records match or do not match using ECM – classifier. Here is an unsupervised learning method to calculate the probability that the records match
# select all the features except for given_name since its our blocking key
features1 = features[['suburb','state','surname','date_of_birth','address_1']]

# Unsupervised learning – probabilistic

ecm = recordlinkage.ECMClassifier()
result_ecm = ecm.learn((features1).astype(int),return_type = 'series')

print(result_ecm)



