# SciPy has two packages for cluster analysis with vector quantization (kmeans) and hierarchy.
# The kmeans method was the easier of the two for implementing and segmenting
#     data into several components based on their spatial characteristics

# DBSCAN algorithm is used in
# the following example. DBSCAN works by finding core points that have many data points
# within a given radius.Once the core is defined, the process is iteratively computed until
# there are no more core points definable within the maximum radius? This algorithm
# does exceptionally well compared to kmeans where there is noise present in the data

import numpy as np
import matplotlib.pyplot as mpl
from scipy.spatial import distance
from sklearn.cluster import DBSCAN

# Creating data
c1 = np.random.randn(100, 2) + 5
c2 = np.random.randn(50, 2)

# Creating a uniformly distributed background
u1 = np.random.uniform(low=-10, high=10, size=100)
u2 = np.random.uniform(low=-10, high=10, size=100)
c3 = np.column_stack([u1, u2])

# Pooling all the data into one 150 x 2 array
data = np.vstack([c1, c2, c3])

# Calculating the cluster with DBSCAN function.
# db.labels_ is an array with identifiers to the
# different clusters in the data.
db = DBSCAN(eps=0.95, min_samples=10).fit(data)
labels = db.labels_

# Retrieving coordinates for points in each
# identified core. There are two clusters
# denoted as 0 and 1 and the noise is denoted
# as -1. Here we split the data based on which
# component they belong to.
dbc1 = data[labels == 0]
dbc2 = data[labels == 1]
noise = data[labels == -1]
print(dbc1)
print(dbc2)
print(noise)