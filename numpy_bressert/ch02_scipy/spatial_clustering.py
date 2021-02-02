# From biological to astrophysical sciences, spatial and clustering analysis are key to identifying
# patterns, groups, and clusters. In biology, for example, the spacing of different
# plant species hints at howseeds are dispersed, interactwith the environment, and grow.
# In astrophysics, these analysis techniques are used to seek and identify star clusters,
# galaxy clusters, and large-scale filaments (composed of galaxy clusters).

# SciPy provides a spatial analysis class (scipy.spatial) and a cluster analysis class
# (scipy.cluster). The spatial class includes functions to analyze distances between data
# points (e.g., k-d trees). The cluster class provides two overarching subclasses: vector
# quantization (vq) and hierarchical clustering (hierarchy). Vector quantization groups large sets of data points (vectors) where each group is represented by centroids. The
# hierarchy subclass contains functions to construct clusters and analyze their substructures
#

# h Vector Quantization

# Vector quantization is a general termthat can be associated with signal processing, data
# compression, and clustering. Here we will focus on the clustering component, starting
# with how to feed data to the vq package in order to identify clusters.
import numpy as np
from scipy.cluster import vq
# Creating data
c1 = np.random.randn(100, 2) + 5
c2 = np.random.randn(30, 2) - 5
c3 = np.random.randn(50, 2)
# Pooling all the data into one 180 x 2 array
data = np.vstack([c1, c2, c3])
# Calculating the cluster centroids and variance
# from kmeans
centroids, variance = vq.kmeans(data, 3)
# The identified variable contains the information
# we need to separate the points in clusters
# based on the vq function.
identified, distance = vq.vq(data, centroids)
# Retrieving coordinates for points in each vq
# identified core
# The result of the identified clusters matches up quite well to the original data
vqc1 = data[identified == 0]
vqc2 = data[identified == 1]
vqc3 = data[identified == 2]
print(vqc1)
print(vqc2)
print(vqc3)

