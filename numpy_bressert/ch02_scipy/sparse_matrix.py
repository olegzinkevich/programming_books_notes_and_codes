# h sparse matrices (разреженная матрица)

# WithNumPy we can operatewith reasonable speeds on arrays containing 106 elements.
# Oncewe go up to 107 elements, operations can start to slowdown and Python’smemory
# will become limited, depending on the amount of RAM available. What’s the best
# solution if you need to work with an array that is far larger—say, 1010 elements? If
# these massive arrays primarily contain zeros, then you’re in luck, as this is the property
# of sparse matrices. If a sparse matrix is treated correctly, operation time and memory
# usage can go down drastically

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import scipy.sparse
import time
N = 3000
# Creating a random sparse matrix
m = scipy.sparse.rand(N, N)
print(m)

# Creating an array clone of it
a = m.toarray()
print('The numpy array data size: ' + str(a.nbytes) + ' bytes')
print('The sparse matrix data size: ' + str(m.data.nbytes) + ' bytes')

# Non-sparse
t0 = time.time()
res1 = eigh(a)
dt = str(np.round(time.time() - t0, 3)) + ' seconds'
print('Non-sparse operation takes ' + dt)

# Sparse
t0 = time.time()
res2 = eigsh(m)
dt = str(np.round(time.time() - t0, 3)) + ' seconds'
print('Sparse operation takes ' + dt)

# The memory allotted to the NumPy array and sparse matrix were 68MB and 0.68MB,
# respectively. In the same order, the times taken to process the Eigen commands were
# 36.6 and 0.2 seconds onmy computer.