import numpy.random as rand
import numpy as np

# Creating a 100-element array with random values
# from a standard normal distribution or, in other
# words, a Gaussian distribution.
# The sigma is 1 and the mean is 0.
a = rand.randn(100)
print(a)

b = np.random.randint(100, size=10)
print(b)

# Generate a 2 x 4 array of ints between 0 and 4, inclusive:
c = np.random.randint(5, size=(2, 4))
print(c)

# Generate a 1 x 3 array with 3 different upper bounds
d = np.random.randint(1, [3, 5, 10])
print(d)

g = np.random.rand(1,100)
print(g)