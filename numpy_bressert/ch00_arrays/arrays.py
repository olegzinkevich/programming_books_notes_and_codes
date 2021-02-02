# But operating on the elements in a list can only be done through iterative
# loops, which is computationally inefficient in Python. The NumPy package enables
# users to overcome the shortcomings of the Python lists by providing a data storage
# object called ndarray

# The ndarray is similar to lists, but rather than being highly flexible by storing different
# types of objects in one list, only the same type of element can be stored in each column.
# For example, with a Python list, you could make the first element a list and the second
# another list or dictionary. With NumPy arrays, you can only store the same type of
# element, e.g., all elements must be floats, integers, or strings. Despite this limitation,
# ndarray wins hands down when it comes to operation times, as the operations are sped
# up significantly. Using the %timeit magic command in IPython, we compare the power
# of NumPy ndarray versus Python lists in terms of speed.

import numpy as np
import timeit

# Create an array with 10^7 elements.
arr = np.arange(1e7)
# Converting ndarray to list
larr = arr.tolist()
# Lists cannot by default broadcast,
# so a function is coded to emulate
# what an ndarray can do.

def list_times(alist, scalar):

    for i, val in enumerate(alist):
        alist[i] = val * scalar

    return alist

# Using IPython's magic timeit command

mycode = ''' 
def list_times(alist, scalar):

    for i, val in enumerate(alist):
        alist[i] = val * scalar

    return alist
'''

# timeit statement
print(timeit.timeit(setup = mycode,
                    stmt = mycode,
                    number = 3))

# Unlike the ndarray objects, matrix objects can and only will be two dimensional. This
# means that trying to construct a third or higher dimension is not possible. Here’s an
# example.

# h array creation

# First we create a list and then
# wrap it with the np.array() function.
alist = [1, 2, 3]
arr = np.array(alist)
print('np.array', arr)

# Creating an array of zeros with five elements
arr = np.zeros(5)
print('np.zeros:', arr)

# What if we want to create an array going from 0 to 100?
arr = np.arange(100)
print('np.arange 100:',)

# Or 10 to 100?
arr = np.arange(10,100)
print('np.arange 10,100:', arr)

# If you want 100 steps from 0 to 1...
arr = np.linspace(0, 1, 100)
print('np.linspace:', arr)

# Or if you want to generate an array from 1 to 10
# in log10 space in 100 steps...
arr = np.logspace(0, 1, 100, base=10.0)
print('np.logspace:', arr)

# Creating a 5x5 array of zeros (an image)
image = np.zeros((5,5))
print('np.zeros:', image)

# Creating a 5x5x5 cube of 1's
# The astype() method sets the array with integer elements.
cube = np.zeros((5,5,5)).astype(int) + 1
print('np.zeros Cube:', cube)

# Or even simpler with 16-bit floating-point precision...
cube = np.ones((5, 5, 5)).astype(np.float16)
print(cube)

# When generating arrays, NumPy will default to the bit depth of the Python environment.
# If you are working with 64-bit Python, then your elements in the arrays will
# default to 64-bit precision. This precision takes a fair chunk memory and is not always
# necessary. You can specify the bit depth when creating arrays by setting the data
# type parameter (dtype) to int, numpy.float16, numpy.float32, or numpy.float64. Here’s
# an example how to do it.

# Array of zero integers
arr = np.zeros(2, dtype=int)
print('array dtype int:', arr)

# Array of zero floats
arr = np.zeros(2, dtype=np.float32)
print('array dtype float32:', arr)

# h Reshaping arrays

print()
print('======== Reshaping array ===========')
print()

# Now that we have created arrays, we can reshape them in many other ways. If we have
# a 25-element array, we can make it a 5× 5 array, or we could make a 3-dimensional
# array from a flat array.

# Creating an array with elements from 0 to 999
arr1d = np.arange(1000)
print('1 dimensional:', arr1d)
# Now reshaping the array to a 10x10x10 3D array
arr3d = arr1d.reshape((10,10,10))
print('reshape to 3 dimensional:', arr3d)

# The reshape command can alternatively be called this way
arr3d = np.reshape(arr1d, (10, 10, 10))

# Inversely, we can flatten arrays
arr4d = np.zeros((10, 10, 10, 10))
arr1d = arr4d.ravel()
print(arr1d)


# h !!! copy function: arrays = the same data in memory

# Keep in mind that the restructured arrays above are just different views
# of the same data in memory. This means that if you modify one of the
# arrays, it will modify the others. For example, if you set the first element
# of arr1d from the example above to 1, then the first element of arr3d will
# also become 1. If you don’t want this to happen, then use the numpy.copy
# function to separate the arrays memory-wise

