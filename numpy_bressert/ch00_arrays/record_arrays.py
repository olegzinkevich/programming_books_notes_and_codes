# Arrays are generally collections of integers or floats, but sometimes it is useful to store
# more complex data structures where columns are composed of different data types.
# In research journal publications, tables are commonly structured so that some columns
# may have string characters for identification and floats for numerical quantities.
# Being able to store this type of information is very beneficial. In NumPy there is the
# numpy.recarray.

import numpy as np

# h Multiple dtypes

# The dtype optional argument is defining the types designated for the first to third
# columns, where i4 corresponds to a 32-bit integer, f4 corresponds to a 32-bit float,
# and a10 corresponds to a string 10 characters long.
# Creating an array of zeros and defining column types
recarr = np.zeros((2,), dtype=('i4,f4,a10'))
toadd = [(1,2.,'Hello'), (2,3.,"World")]
recarr[:] = toadd
print(recarr)

# Thankfully, in Python there is a global function called zip that will create a list of tuples like we see above for the toadd object. So we show how to use zip to populate the same # recarray. Creating an array of zeros and defining column types

# h Assigning names to each column, which

# are now by default called 'f0', 'f1', and 'f2'.
recarr.dtype.names = ('Integers' , 'Floats', 'Strings')

# print names of the columns
print(recarr.dtype.names)

# If we want to access one of the columns by its name, we
# can do the following.
print(recarr['Integers'])

# h slicing

print()
print('==========slicing==========')

# Python index lists begin at zero and theNumPy arrays follow suit. When indexing lists in Python, we normally do the following for a 2 × 2 object:
alist= [[1,2],[3,4]]
# To return the (0,1) element we must index as shown below.
print(alist[0][1])

# If we want to return the right-hand column, there is no trivial way to do so with Python     lists. In NumPy, indexing follows a more convenient syntax.
# Converting the list defined above into an array
arr = np.array(alist)
# To return the (0,1) element we use ...
print(arr[0,1])
# Now to access the last column, we simply use ...
print(arr[:,1])
# Accessing the columns is achieved in the same way,
# which is the bottom row.
print(arr[1,:])

# h indexing with np.where

# Sometimes there are more complex indexing schemes required, such as conditional
# indexing. The most commonly used type is numpy.where(). With this function you can return the desired indices from an array, regardless of its dimensions, based on some conditions(s).
#
# Creating an array
arr = np.arange(5)
# Creating the index array
index = np.where(arr > 2)
print(index)
# >> (array([3, 4]),)
# Creating the desired array
new_arr = arr[index]
print(new_arr)

# However, you may want to remove specific indices instead. To do this you can use
# numpy.delete(). The required input variables are the array and indices that you want
# to remove.
# We use the previous array
new_arr = np.delete(arr, index)
print(new_arr)
# Instead of using the numpy.where function, we can use a simple boolean array to return
# specific elements.
index = arr > 2
print(index)
# [False False True True True]
new_arr = arr[index]
print(new_arr)

# h reshape

a = [1,2,3]
b = np.array(a).reshape((1,3))
print(b)