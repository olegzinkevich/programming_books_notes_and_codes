# Here’s an example of how Python opens and parses text information.

# Opening the text file with the 'r' option,
# which only allows reading capability
f = open('data/somefile.txt', 'r')
# Parsing the file and splitting each line,
# which creates a list where each element of
# it is one line
alist = f.readlines()
# Closing file
f.close()

newdata = 'somesomesome'
# After a few operations, we open a new text file
# to write the data with the 'w' option. If there
# was data already existing in the file, it will be overwritten.
f = open('data/newtextfile.txt', 'w')
# Writing data to file
f.writelines(newdata)
# Closing file
f.close()


# h loadtxt

# Accessing and recording data this way can be very flexible and fast, but there is one
# downside: if the file is large, then accessing or modulating the data will be cumbersome
# and slow. Getting the data directly into a numpy.ndarray would be the best option. We
# can do this by using a NumPy function called loadtxt. If the data is structured with
# rows and columns, then the loadtxt commandwillwork verywell as long as all the data
# is of a similar type, i.e., integers or floats

import numpy as np

# from io import StringIO
# f = open('somefile.txt', 'r')
# # StringIO behaves like a file object
# c = StringIO(str(alist))
# print(c)

arr = np.loadtxt('data/somefile.txt', dtype=np.str)
print(arr)

# or use genfromtxt
print(np.genfromtxt('data/somefile.txt',dtype='str'))

# np.savetxt('somenewfile.txt')

# If each column is different in terms of formatting, loadtxt can still read the data, but
# the column types need to be predefined. The final construct from reading the data will be a recarray


# example.txt file looks like the following
#
# XR21 32.789 1
# XR22 33.091 2

table = np.loadtxt('data/example.txt',
                   dtype={'names': ('ID', 'Result', 'Type'),
                                  'formats': ('S4', 'f4', 'i2')})
print(table)
# array([('XR21', 32.78900146484375, 1),
# ('XR22', 33.090999603271484, 2)],
# dtype=[('ID', '|S4'), ('Result', '<f4'), ('Type', '<i2')])
# print names of the columns
print(table.dtype.names)
print(table['ID'])
# fmt = %s format in strings
np.savetxt('data/newtextfile.txt', table, fmt="%s")

with open('data/newtextfile2.txt',"w") as f:
    f.write("\n".join(" ".join(map(str, x)) for x in table))

# h binary files

# Binary files in retrospect are harder to deal
# with, as formatting, readability, and portability are trickier. Yet they have two notable
# advantages over text-based files: file size and read/write speeds. This is especially
# important when working with big data In NumPy, files can be accessed in binary format using numpy.save and numpy.load.
# The primary limitation is that the binary format is only readable to other systems that
# are using NumPy. If you want to read and write files in a more portable format, then
# scipy.io will do the job.

# Creating a large array
data = np.empty((1000, 1000))
print(data)

# Saving the array with numpy.save
np.save('data/test.npy', data)
# If space is an issue for large files, then
# use numpy.savez instead. It is slower than
# numpy.save because it compresses the binary
# file.
np.savez('data/test.npz', data)
# Loading the data array
newdata = np.load('data/test.npy')
print(newdata)